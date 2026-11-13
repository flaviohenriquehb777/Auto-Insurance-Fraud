"""Contrato de dados na fronteira do app — Fase 6, Parte 1 (camada 2 da vírgula).

Por que este módulo existe
--------------------------
A "falha silenciosa da vírgula" é o pior modo de falha possível num app de
triagem: `valor_reclamado` chega como texto pt-BR (`"8837,07"`), uma conversão
descuidada devolve `NaN` **sem levantar exceção**, o Pipeline imputa a mediana do
treino e devolve uma predição **normal, com o número errado**. Ninguém vê o erro —
o sinistro só entra na fila com a prioridade trocada.

A blindagem tem três camadas:

1. **Porta única** — todo dado passa por `pipeline.predict` (existe desde a Fase 3).
2. **Contrato na fronteira** (este módulo) — o dado é validado **antes** de chegar
   ao modelo; o que não converte é **recusado com erro explícito**, nunca imputado.
3. **Alarme no silêncio** — `quilha.monitoramento.resumo_imputacoes` mede a fração
   imputada por feature e denuncia o que passou pela fresta.

Regra de ouro desta camada
--------------------------
**Se o número não converte, o payload é recusado.** O app mostra o erro e pede a
correção; ele nunca completa o dado faltante por conta própria. Imputar na
fronteira seria repetir exatamente a falha que esta camada existe para impedir.

O que é validado (e o que não é)
--------------------------------
Só é validado aquilo de que o modelo **se alimenta**: as 14 colunas de abertura,
o domínio das categóricas, e a conversão numérica/data/hora. O minuto de
`hora_ocorrencia`, por exemplo, **não** entra no modelo (só a hora é usada em
`preparar`), então não é objeto do contrato — recusar por ele seria criar atrito
sem proteger nada.

Data: por que só `AAAA-MM-DD`
-----------------------------
`pd.to_datetime(..., format="mixed")` — o parser usado em `preparar` — aceita
**qualquer** texto plausível e resolve o resto pelo calendário de hoje:

* `"13:53"` (uma hora trocada de campo) vira **a data de hoje às 13:53** — sem
  erro, sem nulo, e o `atraso_abertura_dias` do sinistro sai do tamanho do
  intervalo entre a ocorrência e o dia da execução;
* `"03/04/2026"` é ambíguo e o pandas decide **mês primeiro** (`3 de abril` vira
  `4 de março`), enquanto quem digitou quis dizer o contrário.

Um contrato que aceitasse esses textos entregaria ao modelo uma data errada com
cara de data certa. Por isso a fronteira aceita **apenas** data explícita: texto
ISO `AAAA-MM-DD` ou um objeto de data de verdade (`datetime.date`/`Timestamp`,
que é o que um seletor de data do app devolve). Qualquer outra forma é recusada
com a instrução do formato — e o dado **não** chega ao modelo.

O critério de número é o **mesmo** de `preparacao._numero` (importado de lá, não
reescrito): uma única fonte de verdade para a vírgula decimal. Se alguém afrouxar
o parser, o teste `tests/test_contrato.py` fica vermelho.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from quilha.preparacao import _numero

# ---------------------------------------------------------------------------
# Os 14 campos que o app recebe na abertura do sinistro
# ---------------------------------------------------------------------------
CAMPOS_ABERTURA: tuple[str, ...] = (
    "data_ocorrencia",
    "hora_ocorrencia",
    "data_abertura",
    "canal_abertura",
    "tipo_sinistro",
    "uf",
    "idade_segurado",
    "dias_de_apolice",
    "sinistros_12m",
    "idade_veiculo",
    "valor_fipe",
    "valor_reclamado",
    "boletim_ocorrencia",
    "terceiro_identificado",
)

# Campos conversíveis em número, datas e horários.
CAMPOS_NUMERICOS: tuple[str, ...] = (
    "idade_segurado",
    "dias_de_apolice",
    "sinistros_12m",
    "idade_veiculo",
    "valor_fipe",
    "valor_reclamado",
)
CAMPOS_DATA: tuple[str, ...] = ("data_ocorrencia", "data_abertura")
CAMPOS_HORA: tuple[str, ...] = ("hora_ocorrencia",)

# Domínios categóricos — extraídos da base do case (60.000 sinistros) e conferidos
# pelo teste `test_dominios_batem_com_a_base`. A string vazia só é aceita em
# `boletim_ocorrencia`: ali o branco é um **caso de negócio** (4,83% da base, taxa
# de fraude de 3,24%, acima da média) e vira a categoria "Não informado" dentro do
# `preparar` — nunca imputação.
DOMINIOS_CATEGORICOS: dict[str, tuple[str, ...]] = {
    "canal_abertura": ("App", "Corretor", "Central 0800"),
    "tipo_sinistro": (
        "Colisão",
        "Vidros",
        "Roubo/Furto",
        "Terceiros",
        "Alagamento",
        "Incêndio",
    ),
    "uf": (
        "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT",
        "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
    ),
    "boletim_ocorrencia": ("Sim", "Não", ""),
    "terceiro_identificado": ("Sim", "Não", "Não se aplica"),
}
CAMPOS_CATEGORICOS: tuple[str, ...] = tuple(DOMINIOS_CATEGORICOS)

# Formato exigido de `hora_ocorrencia` no app: dois dígitos, dois pontos, dois dígitos.
REGEX_HORA = r"^\d{2}:\d{2}$"

# Único formato de texto aceito para data (texto ISO). Objetos de data de verdade
# (o que um seletor de data devolve) passam sem conversão de texto.
FORMATO_DATA_CANONICO = "%Y-%m-%d"


@dataclass(frozen=True)
class ErroContrato:
    """Uma recusa, com o campo, o valor recebido e o motivo — legível por gente.

    `linha` é a posição da linha no lote (0-based) e existe porque o app recebe a
    base do mês inteira (~12.000 sinistros) na tela de fila: sem a linha, um erro
    num lote desse tamanho não é acionável. Vem `None` quando o erro é do payload
    como um todo (coluna ausente).
    """

    campo: str
    valor_recebido: str
    motivo: str
    linha: int | None = None

    def __str__(self) -> str:
        local = "" if self.linha is None else f"[linha {self.linha}] "
        return f"{local}{self.campo} = {self.valor_recebido!r}: {self.motivo}"


# ---------------------------------------------------------------------------
# Auxiliares de leitura do payload
# ---------------------------------------------------------------------------
def _vazio(valor) -> bool:
    """Verdadeiro para ausente (`None`, `NaN`, `NaT`, `pd.NA`) ou texto só de espaços."""
    if valor is None:
        return True
    try:
        if bool(pd.isna(valor)):
            return True
    except (TypeError, ValueError):
        pass
    return isinstance(valor, str) and not valor.strip()


def _como_texto(valor) -> str:
    """Valor em texto curto para a mensagem de erro; vazio vira `(vazio)`."""
    return "(vazio)" if _vazio(valor) else str(valor).strip()


def _como_dataframe(payload) -> pd.DataFrame:
    """Aceita um `DataFrame` (lote) ou um dicionário de campos (um sinistro)."""
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, Mapping):
        return pd.DataFrame([payload])
    raise TypeError(
        "payload deve ser um pandas.DataFrame (lote) ou um dicionário com as 14 "
        f"colunas de abertura; recebido: {type(payload).__name__}"
    )


def _erros_numericos(campo: str, serie: pd.Series) -> list[ErroContrato]:
    """Recusa campo numérico vazio ou que não converte (mesmo critério de `_numero`)."""
    convertido = _numero(serie)
    erros: list[ErroContrato] = []
    valores = zip(
        serie.to_numpy(dtype=object), convertido.to_numpy(dtype=object), strict=True
    )
    for linha, (bruto, numero) in enumerate(valores):
        if _vazio(bruto):
            erros.append(
                ErroContrato(
                    campo,
                    "(vazio)",
                    "campo numérico vazio: o contrato não imputa na fronteira",
                    linha,
                )
            )
        elif pd.isna(numero):
            erros.append(
                ErroContrato(
                    campo,
                    _como_texto(bruto),
                    "não converte para número (formatos aceitos: '8837,07' ou '8837.07')",
                    linha,
                )
            )
    return erros


def _data_valida(valor) -> bool:
    """Objeto de data de verdade, ou texto exatamente no formato `AAAA-MM-DD`.

    Nada de inferência: `format="mixed"` aceitaria `"13:53"` como *hoje às 13:53* e
    `"03/04/2026"` como 4 de março. Aqui, ou o texto é ISO, ou é recusado.
    """
    if isinstance(valor, dt.date | dt.datetime | pd.Timestamp):
        return not pd.isna(valor)
    try:
        return not pd.isna(pd.to_datetime(valor, format=FORMATO_DATA_CANONICO, errors="coerce"))
    except (TypeError, ValueError):
        return False


def _hora_valida(valor) -> bool:
    """Objeto de hora de verdade, ou texto exatamente no formato `HH:MM` (00–23)."""
    if isinstance(valor, dt.time):
        return 0 <= valor.hour <= 23
    texto = str(valor).strip()
    if re.match(REGEX_HORA, texto) is None:
        return False
    return 0 <= int(texto[:2]) <= 23


def _erros_datas(campo: str, serie: pd.Series) -> list[ErroContrato]:
    """Recusa data vazia ou que não esteja em formato explícito (ver `_data_valida`)."""
    erros: list[ErroContrato] = []
    for linha, bruto in enumerate(serie.to_numpy(dtype=object)):
        if _vazio(bruto):
            erros.append(ErroContrato(campo, "(vazio)", "data vazia", linha))
        elif not _data_valida(bruto):
            erros.append(
                ErroContrato(
                    campo,
                    _como_texto(bruto),
                    "não é uma data explícita (formato aceito: 'AAAA-MM-DD', "
                    "por exemplo 2026-07-18)",
                    linha,
                )
            )
    return erros


def _erros_horas(campo: str, serie: pd.Series) -> list[ErroContrato]:
    """Recusa hora fora de `HH:MM` ou fora de 00:00–23:59.

    `preparar` deriva a hora dos dois primeiros caracteres e zera o que estiver fora
    de 0–23; sem esta checagem, `"99:99"` viraria nulo em silêncio e o sinistro
    entraria na fila com a hora imputada pela mediana do treino.
    """
    erros: list[ErroContrato] = []
    for linha, bruto in enumerate(serie.to_numpy(dtype=object)):
        if _vazio(bruto):
            erros.append(ErroContrato(campo, "(vazio)", "hora vazia", linha))
        elif not _hora_valida(bruto):
            erros.append(
                ErroContrato(
                    campo,
                    _como_texto(bruto),
                    "não está no formato 'HH:MM' entre 00:00 e 23:59",
                    linha,
                )
            )
    return erros


def _erros_categoricos(campo: str, serie: pd.Series) -> list[ErroContrato]:
    """Recusa valor fora do domínio da coluna (categoria nova não é adivinhada)."""
    permitidos = DOMINIOS_CATEGORICOS[campo]
    erros: list[ErroContrato] = []
    for linha, bruto in enumerate(serie.to_numpy(dtype=object)):
        texto = "" if _vazio(bruto) else str(bruto).strip()
        if texto == "":
            if "" in permitidos:
                continue  # boletim em branco: caso de negócio, vira "Não informado"
            erros.append(ErroContrato(campo, "(vazio)", "campo categórico vazio", linha))
        elif texto not in permitidos:
            erros.append(
                ErroContrato(
                    campo,
                    texto,
                    f"valor fora do domínio: esperado um de {list(permitidos)}",
                    linha,
                )
            )
    return erros


# ---------------------------------------------------------------------------
# Função pública
# ---------------------------------------------------------------------------
def validar_payload(payload: pd.DataFrame | Mapping) -> list[ErroContrato]:
    """Valida o payload do app contra o contrato das 14 colunas de abertura.

    Devolve a **lista de recusas** — lista vazia significa payload válido e apto a
    seguir para `predict_proba`. Nunca corrige, completa ou imputa nada: quem
    corrige é quem digitou.

    Erros de lote vêm todos de uma vez (não para no primeiro) e cada um traz a
    linha, para a tela de fila conseguir apontar o sinistro problemático.
    """
    df = _como_dataframe(payload)

    erros: list[ErroContrato] = [
        ErroContrato(campo, "(ausente)", "coluna obrigatória ausente no payload")
        for campo in CAMPOS_ABERTURA
        if campo not in df.columns
    ]

    for campo in CAMPOS_ABERTURA:
        if campo not in df.columns:
            continue
        if campo in CAMPOS_NUMERICOS:
            erros.extend(_erros_numericos(campo, df[campo]))
        elif campo in CAMPOS_DATA:
            erros.extend(_erros_datas(campo, df[campo]))
        elif campo in CAMPOS_HORA:
            erros.extend(_erros_horas(campo, df[campo]))
        else:
            erros.extend(_erros_categoricos(campo, df[campo]))

    return erros

"""Preparação determinística dos sinistros (Fase 3 do CRISP-DM).

Este módulo concentra toda a engenharia de atributos do projeto. A função pública
`preparar` é **pura e determinística**: não aprende nada do treino — nada de média,
mediana, moda ou quantil. Toda estatística aprendida é papel da etapa de imputação,
dentro do Pipeline (ver `quilha.pipeline`).

Contrato de `preparar`
----------------------
Entrada
    DataFrame com o schema **cru** do CSV, **sem** `fraude` e **sem**
    `valor_indenizado` (o app recebe um chamado novo, sem gabarito).
    Aceita a base inteira **ou uma linha**; aceita as colunas em **qualquer ordem**;
    **ignora** colunas extras (ex.: `id_sinistro` e campos de controle).
Saída
    DataFrame tipado, com as features derivadas, colunas e ordem **estáveis**.
    Campos vazios, textos inválidos ou datas malformadas viram nulo
    (`NaN`/`NaT`/`<NA>`) — a imputação do Pipeline resolve depois.

Robustez para dado novo
-----------------------
Nenhuma entrada provoca exceção. `valor_reclamado` é aceito como `"8837,07"`
(texto com vírgula), `"8837.07"` (texto com ponto) ou `8837.07` (float já
convertido). Nulo em campo numérico é aceitável e esperado para dado sujo: quem
trata é o `SimpleImputer`, nunca esta função.

Decisões de tratamento (cada uma sustentada por um achado da Fase 2)
--------------------------------------------------------------------
* `valor_indenizado` — **excluída**. Presente em 13,44% dos sinistros com valor
  zero e taxa de fraude de 17,37% neles contra 0,15% nos demais (115x). É
  preenchida no encerramento: vazamento.
* `id_sinistro` — chave de controle, nunca feature.
* `data_ocorrencia` / `data_abertura` — data crua não é feature; permanecem apenas
  para a divisão temporal e auditoria. O tempo entra por atributo derivado.
* `boletim_ocorrencia` em branco — **não** é imputado: vira a categoria explícita
  "Não informado", porque o branco é mais fraudulento que a média (3,24% contra
  2,47%).
* `terceiro_identificado` — a coluna original é redundante (só varia dentro de
  "Colisão"); recupera-se o sinal útil em `tem_terceiro_identificado` e a original
  é descartada.
* `data_ocorrencia` fora da janela — não é descartada (688 registros legítimos:
  evento em março, abertura em abril).
* Nenhuma linha é descartada por nulo e nenhuma por outlier sem regra de negócio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Catálogo de colunas
# ---------------------------------------------------------------------------
# Colunas de CONTROLE: não entram no modelo. Preservadas na saída apenas quando
# vierem na entrada, para rastreabilidade e para a divisão temporal.
COLUNAS_CONTROLE: tuple[str, ...] = ("id_sinistro", "data_ocorrencia", "data_abertura")

# Ordem canônica das features. A saída de `preparar` é sempre esta sequência,
# independentemente da ordem em que as colunas chegam.
COLUNAS_NUMERICAS: tuple[str, ...] = (
    "idade_segurado",
    "dias_de_apolice",
    "sinistros_12m",
    "idade_veiculo",
    "valor_fipe",
    "valor_reclamado",
    "hora",
    "atraso_abertura_dias",
    "razao_reclamado_fipe",
)
COLUNAS_BOOLEANAS: tuple[str, ...] = ("tem_terceiro_identificado",)
COLUNAS_CATEGORICAS: tuple[str, ...] = (
    "canal_abertura",
    "tipo_sinistro",
    "uf",
    "boletim_ocorrencia",
    "periodo_dia",
    "fim_de_semana",
    "faixa_apolice",
    "faixa_valor_reclamado",
)
FEATURES: tuple[str, ...] = COLUNAS_NUMERICAS + COLUNAS_BOOLEANAS + COLUNAS_CATEGORICAS

# Rótulo do boletim em branco (item d). Categoria própria, nunca imputação.
BOLETIM_NAO_INFORMADO = "Não informado"

# Faixas do case ("Quadro de resposta", Pergunta 2). Intervalos fechados à direita.
CORTES_APOLICE: list[float] = [-1, 30, 90, 180, 365, np.inf]
ROTULOS_APOLICE: list[str] = ["até 30", "31 a 90", "91 a 180", "181 a 365", "mais de 365"]
CORTES_VALOR: list[float] = [0, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, np.inf]
ROTULOS_VALOR: list[str] = [
    "até 10 mil",
    "10 a 20",
    "20 a 30",
    "30 a 40",
    "40 a 50",
    "50 a 60",
    "acima de 60",
]

# Períodos do dia (item f). 00:00–05:59 madrugada, 06–11 manhã, 12–17 tarde, 18–23 noite.
CORTES_HORA: list[float] = [-1, 5, 11, 17, 23]
ROTULOS_PERIODO: list[str] = ["madrugada", "manhã", "tarde", "noite"]


# ---------------------------------------------------------------------------
# Conversores tolerantes a lixo (nunca lançam exceção)
# ---------------------------------------------------------------------------
def _vazio_texto(indice: pd.Index) -> pd.Series:
    """Coluna de texto toda nula, com dtype object."""
    return pd.Series(np.nan, index=indice, dtype=object)


def _texto(serie: pd.Series) -> pd.Series:
    """Texto sem espaços nas pontas; vazio ou ausente vira `np.nan`.

    Devolve sempre `object` com `np.nan` — nunca `pd.NA`. O `SimpleImputer` do
    scikit-learn quebra com `pd.NA` em coluna object ("boolean value of NA is
    ambiguous"), então a normalização acontece aqui, na fronteira.
    """
    limpo = serie.astype("string").str.strip()
    limpo = limpo.mask(limpo.eq(""))
    return pd.Series(limpo.to_numpy(dtype=object, na_value=np.nan), index=serie.index, dtype=object)


def _numero(serie: pd.Series) -> pd.Series:
    """Número a partir de `"8837,07"`, `"8837.07"` ou `8837.07`; inválido vira `<NA>`.

    Sempre devolve `Float64` (nullable): a conversão final para inteiro acontece em
    `_inteiro`, para as colunas que são contagem.
    """
    if pd.api.types.is_bool_dtype(serie) or pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce").astype("Float64")
    texto = serie.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(texto, errors="coerce").astype("Float64")


def _inteiro(serie: pd.Series) -> pd.Series:
    """Inteiro nullable (`Int64`), arredondando: é o "inteiro tratando NaN" da Fase 2."""
    return serie.round().astype("Int64")


def _data(serie: pd.Series) -> pd.Series:
    """Data; qualquer valor malformado vira `NaT` em vez de exceção."""
    if pd.api.types.is_datetime64_any_dtype(serie):
        return pd.to_datetime(serie, errors="coerce")
    return pd.to_datetime(serie.astype("string"), errors="coerce", format="mixed")


def _rotular(condicao: pd.Series, verdadeiro: str, falso: str) -> pd.Series:
    """Rótulo texto a partir de uma condição booleana; nulo onde a condição é nula."""
    saida = pd.Series(np.nan, index=condicao.index, dtype=object)
    valido = condicao.notna()
    if bool(valido.any()):
        marca = condicao.loc[valido].astype(bool).to_numpy()
        saida.loc[valido] = np.where(marca, verdadeiro, falso)
    return saida


def _col_texto(df: pd.DataFrame, nome: str) -> pd.Series:
    """Texto da coluna, ou coluna toda nula se a coluna não vier na entrada."""
    return _texto(df[nome]) if nome in df.columns else _vazio_texto(df.index)


def _col_numero(df: pd.DataFrame, nome: str) -> pd.Series:
    """Número da coluna, ou coluna toda nula se a coluna não vier na entrada."""
    if nome in df.columns:
        return _numero(df[nome])
    return pd.Series(np.nan, index=df.index, dtype="Float64")


def _col_data(df: pd.DataFrame, nome: str) -> pd.Series:
    """Data da coluna, ou coluna toda `NaT` se a coluna não vier na entrada."""
    if nome in df.columns:
        return _data(df[nome])
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")


# ---------------------------------------------------------------------------
# Função pública
# ---------------------------------------------------------------------------
def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Trata o sinistro cru e devolve o DataFrame de features derivadas.

    Pura, determinística e idempotente: rodar duas vezes dá exatamente o mesmo
    resultado, e o resultado de uma linha isolada é idêntico à linha correspondente
    da base inteira. Ignora colunas extras e nunca lança exceção.
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # Índice posicional interno: evita alinhamento por rótulo (índice duplicado não
    # explode a montagem). O índice original volta no fim.
    indice_original = df.index
    base = df.copy()
    base.index = pd.RangeIndex(len(base))

    # --- controle (não são features) -------------------------------------
    id_sinistro = _col_texto(base, "id_sinistro")
    ocorrencia = _col_data(base, "data_ocorrencia")
    abertura = _col_data(base, "data_abertura")

    # --- numéricas --------------------------------------------------------
    idade_segurado = _inteiro(_col_numero(base, "idade_segurado"))
    dias_de_apolice = _inteiro(_col_numero(base, "dias_de_apolice"))
    sinistros_12m = _inteiro(_col_numero(base, "sinistros_12m"))
    idade_veiculo = _inteiro(_col_numero(base, "idade_veiculo"))
    valor_fipe = _col_numero(base, "valor_fipe")
    valor_reclamado = _col_numero(base, "valor_reclamado")

    # (f) hora do evento, a partir de `hora_ocorrencia` (texto "HH:MM").
    hora_texto = _col_texto(base, "hora_ocorrencia")
    hora = pd.to_numeric(
        hora_texto.astype("string").str.slice(0, 2), errors="coerce"
    ).astype("Int64")
    hora = hora.where(hora.between(0, 23))  # fora de 00–23 é inválido -> nulo

    # (f) período do dia, derivado da hora (faixas fixas, não aprendidas).
    periodo_dia = pd.cut(
        hora, bins=CORTES_HORA, labels=ROTULOS_PERIODO, right=True, include_lowest=True
    ).astype(object)

    # (f) fim de semana, derivado de `data_ocorrencia`.
    fim_de_semana = _rotular(ocorrencia.dt.day_of_week.astype("Int64").ge(5), "Sim", "Não")

    # (g) atraso entre a ocorrência e a abertura: conhecido no momento da decisão.
    atraso_abertura_dias = (abertura - ocorrencia).dt.days.astype("Int64")

    # (i) razão reclamado/FIPE. FIPE nula ou não positiva -> nulo (sem divisão por zero).
    razao_reclamado_fipe = (valor_reclamado / valor_fipe.where(valor_fipe > 0)).astype("Float64")

    # --- booleanas --------------------------------------------------------
    # (e) sinal útil escondido dentro de "Colisão". "Não se aplica" é ausência de
    # terceiro, logo False; valor desconhecido/ausente fica nulo.
    terceiro = _col_texto(base, "terceiro_identificado")
    tem_terceiro_identificado = terceiro.map(
        {"Sim": True, "Não": False, "Não se aplica": False}
    ).astype("boolean")

    # --- categóricas ------------------------------------------------------
    canal_abertura = _col_texto(base, "canal_abertura")
    tipo_sinistro = _col_texto(base, "tipo_sinistro")
    uf = _col_texto(base, "uf")

    # (d) branco vira categoria explícita — o branco é mais fraudulento que a média.
    boletim_ocorrencia = _col_texto(base, "boletim_ocorrencia").fillna(BOLETIM_NAO_INFORMADO)

    # (h) faixa de tempo de apólice, nas faixas do case.
    faixa_apolice = pd.cut(
        dias_de_apolice, bins=CORTES_APOLICE, labels=ROTULOS_APOLICE, right=True
    ).astype(object)

    # (i) faixa de valor reclamado, nas faixas do case.
    faixa_valor_reclamado = pd.cut(
        valor_reclamado, bins=CORTES_VALOR, labels=ROTULOS_VALOR, right=True, include_lowest=True
    ).astype(object)

    # --- montagem, na ordem canônica --------------------------------------
    colunas: dict[str, pd.Series] = {}
    if "id_sinistro" in base.columns:
        colunas["id_sinistro"] = id_sinistro
    if "data_ocorrencia" in base.columns:
        colunas["data_ocorrencia"] = ocorrencia
    if "data_abertura" in base.columns:
        colunas["data_abertura"] = abertura

    colunas.update(
        {
            "idade_segurado": idade_segurado,
            "dias_de_apolice": dias_de_apolice,
            "sinistros_12m": sinistros_12m,
            "idade_veiculo": idade_veiculo,
            "valor_fipe": valor_fipe,
            "valor_reclamado": valor_reclamado,
            "hora": hora,
            "atraso_abertura_dias": atraso_abertura_dias,
            "razao_reclamado_fipe": razao_reclamado_fipe,
            "tem_terceiro_identificado": tem_terceiro_identificado,
            "canal_abertura": canal_abertura,
            "tipo_sinistro": tipo_sinistro,
            "uf": uf,
            "boletim_ocorrencia": boletim_ocorrencia,
            "periodo_dia": periodo_dia,
            "fim_de_semana": fim_de_semana,
            "faixa_apolice": faixa_apolice,
            "faixa_valor_reclamado": faixa_valor_reclamado,
        }
    )

    saida = pd.DataFrame(colunas, index=base.index)
    saida.index = indice_original
    return saida

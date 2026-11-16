"""Normalização determinística de representação, com relatório — melhoria B (spec 07).

Princípio
---------
**Representação → normaliza e reporta. Significado, ambiguidade ou domínio → recusa.**

Este módulo faz uma coisa só, e estreita de propósito: pega o dado cru que é apenas
**outra forma de escrever o mesmo valor** e o reescreve na forma canônica, devolvendo
um **relatório** de tudo o que mudou. Nada é corrigido às escondidas.

* **B1** — número pt-BR com os **dois** separadores: `"1.234,56"` → `1234.56`.
* **B2** — categórica com caixa/espaço: `" sp "` → `"SP"`.
* **B3** — data `DD/MM/AAAA` (convenção pt-BR): `"15/08/2026"` → `"2026-08-15"`.

O que continua barrado (é o ponto do desenho)
---------------------------------------------
`"abc"`, `"99:99"`, `"Telefone"` e o número com **um só** ponto (`"1.234"`, ambíguo)
não são representação. O normalizador os **deixa como estão** e o contrato
(`quilha.contrato.validar_payload`) os recusa — com linha, campo, valor e motivo.

Uma fonte de verdade
--------------------
A reescrita usa as **mesmas** regras do contrato (`canonico_categorico`,
`canonico_data`) e o **mesmo** reconhecimento de número do parser canônico
(`preparacao.PADRAO_MILHAR_PT_BR`). Não há segunda definição de "número válido" nem
de "categoria válida" neste projeto.

Ordem de chamada (obrigatória)
------------------------------
`normalizar` roda **antes** do contrato e **antes** do modelo. O app o chama ao
carregar o lote e ao digitar um chamado; `pontuar_lote` o chama de novo (idempotente)
para que nenhuma categoria crua chegue ao Pipeline. É esta ordem que fecha a falha
silenciosa: o modelo nunca vê `" sp "` — vê `"SP"`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import partial

import pandas as pd

from quilha.contrato import (
    CAMPOS_CATEGORICOS,
    CAMPOS_DATA,
    CAMPOS_NUMERICOS,
    canonico_categorico,
    canonico_data,
)
from quilha.preparacao import PADRAO_MILHAR_PT_BR

# Colunas do relatório de normalização, na ordem em que a tela as mostra.
COLUNAS_RELATORIO: tuple[str, ...] = ("linha", "campo", "de", "para", "regra")

# Rótulo da regra que reescreveu cada valor — o relatório diz sempre **por quê**.
REGRA_MILHAR = "B1 · milhar pt-BR"
REGRA_CATEGORICA = "B2 · caixa/espaço"
REGRA_DATA = "B3 · data DD/MM/AAAA"


def _reescrever_numero(valor) -> str | None:
    """Forma decimal canônica **só** do milhar pt-BR inequívoco; `None` se não for o caso.

    `"1.234,56"` → `"1234.56"` (só desmonta o milhar e troca a vírgula pelo ponto —
    transparência no relatório). `"8837,07"` já é a convenção decimal corrente e
    `"1.234"` é ambíguo: os dois voltam `None` e seguem para o parser canônico.
    """
    if not isinstance(valor, str):
        return None
    texto = valor.strip()
    if re.fullmatch(PADRAO_MILHAR_PT_BR, texto) is None:
        return None
    return texto.replace(".", "").replace(",", ".")


def _reescrever_categorico(valor, campo: str) -> str | None:
    """Forma canônica do domínio (caixa/espaço); `None` quando nada muda.

    `" sp "` → `"SP"`. Um valor fora do domínio (`"Telefone"`) não casa e volta
    `None`: quem o recusa é o contrato, não este módulo.
    """
    if not isinstance(valor, str):
        return None
    canonico = canonico_categorico(campo, valor)
    if not canonico or canonico == valor:
        return None
    return canonico


def _reescrever_data(valor) -> str | None:
    """Forma ISO só quando o texto muda (`DD/MM/AAAA` ou espaço); `None` se já é ISO."""
    if not isinstance(valor, str):
        return None
    canonico = canonico_data(valor)
    if canonico is None or canonico == valor:
        return None
    return canonico


# Regra de reescrita por família de campo. Cada uma devolve a forma canônica ou `None`.
_REGRAS: tuple[tuple[tuple[str, ...], str, Callable], ...] = (
    (CAMPOS_NUMERICOS, REGRA_MILHAR, _reescrever_numero),
    (CAMPOS_DATA, REGRA_DATA, _reescrever_data),
)


def _regra_do_campo(campo: str) -> tuple[str, Callable] | None:
    """Qual regra (e rótulo) rege a coluna; `None` para coluna que não é do contrato."""
    for campos, regra, reescrever in _REGRAS:
        if campo in campos:
            return regra, reescrever
    if campo in CAMPOS_CATEGORICOS:
        return REGRA_CATEGORICA, partial(_reescrever_categorico, campo=campo)
    return None


def normalizar(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reescreve a **representação** do lote na forma canônica e devolve `(lote, relatorio)`.

    O lote devolvido é uma **cópia** — a entrada não é alterada. O relatório traz uma
    linha por célula reescrita: `linha` (posição 0-based, a **mesma** dos erros do
    contrato), `campo`, `de`, `para` e `regra` (B1/B2/B3). Sem nada a reescrever, o
    relatório volta **vazio, mas com as colunas** — a tela sabe distinguir "nada mudou"
    de "não houve normalização".
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"normalizar espera um pandas.DataFrame; recebido: {type(df).__name__}"
        )

    lote = df.copy()
    registros: list[dict[str, object]] = []

    for campo in lote.columns:
        regra_do_campo = _regra_do_campo(campo)
        if regra_do_campo is None:
            continue
        regra, reescrever = regra_do_campo
        coluna = lote.columns.get_loc(campo)

        for posicao, bruto in enumerate(lote[campo].to_numpy(dtype=object)):
            canonico = reescrever(bruto)
            if canonico is None:
                continue
            registros.append(
                {
                    "linha": posicao,
                    "campo": campo,
                    "de": str(bruto),
                    "para": canonico,
                    "regra": regra,
                }
            )
            lote.iat[posicao, coluna] = canonico

    relatorio = pd.DataFrame(registros, columns=list(COLUNAS_RELATORIO))
    return lote, relatorio

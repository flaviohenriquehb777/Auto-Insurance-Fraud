"""Leitura da base crua e divisão temporal (Fase 3 do CRISP-DM).

Responsabilidade exclusiva deste módulo: transformar arquivo em dados e separar
passado (treino) de futuro (teste). Nenhum tratamento de feature mora aqui.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Caminhos do projeto
# ---------------------------------------------------------------------------
NOME_RAIZ = "pyproject.toml"
CAMINHO_RELATIVO_BASE = Path("data") / "raw" / "quilha_sinistros.csv"


def encontrar_raiz(inicio: Path | None = None) -> Path:
    """Sobe na árvore de diretórios até achar a raiz do projeto (onde está o pyproject.toml).

    Necessário porque notebook, script e app rodam a partir de diretórios diferentes.
    """
    atual = (inicio or Path.cwd()).resolve()
    for pasta in [atual, *atual.parents]:
        if (pasta / NOME_RAIZ).exists():
            return pasta
    raise FileNotFoundError(f"Raiz do projeto não encontrada (nenhum {NOME_RAIZ} acima).")


RAIZ_PROJETO = encontrar_raiz()
CAMINHO_BASE_CRUA = RAIZ_PROJETO / CAMINHO_RELATIVO_BASE
PASTA_MODELOS = RAIZ_PROJETO / "models"
PASTA_FIGURAS = RAIZ_PROJETO / "reports" / "figures"

# ---------------------------------------------------------------------------
# Regra de divisão temporal
# ---------------------------------------------------------------------------
# Decisão tomada na Fase 3 com a evidência da taxa de fraude por mês de abertura
# (ver docs/03_preparacao_dados.md, seção 2):
#   2026-04: 11.692 sinistros, 283 fraudes (2,42%)
#   2026-05: 12.146 sinistros, 313 fraudes (2,58%)
#   2026-06: 11.947 sinistros, 314 fraudes (2,63%)
#   2026-07: 12.311 sinistros, 289 fraudes (2,35%)
#   2026-08: 11.904 sinistros, 281 fraudes (2,36%)
# A taxa é estável nos 5 meses (amplitude de 0,28 ponto percentual) e agosto é o
# mês mais recente com volume completo. O corte em 01/08/2026 deixa:
#   treino = 01/04/2026 a 31/07/2026 (4 meses, 48.096 sinistros, 1.199 fraudes)
#   teste  = 01/08/2026 a 31/08/2026 (1 mês,   11.904 sinistros,   281 fraudes)
# O teste ter exatamente 1 mês é essencial: a capacidade de perícia do case é
# mensal (900/mês), então a avaliação da política na Fase 5 fica 1:1, sem conversão.
DATA_CORTE = pd.Timestamp("2026-08-01")


def carregar_base_crua(caminho: Path | None = None) -> pd.DataFrame:
    """Carrega a base exatamente como ela foi entregue, tudo como texto.

    `dtype=str` de propósito: preserva o formato original de cada coluna
    (inclusive a vírgula decimal de `valor_reclamado`) para auditoria e permite
    testar a função `preparar` com o dado do jeito que ele chega do CSV.
    """
    return pd.read_csv(caminho or CAMINHO_BASE_CRUA, dtype=str)


def dividir_temporal(
    base: pd.DataFrame, data_corte: pd.Timestamp | str = DATA_CORTE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa treino (passado) e teste (futuro) por `data_abertura`.

    Sem `shuffle` e sem `stratify`: a divisão é cronológica, o teste é sempre o
    período mais recente. O case exige avaliar em sinistros mais recentes que os
    usados no treino.
    """
    corte = pd.Timestamp(data_corte)
    abertura = pd.to_datetime(base["data_abertura"])
    treino = base.loc[abertura < corte].copy()
    teste = base.loc[abertura >= corte].copy()
    if treino.empty or teste.empty:
        raise ValueError(f"Divisão temporal vazia com o corte {corte.date()}: "
                         f"treino={len(treino)}, teste={len(teste)}")
    return treino, teste

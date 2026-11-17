"""Estágios do pipeline DVC — a cadeia que o `dvc repro` executa.

Por que este módulo existe
--------------------------
O pacote `quilha` **não** é instalado no ambiente virtual: o projeto roda com
`PYTHONPATH=src` (o mesmo contrato do `app.py` da raiz e do `pytest`). Para o DVC
executar as etapas em qualquer sistema operacional, o `PYTHONPATH` é declarado em
`dvc.env` e carregado por `uv run --env-file dvc.env` — ver `dvc.yaml`.

O que o pipeline reproduz
-------------------------
O artefato auditado (`models/pipeline_fraude.joblib`) entra como **entrada** e nunca é
regravado: não existe estágio de treino. A cadeia reproduz o que a Fase 5 afirmou, para
que os números do comitê deixem de ser "digitados" e passem a ser recalculados:

    data/raw/quilha_sinistros.csv  → preparo   → data/processed/{treino,teste}.csv
                                   → avaliacao → reports/metricas/avaliacao.json

`preparo` refaz a divisão temporal (abr-jul treino / ago teste). `avaliacao` mede a
política do campeão no mês de teste contra a regra atual de R$ 50 mil e grava o
resultado como métrica versionada.

Nada aqui treina, ajusta hiperparâmetro ou decide nada de negócio: a medição vem de
`quilha.avaliacao`, a mesma função que a Fase 5 usa.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from quilha.avaliacao import (
    construir_politica_campeao,
    construir_referencias,
    medir_politicas_no_mes,
)
from quilha.dados import RAIZ_PROJETO, carregar_base_crua, dividir_temporal
from quilha.modelagem import CAPACIDADE_MENSAL, COLUNAS_EXCLUIDAS_ENTRADA
from quilha.pipeline import carregar_pipeline

# ---------------------------------------------------------------------------
# Caminhos dos artefatos do pipeline
# ---------------------------------------------------------------------------
PASTA_PROCESSADOS = RAIZ_PROJETO / "data" / "processed"
CAMINHO_TREINO = PASTA_PROCESSADOS / "treino.csv"
CAMINHO_TESTE = PASTA_PROCESSADOS / "teste.csv"

PASTA_METRICAS = RAIZ_PROJETO / "reports" / "metricas"
CAMINHO_METRICAS = PASTA_METRICAS / "avaliacao.json"

# Políticas de referência que precisam de ajuste no treino antes de medir.
REFERENCIAS_A_AJUSTAR = ("regra_atual", "aleatoria")


# ---------------------------------------------------------------------------
# Estágio 1 — preparo
# ---------------------------------------------------------------------------
def estagio_preparo() -> None:
    """Divide a base crua em treino (abr-jul) e teste (ago) e grava os dois CSVs.

    A divisão é cronológica e vem de `quilha.dados.divir_temporal` — nenhuma regra de
    corte é reescrita aqui. Os arquivos saem como texto (`dtype=str` na leitura), para
    que o valor reclamado preserve a vírgula decimal do formato original.
    """
    treino, teste = dividir_temporal(carregar_base_crua())
    PASTA_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    treino.to_csv(CAMINHO_TREINO, index=False)
    teste.to_csv(CAMINHO_TESTE, index=False)
    print(f"preparo: treino={len(treino):,} linhas | teste={len(teste):,} linhas".replace(",", "."))


# ---------------------------------------------------------------------------
# Estágio 2 — avaliação
# ---------------------------------------------------------------------------
def estagio_avaliacao() -> dict[str, float | int]:
    """Mede a política do campeão congelado e grava `avaliacao.json` como métrica.

    O `k` é a capacidade mensal (900). A regra atual e a referência simples são
    ajustadas no **treino** e medidas no **teste**, sob o mesmo critério econômico da
    Fase 5 (`21.700 × TP − 700 × k`).
    """
    treino = pd.read_csv(CAMINHO_TREINO, dtype=str)
    teste = pd.read_csv(CAMINHO_TESTE, dtype=str)

    colunas_alvo = list(COLUNAS_EXCLUIDAS_ENTRADA)
    y_treino, x_treino = treino["fraude"].astype(int), treino.drop(columns=colunas_alvo)
    y_teste, x_teste = teste["fraude"].astype(int), teste.drop(columns=colunas_alvo)

    politicas = {"campeao": construir_politica_campeao(carregar_pipeline(), k=CAPACIDADE_MENSAL)}
    referencias = construir_referencias(k=CAPACIDADE_MENSAL)
    for chave in REFERENCIAS_A_AJUSTAR:
        referencias[chave].estimador.fit(x_treino, y_treino)

    tabela = medir_politicas_no_mes({**politicas, **referencias}, x_teste, y_teste)
    por_chave = tabela.set_index("chave").to_dict("index")
    campeao, regra = por_chave["campeao"], por_chave["regra_atual"]

    metricas: dict[str, float | int] = {
        "k_pericias": int(campeao["n_periciados"]),
        "n_teste": int(len(teste)),
        "n_fraudes_teste": int(y_teste.sum()),
        "tp_campeao": int(campeao["tp"]),
        "fp_campeao": int(campeao["fp"]),
        "fn_campeao": int(campeao["fn"]),
        "precisao_campeao": round(float(campeao["precisao"]), 4),
        "cobertura_campeao": round(float(campeao["cobertura"]), 4),
        "resultado_campeao": round(float(campeao["economia"]), 2),
        "resultado_regra_atual": round(float(regra["economia"]), 2),
        "ganho_campeao_sobre_regra_atual": round(
            float(campeao["economia"]) - float(regra["economia"]), 2
        ),
    }

    PASTA_METRICAS.mkdir(parents=True, exist_ok=True)
    texto = json.dumps(metricas, ensure_ascii=False, indent=2) + "\n"
    CAMINHO_METRICAS.write_text(texto, encoding="utf-8")
    print(texto, end="")
    return metricas


# ---------------------------------------------------------------------------
# CLI — o DVC chama `python -m quilha.estagios <nome>`
# ---------------------------------------------------------------------------
ESTAGIOS = {"preparo": estagio_preparo, "avaliacao": estagio_avaliacao}


def main(argv: list[str] | None = None) -> int:
    """Executa um estágio pelo nome. É o único ponto de entrada do `dvc.yaml`."""
    parser = argparse.ArgumentParser(description="Estágios do pipeline DVC da Quilha.")
    parser.add_argument("estagio", choices=sorted(ESTAGIOS))
    argumentos = parser.parse_args(argv)
    ESTAGIOS[argumentos.estagio]()
    return 0


if __name__ == "__main__":
    sys.exit(main())

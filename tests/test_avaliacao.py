from __future__ import annotations

import numpy as np
import pandas as pd

from quilha.avaliacao import (
    CAMPOS_QUADRO_RESPOSTAS,
    PoliticaPericia,
    aplicar_politica,
    construir_politica_campeao,
    montar_quadro_respostas,
    piso_viabilidade,
    resultado_economico,
)
from quilha.dados import dividir_temporal
from quilha.modelagem import (
    CAPACIDADE_MENSAL,
    CUSTO_FRAUDE_PAGA,
    CUSTO_PERICIA,
    RegraAtualQuilha,
    carregar_treino_teste,
)
from quilha.pipeline import CAMINHO_PIPELINE, carregar_pipeline


class _EstimadorSintetico:
    """Estimador mínimo que devolve escores fixos — só para testar a política."""

    def __init__(self, escores: list[float]) -> None:
        self._escores = np.asarray(escores, dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        escore = self._escores[: len(X)]
        return np.column_stack([1.0 - escore, escore])


def test_resultado_economico_confere_conta_manual() -> None:
    y_true = np.array([1, 0, 1, 1, 0])
    indices = np.array([0, 1, 2])  # k=3 -> TP=2 (índices 0 e 2), FP=1

    resultado = resultado_economico(y_true, indices)

    assert resultado["n_periciados"] == 3
    assert resultado["tp"] == 2
    assert resultado["fp"] == 1
    assert resultado["fn"] == 1
    assert resultado["economia"] == (CUSTO_FRAUDE_PAGA * 2) - (CUSTO_PERICIA * 3)
    assert resultado["custo_total"] == (CUSTO_FRAUDE_PAGA * 1) + (CUSTO_PERICIA * 1)
    assert resultado["custo_sem_pericia"] == CUSTO_FRAUDE_PAGA * 3


def test_piso_de_viabilidade_e_3_2258_por_cento() -> None:
    assert piso_viabilidade() == CUSTO_PERICIA / CUSTO_FRAUDE_PAGA
    assert round(piso_viabilidade() * 100, 4) == 3.2258


def test_regra_atual_nota_monotonica_e_zero_abaixo_do_limiar() -> None:
    X = pd.DataFrame({"valor_reclamado": [20_000.0, 50_000.0, 60_000.0, 120_000.0]})

    notas = RegraAtualQuilha().fit(X).predict_proba(X)[:, 1]

    assert notas[0] == 0.0
    assert notas[1] == 0.0
    assert 0.0 < notas[2] < notas[3] < 1.0


def test_aplicar_politica_respeita_k_e_desempate_estavel() -> None:
    X = pd.DataFrame({"x": range(4)})
    escores = [0.2, 0.9, 0.9, 0.5]

    politica = PoliticaPericia(nome="teste", k=2, estimador=_EstimadorSintetico(escores))
    selecionados = aplicar_politica(politica, X)

    # Top-2 por escore; o empate em 0,9 preserva a ordem original (ordenação estável).
    assert list(selecionados) == [1, 2]

    # `k` maior que a base devolve tudo; `k=0` devolve vazio.
    politica_tudo = PoliticaPericia(nome="tudo", k=10, estimador=_EstimadorSintetico(escores))
    assert len(aplicar_politica(politica_tudo, X)) == 4
    assert len(aplicar_politica(PoliticaPericia("nada", 0, _EstimadorSintetico(escores)), X)) == 0


def test_dividir_temporal_nao_vaza() -> None:
    base = pd.DataFrame(
        {
            "data_abertura": ["2026-07-31", "2026-08-01", "2026-08-15", "2026-07-01"],
            "fraude": [0, 1, 0, 1],
        }
    )

    treino, teste = dividir_temporal(base, data_corte="2026-08-01")

    assert pd.to_datetime(treino["data_abertura"]).max() < pd.to_datetime(
        teste["data_abertura"]
    ).min()


def test_resultado_900_do_campeao_reproduz_a_fase_4() -> None:
    """O artefato carregado, cortado em 900, tem de reproduzir o número auditado."""
    pipeline = carregar_pipeline(CAMINHO_PIPELINE)
    _, _, X_teste, y_teste = carregar_treino_teste()

    politica = construir_politica_campeao(pipeline, k=CAPACIDADE_MENSAL)
    resultado = resultado_economico(y_teste, aplicar_politica(politica, X_teste))

    assert resultado["n_periciados"] == 900
    assert resultado["tp"] == 201
    assert resultado["fp"] == 699
    assert resultado["fn"] == 80
    assert resultado["economia"] == 3_731_700.0


def test_montar_quadro_respostas_cobre_os_24_campos_em_dataframe_e_markdown() -> None:
    respostas = {chave: f"valor-{chave}" for _, chave, _ in CAMPOS_QUADRO_RESPOSTAS}

    quadro, markdown = montar_quadro_respostas(respostas)

    assert list(quadro.columns) == ["pergunta", "campo", "resposta"]
    assert len(quadro) == 24
    assert sorted(quadro["pergunta"].unique()) == [1, 2, 3, 4, 5]

    linhas = markdown.splitlines()
    assert linhas[0] == "| Pergunta | Campo | Resposta |"
    assert len(linhas) == 2 + len(CAMPOS_QUADRO_RESPOSTAS)
    # Campo ausente não some: vira "não informado".
    quadro_incompleto, _ = montar_quadro_respostas({})
    assert (quadro_incompleto["resposta"] == "não informado").all()

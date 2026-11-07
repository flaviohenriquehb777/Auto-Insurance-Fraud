"""Testes da reamostragem da Parte 4 (undersampling controlado)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from quilha.amostragem import (
    METODO_ALEATORIO,
    METODO_CONTROLE,
    METODO_NEARMISS,
    EstrategiaAmostragem,
    ModeloComSubamostragem,
    estrategias_parte4,
    indices_aleatorios,
    indices_nearmiss1,
    reamostrar,
    selecionar_indices,
)
from quilha.modelagem import carregar_treino_teste
from quilha.pipeline import construir_pipeline


def _alvo_sintetico(n_positivos: int = 20, n_negativos: int = 200) -> np.ndarray:
    """Alvo com positivos no começo, como acontece de fato no treino."""
    return np.array([1] * n_positivos + [0] * n_negativos)


# ---------------------------------------------------------------------------
# Estratégias
# ---------------------------------------------------------------------------


def test_estrategias_parte4_cobre_as_cinco_do_prompt() -> None:
    estrategias = estrategias_parte4()

    assert [estrategia.rotulo for estrategia in estrategias] == [
        "controle",
        "aleatorio_1_5",
        "aleatorio_1_10",
        "nearmiss1_1_5",
        "nearmiss1_1_10",
    ]
    assert estrategias[0].metodo == METODO_CONTROLE
    assert estrategias[1].metodo == METODO_ALEATORIO
    assert estrategias[3].metodo == METODO_NEARMISS


def test_nearmiss_roda_nas_mesmas_proporcoes_do_sorteio() -> None:
    """A comparação só isola o critério de escolha se o volume de dados for igual."""
    estrategias = {estrategia.rotulo: estrategia for estrategia in estrategias_parte4()}

    assert estrategias["aleatorio_1_5"].proporcao == estrategias["nearmiss1_1_5"].proporcao
    assert estrategias["aleatorio_1_10"].proporcao == estrategias["nearmiss1_1_10"].proporcao


def test_descricao_das_estrategias_menciona_o_metodo() -> None:
    for estrategia in estrategias_parte4():
        assert estrategia.rotulo[:4].lower() in estrategia.descricao().lower().replace("-", "")


# ---------------------------------------------------------------------------
# Sorteio aleatório
# ---------------------------------------------------------------------------


def test_sorteio_mantem_todos_os_positivos_e_respeita_a_proporcao() -> None:
    alvo = _alvo_sintetico(n_positivos=20, n_negativos=200)

    indices = indices_aleatorios(y=alvo, proporcao=5.0, seed=42)

    mantidos = alvo[indices]
    assert int(np.sum(mantidos == 1)) == 20
    assert int(np.sum(mantidos == 0)) == 100
    assert len(indices) == 120


def test_sorteio_troca_as_linhas_quando_muda_a_semente() -> None:
    alvo = _alvo_sintetico()

    primeira = indices_aleatorios(y=alvo, proporcao=5.0, seed=42)
    repetida = indices_aleatorios(y=alvo, proporcao=5.0, seed=42)
    outra = indices_aleatorios(y=alvo, proporcao=5.0, seed=7)

    assert np.array_equal(primeira, repetida)
    assert not np.array_equal(primeira, outra)


def test_sorteio_nunca_pede_mais_negativos_do_que_existe() -> None:
    alvo = _alvo_sintetico(n_positivos=20, n_negativos=30)

    indices = indices_aleatorios(y=alvo, proporcao=10.0, seed=42)

    assert int(np.sum(alvo[indices] == 0)) == 30


# ---------------------------------------------------------------------------
# NearMiss-1
# ---------------------------------------------------------------------------


def test_nearmiss1_prefere_negativos_proximos_da_fronteira() -> None:
    """Dois grupos de negativos: os perto dos positivos devem sobreviver."""
    matriz = pd.DataFrame(
        {
            "proxima": [0.0] * 5 + [0.1] * 4 + [100.0] * 4,
            "distante": [0.0] * 5 + [0.0] * 4 + [100.0] * 4,
        }
    )
    alvo = np.array([1] * 5 + [0] * 8)

    indices = indices_nearmiss1(X=matriz, y=alvo, proporcao=0.8, n_vizinhos=1)

    sobreviveu = np.zeros(len(alvo), dtype=bool)
    sobreviveu[indices] = True
    assert sobreviveu[5:9].all()
    assert not sobreviveu[9:].any()


def test_nearmiss1_e_deterministico_e_mantem_todos_os_positivos() -> None:
    rng = np.random.default_rng(42)
    matriz = pd.DataFrame(rng.random((300, 4)))
    alvo = _alvo_sintetico(n_positivos=30, n_negativos=270)

    primeira = indices_nearmiss1(X=matriz, y=alvo, proporcao=5.0)
    segunda = indices_nearmiss1(X=matriz, y=alvo, proporcao=5.0)

    assert np.array_equal(primeira, segunda)
    assert int(np.sum(alvo[primeira] == 1)) == 30
    assert int(np.sum(alvo[primeira] == 0)) == 150


# ---------------------------------------------------------------------------
# Seleção e reamostragem
# ---------------------------------------------------------------------------


def test_controle_nao_corta_nenhuma_linha() -> None:
    matriz = pd.DataFrame({"a": np.arange(10, dtype=float)})
    alvo = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    assert selecionar_indices(matriz, alvo, None) is None
    assert selecionar_indices(matriz, alvo, EstrategiaAmostragem("c", METODO_CONTROLE)) is None

    X_saida, y_saida = reamostrar(matriz, alvo, None)
    assert X_saida is matriz
    assert y_saida is alvo


def test_reamostrar_troca_o_numero_de_linhas_apenas_com_estimulo() -> None:
    alvo = _alvo_sintetico(n_positivos=20, n_negativos=200)
    matriz = pd.DataFrame({"a": np.arange(len(alvo), dtype=float)})

    X_saida, y_saida = reamostrar(
        matriz,
        alvo,
        EstrategiaAmostragem("aleatorio_1_5", METODO_ALEATORIO, 5.0),
    )

    assert len(X_saida) == 120
    assert len(y_saida) == 120


# ---------------------------------------------------------------------------
# Modelo com subamostragem
# ---------------------------------------------------------------------------


def test_modelo_com_subamostragem_registra_o_corte_do_treino() -> None:
    alvo = _alvo_sintetico(n_positivos=20, n_negativos=200)
    matriz = pd.DataFrame({"a": np.arange(len(alvo), dtype=float), "b": 1.0})

    modelo = ModeloComSubamostragem(
        estimador=DecisionTreeClassifier(random_state=42),
        estrategia=EstrategiaAmostragem("aleatorio_1_5", METODO_ALEATORIO, 5.0),
        seed=42,
    ).fit(matriz, alvo)

    assert modelo.n_linhas_original_ == 220
    assert modelo.n_linhas_treino_ == 120
    assert modelo.n_fraudes_treino_ == 20
    assert modelo.proporcao_treino_ == 6.0


def test_modelo_com_subamostragem_sem_estimulo_usa_o_treino_integral() -> None:
    alvo = _alvo_sintetico(n_positivos=20, n_negativos=200)
    matriz = pd.DataFrame({"a": np.arange(len(alvo), dtype=float), "b": 1.0})

    modelo = ModeloComSubamostragem(
        estimador=DecisionTreeClassifier(random_state=42),
        estrategia=None,
        seed=42,
    ).fit(matriz, alvo)

    assert modelo.n_linhas_treino_ == 220
    assert modelo.n_fraudes_treino_ == 20


def test_modelo_com_subamostragem_roda_dentro_do_pipeline_da_quilha() -> None:
    """Integração real: o Pipeline inteiro da Fase 3 com o reamostrador no lugar do modelo."""
    X_treino, y_treino, _, _ = carregar_treino_teste()
    X_amostra = X_treino.head(2_000)
    y_amostra = y_treino.head(2_000)

    pipeline = construir_pipeline(
        modelo=ModeloComSubamostragem(
            estimador=DecisionTreeClassifier(random_state=42, max_depth=4),
            estrategia=EstrategiaAmostragem("nearmiss1_1_5", METODO_NEARMISS, 5.0),
            seed=42,
        )
    ).fit(X_amostra, y_amostra)

    probabilidades = pipeline.predict_proba(X_amostra)[:, 1]

    assert len(probabilidades) == len(X_amostra)
    assert probabilidades.min() >= 0.0
    assert probabilidades.max() <= 1.0
    passo = pipeline.named_steps["modelo"]
    assert passo.n_linhas_treino_ < passo.n_linhas_original_

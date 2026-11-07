from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from quilha.dados import carregar_base_crua
from quilha.modelagem import (
    CUSTO_FRAUDE_PAGA,
    CUSTO_PERICIA,
    LIMIAR_CORTE_TEORICO,
    PESO_BALANCEAMENTO_39,
    PESO_CUSTO_31,
    VALORES_CCP_ALPHA,
    RegraAtualQuilha,
    calcular_k_proporcional,
    calcular_metricas_holdout,
    construir_estimador_modelo,
    extrair_configuracao_arvore,
    gerar_buscas_parte3,
    gerar_grid_ccp_alpha,
    peso_customizado,
    reconstruir_parametros_vencedor,
    scorer_pr_auc,
    scorer_precisao_top_k,
    scorer_resultado_financeiro,
    scorer_roc_auc,
)
from quilha.pipeline import CAMINHO_PIPELINE, carregar_pipeline, salvar_pipeline


def test_regra_atual_devolve_nota_monotonica_e_zero_abaixo_do_limiar() -> None:
    X = pd.DataFrame({"valor_reclamado": [20_000.0, 50_000.0, 60_000.0, 100_000.0]})

    modelo = RegraAtualQuilha().fit(X)
    probabilidades = modelo.predict_proba(X)[:, 1]

    assert probabilidades[0] == 0.0
    assert probabilidades[1] == 0.0
    assert 0.0 < probabilidades[2] < probabilidades[3] < 1.0


def test_calcular_metricas_holdout_confere_resultado_manual() -> None:
    y_true = np.array([1, 0, 1, 0, 0])
    probabilidades = np.array([0.90, 0.80, 0.70, 0.20, 0.10])

    metricas = calcular_metricas_holdout(y_true, probabilidades, k=2)

    # Top-2 = indices 0 e 1 => 1 TP, 1 FP e 1 FN.
    assert metricas["tp_900"] == 1
    assert metricas["fp_900"] == 1
    assert metricas["fn_900"] == 1
    assert metricas["precisao_900"] == 0.5
    assert metricas["cobertura_900"] == 0.5
    assert metricas["resultado_900"] == (CUSTO_FRAUDE_PAGA * 1) - (CUSTO_PERICIA * 2)
    assert metricas["custo_total_900"] == (CUSTO_FRAUDE_PAGA * 1) + (CUSTO_PERICIA * 1)


def test_calcular_k_proporcional_confere_regra_do_prompt() -> None:
    assert calcular_k_proporcional(9_619) == 720
    assert calcular_k_proporcional(9_620) == 720
    assert calcular_k_proporcional(11_904) == 891


def test_scorers_da_cv_usam_k_proporcional_do_fold() -> None:
    y_true = np.array([1, 0, 1, 0, 0, 1])
    probabilidades = np.array([0.95, 0.90, 0.80, 0.70, 0.20, 0.10])

    resultado = scorer_resultado_financeiro(y_true, probabilidades)
    precisao = scorer_precisao_top_k(y_true, probabilidades)

    # Para 6 exemplos, k proporcional = round(900 * 6 / 12024) = 0.
    assert resultado == 0.0
    assert precisao == 0.0


def test_scorer_financeiro_confere_conta_manual_em_k_mensal() -> None:
    quantidade = 12_024
    y_true = np.zeros(quantidade, dtype=int)
    y_true[:40] = 1
    probabilidades = np.linspace(1.0, 0.0, quantidade)

    resultado = scorer_resultado_financeiro(y_true, probabilidades)

    assert resultado == (21_700 * 40) - (700 * 900)


def test_corte_teorico_corresponde_a_3_2258_por_cento() -> None:
    assert LIMIAR_CORTE_TEORICO == CUSTO_PERICIA / CUSTO_FRAUDE_PAGA
    assert round(LIMIAR_CORTE_TEORICO * 100, 4) == 3.2258


def test_artefato_campeao_preserva_predicoes_ao_salvar_e_recarregar(
    tmp_path,
) -> None:
    """O Pipeline do aplicativo deve ser o XGBoost campeão e serializar sem drift."""
    pipeline = carregar_pipeline(CAMINHO_PIPELINE)
    assert isinstance(pipeline.named_steps["modelo"], XGBClassifier)

    base = carregar_base_crua()
    X = base.drop(columns=["fraude", "valor_indenizado"]).head(32)
    probabilidades_memoria = pipeline.predict_proba(X)

    caminho = salvar_pipeline(pipeline, tmp_path / "pipeline_fraude.joblib")
    recarregado = carregar_pipeline(caminho)
    probabilidades_disco = recarregado.predict_proba(X)

    np.testing.assert_allclose(probabilidades_disco, probabilidades_memoria, rtol=0, atol=1e-12)


def test_scorers_pr_auc_e_roc_auc_extraem_coluna_positiva() -> None:
    y_true = np.array([0, 1, 0, 1])
    probabilidades = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.1, 0.9],
        ]
    )

    assert scorer_pr_auc(y_true, probabilidades) == average_precision_score(
        y_true, probabilidades[:, 1]
    )
    assert scorer_roc_auc(y_true, probabilidades) == roc_auc_score(
        y_true, probabilidades[:, 1]
    )


def test_gerar_buscas_parte3_carrega_pesos_31_e_39_11() -> None:
    buscas = gerar_buscas_parte3()

    pesos_arvore = {str(config["class_weight"]) for config in buscas["decision_tree"]}
    assert str(peso_customizado(PESO_CUSTO_31)) in pesos_arvore
    assert str(peso_customizado(PESO_BALANCEAMENTO_39)) in pesos_arvore

    pesos_xgboost = {float(config["scale_pos_weight"]) for config in buscas["xgboost"]}
    assert PESO_CUSTO_31 in pesos_xgboost
    assert PESO_BALANCEAMENTO_39 in pesos_xgboost


def test_construir_estimador_modelo_aplica_parametros_customizados() -> None:
    arvore = construir_estimador_modelo(
        "decision_tree",
        {"max_depth": 7, "class_weight": peso_customizado(PESO_CUSTO_31)},
    )
    assert arvore.max_depth == 7
    assert arvore.class_weight == peso_customizado(PESO_CUSTO_31)


def test_extrair_configuracao_arvore_recupera_vencedora_da_tabela() -> None:
    tabela_busca = pd.DataFrame(
        {
            "familia_modelo": ["decision_tree", "decision_tree", "xgboost"],
            "resultado_k_cv_media": [1_000.0, 2_000.0, 9_999.0],
            "pr_auc_cv_media": [0.10, 0.20, 0.90],
            "param_max_depth": [3.0, 9.0, 4.0],
            "param_min_samples_leaf": [1.0, 20.0, 1.0],
            "param_class_weight": ["None", str(peso_customizado(PESO_CUSTO_31)), "None"],
        }
    )

    configuracao = extrair_configuracao_arvore(tabela_busca)

    assert configuracao["max_depth"] == 9
    assert configuracao["min_samples_leaf"] == 20
    assert configuracao["class_weight"] == peso_customizado(PESO_CUSTO_31)


def test_extrair_configuracao_arvore_trata_profundidade_none() -> None:
    tabela_busca = pd.DataFrame(
        {
            "familia_modelo": ["decision_tree"],
            "resultado_k_cv_media": [2_000.0],
            "pr_auc_cv_media": [0.20],
            "param_max_depth": [np.nan],
            "param_min_samples_leaf": [5.0],
            "param_class_weight": ["balanced"],
        }
    )

    configuracao = extrair_configuracao_arvore(tabela_busca)

    assert configuracao["max_depth"] is None
    assert configuracao["class_weight"] == "balanced"


def test_extrair_configuracao_arvore_trata_class_weight_ausente_vindo_de_csv() -> None:
    tabela_busca = pd.DataFrame(
        {
            "familia_modelo": ["decision_tree"],
            "resultado_k_cv_media": [2_000.0],
            "pr_auc_cv_media": [0.20],
            "param_max_depth": [10.0],
            "param_min_samples_leaf": [20.0],
            "param_class_weight": [float("nan")],
        }
    )

    configuracao = extrair_configuracao_arvore(tabela_busca)

    assert configuracao["class_weight"] is None
    assert configuracao["max_depth"] == 10


def test_gerar_grid_ccp_alpha_cobre_a_varredura_de_poda() -> None:
    base = {"max_depth": 9, "min_samples_leaf": 20, "class_weight": None}

    grade = gerar_grid_ccp_alpha(base)

    assert len(grade) == len(VALORES_CCP_ALPHA)
    assert all(config["ccp_alpha"] in VALORES_CCP_ALPHA for config in grade)
    assert all(config["max_depth"] == 9 for config in grade)
    assert grade[0]["ccp_alpha"] == 0.0


def test_reconstruir_parametros_vencedor_devolve_contagem_como_int() -> None:
    """O CSV devolve `10.0`; o sklearn exige `int` em `max_depth` e `n_estimators`."""
    linha = pd.Series(
        {
            "familia_modelo": "xgboost",
            "param_max_depth": 4.0,
            "param_n_estimators": 350.0,
            "param_learning_rate": 0.03,
            "param_subsample": 0.85,
            "param_scale_pos_weight": 1.0,
        }
    )

    parametros = reconstruir_parametros_vencedor(linha)

    assert parametros["max_depth"] == 4
    assert isinstance(parametros["max_depth"], int)
    assert parametros["n_estimators"] == 350
    assert isinstance(parametros["n_estimators"], int)
    assert parametros["learning_rate"] == 0.03
    assert parametros["subsample"] == 0.85
    assert "familia_modelo" not in parametros


def test_reconstruir_parametros_vencedor_descarta_ausentes_e_le_peso_do_csv() -> None:
    linha = pd.Series(
        {
            "param_max_depth": float("nan"),
            "param_min_samples_leaf": 20.0,
            "param_class_weight": str(peso_customizado(PESO_CUSTO_31)),
        }
    )

    parametros = reconstruir_parametros_vencedor(linha)

    assert "max_depth" not in parametros
    assert parametros["min_samples_leaf"] == 20
    assert parametros["class_weight"] == peso_customizado(PESO_CUSTO_31)

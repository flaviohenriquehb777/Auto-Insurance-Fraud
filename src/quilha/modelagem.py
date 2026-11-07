"""Modelagem da Fase 4: validação, seleção financeira e explicabilidade.

Este módulo reúne as rotinas das Partes 1 a 5 do prompt:

* compara seis baselines com o mesmo protocolo temporal e validação cruzada;
* ajusta hiperparâmetros e testa pesos de classe e estratégias de subamostragem;
* seleciona o campeão por Resultado Líquido@900 e mede robustez em cinco sementes;
* decompõe a função de custo 31:1 e gera explicações SHAP globais e locais;
* registra os Pipelines completos no MLflow para integração com o aplicativo.

O ajuste de qualquer estimador ocorre apenas sobre os dados de treino ou dentro
dos folds de validação. O holdout temporal é reservado para a avaliação final.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, make_scorer, roc_auc_score
from sklearn.model_selection import (
    ParameterGrid,
    ParameterSampler,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from quilha.amostragem import (
    EstrategiaAmostragem,
    ModeloComSubamostragem,
    estrategias_parte4,
)
from quilha.dados import PASTA_FIGURAS, RAIZ_PROJETO, carregar_base_crua, dividir_temporal
from quilha.explicacao import contribuicoes_agregadas, explicar_sinistro
from quilha.mlops import contexto_run, nome_run
from quilha.pipeline import construir_pipeline
from quilha.preparacao import FEATURES

SEED = 42
N_SPLITS_CV = 5
CAPACIDADE_MENSAL = 900
SINISTROS_POR_MES = 12_024
LIMIAR_REGRA_ATUAL = 50_000.0
CUSTO_PERICIA = 700.0
CUSTO_FRAUDE_PAGA = 21_700.0
COLUNAS_EXCLUIDAS_ENTRADA = ("fraude", "valor_indenizado")
PESO_CUSTO_31 = 31.0
PESO_BALANCEAMENTO_39 = 39.11
N_ITER_RANDOM_SEARCH = 12

# Varredura de poda pós-treino (cost-complexity) da árvore vencedora.
VALORES_CCP_ALPHA: tuple[float, ...] = (
    0.0,
    0.00001,
    0.00005,
    0.0001,
    0.0005,
    0.001,
    0.005,
    0.01,
)

# Hiperparâmetros que são contagem e precisam voltar como `int` ao sair do CSV.
PARAMETROS_INTEIROS: frozenset[str] = frozenset(
    {
        "max_depth",
        "min_samples_leaf",
        "min_samples_split",
        "min_child_weight",
        "min_child_samples",
        "max_leaf_nodes",
        "n_estimators",
        "num_leaves",
    }
)


@dataclass(frozen=True)
class ResultadoModelo:
    """Resultado consolidado de um modelo no holdout temporal."""

    modelo: str
    papel: str
    pr_auc_holdout: float
    roc_auc_holdout: float
    brier_holdout: float
    precisao_900_holdout: float
    cobertura_900_holdout: float
    resultado_900_holdout: float
    custo_total_900_holdout: float
    tp_900_holdout: int
    fp_900_holdout: int
    fn_900_holdout: int
    run_id: str | None
    uri_modelo: str | None


@dataclass(frozen=True)
class ResultadoModeloValidado:
    """Resultado consolidado com CV no treino e holdout temporal."""

    modelo: str
    papel: str
    k_fold: int
    pr_auc_cv_media: float
    pr_auc_cv_desvio: float
    roc_auc_cv_media: float
    roc_auc_cv_desvio: float
    precisao_k_cv_media: float
    precisao_k_cv_desvio: float
    resultado_k_cv_media: float
    resultado_k_cv_desvio: float
    pr_auc_holdout: float
    roc_auc_holdout: float
    brier_holdout: float
    precisao_900_holdout: float
    cobertura_900_holdout: float
    resultado_900_holdout: float
    custo_total_900_holdout: float
    tp_900_holdout: int
    fp_900_holdout: int
    fn_900_holdout: int
    run_id: str | None
    uri_modelo: str | None


@dataclass(frozen=True)
class ResultadoBusca:
    """Resultado de uma configuração candidata na Parte 3."""

    familia_modelo: str
    iteracao: int
    parametros_modelo: dict
    origem_pesos: str
    n_folhas_holdout: float | None
    k_fold: int
    pr_auc_cv_media: float
    pr_auc_cv_desvio: float
    roc_auc_cv_media: float
    roc_auc_cv_desvio: float
    precisao_k_cv_media: float
    precisao_k_cv_desvio: float
    resultado_k_cv_media: float
    resultado_k_cv_desvio: float
    pr_auc_holdout: float
    roc_auc_holdout: float
    brier_holdout: float
    precisao_900_holdout: float
    cobertura_900_holdout: float
    resultado_900_holdout: float
    custo_total_900_holdout: float
    tp_900_holdout: int
    fp_900_holdout: int
    fn_900_holdout: int
    run_id: str | None
    uri_modelo: str | None


class RegraAtualQuilha(BaseEstimator, ClassifierMixin):
    """Baseline de negócio da Quilha, compatível com o protocolo dos demais modelos.

    A regra atual prioriza sinistros com `valor_reclamado` acima de R$ 50 mil.
    Para entrar no mesmo protocolo de métricas e ranking, ela devolve uma nota
    monotônica no `predict_proba`, zerada abaixo do limiar.
    """

    def __init__(self, limiar_valor: float = LIMIAR_REGRA_ATUAL):
        self.limiar_valor = limiar_valor

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401, ANN001 - assinatura sklearn
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        valores = self._extrair_valor_reclamado(X)
        notas = np.where(
            valores > self.limiar_valor,
            1.0 - (self.limiar_valor / valores),
            0.0,
        )
        notas = np.nan_to_num(notas, nan=0.0, posinf=1.0, neginf=0.0)
        notas = np.clip(notas, 0.0, 1.0)
        return np.column_stack([1.0 - notas, notas])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0).astype(int)

    @staticmethod
    def _extrair_valor_reclamado(X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("A Regra Atual da Quilha espera um DataFrame com nomes de colunas.")
        if "valor_reclamado" in X.columns:
            coluna = "valor_reclamado"
        else:
            candidatas = [nome for nome in X.columns if nome.endswith("__valor_reclamado")]
            if not candidatas:
                raise KeyError("A coluna 'valor_reclamado' é obrigatória para a Regra Atual.")
            coluna = candidatas[0]
        return pd.to_numeric(X[coluna], errors="coerce").fillna(0.0).to_numpy()


def separar_entrada_alvo(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa o chamado cru (`X`) do alvo binário (`y`)."""
    X = base.drop(columns=list(COLUNAS_EXCLUIDAS_ENTRADA), errors="ignore").copy()
    y = pd.to_numeric(base["fraude"], errors="raise").astype(int)
    return X, y


def carregar_treino_teste() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Carrega a base crua e aplica a divisão temporal oficial do projeto."""
    base = carregar_base_crua()
    treino, teste = dividir_temporal(base)
    X_treino, y_treino = separar_entrada_alvo(treino)
    X_teste, y_teste = separar_entrada_alvo(teste)
    return X_treino, y_treino, X_teste, y_teste


def construir_validacao_cruzada(
    n_splits: int = N_SPLITS_CV,
    seed: int = SEED,
) -> StratifiedKFold:
    """Validação cruzada oficial da Fase 4, restrita ao treino."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def calcular_k_proporcional(
    quantidade_sinistros: int,
    capacidade_mensal: int = CAPACIDADE_MENSAL,
    sinistros_por_mes: int = SINISTROS_POR_MES,
) -> int:
    """Converte capacidade mensal em `k` proporcional ao tamanho do fold."""
    return int(round(capacidade_mensal * quantidade_sinistros / sinistros_por_mes))


def construir_modelos_base(seed: int = SEED) -> dict[str, tuple[str, BaseEstimator]]:
    """Define o elenco base da Parte 1, sem tuning e com `random_state=42`."""
    return {
        "dummy_prior": (
            'Referência simples (DummyClassifier(strategy="prior"))',
            DummyClassifier(strategy="prior"),
        ),
        "regra_atual_quilha": (
            "Baseline de negócio",
            RegraAtualQuilha(),
        ),
        "decision_tree": (
            "Interpretabilidade / profundidade ótima",
            DecisionTreeClassifier(random_state=seed),
        ),
        "random_forest": (
            "Bagging",
            RandomForestClassifier(random_state=seed, n_jobs=-1),
        ),
        "xgboost": (
            "Boosting",
            XGBClassifier(
                random_state=seed,
                eval_metric="logloss",
                n_jobs=1,
            ),
        ),
        "lightgbm": (
            "Boosting",
            LGBMClassifier(
                random_state=seed,
                verbose=-1,
            ),
        ),
    }


def peso_customizado(valor: float) -> dict[int, float]:
    """Converte um peso positivo em `class_weight` binário explícito."""
    return {0: 1.0, 1: float(valor)}


def descrever_peso(parametros_modelo: dict) -> str:
    """Identifica a origem do rebalanceamento para facilitar a leitura do resultado."""
    if "class_weight" in parametros_modelo:
        class_weight = parametros_modelo["class_weight"]
        if class_weight == "balanced":
            return "balanced"
        if class_weight == "balanced_subsample":
            return "balanced_subsample"
        if class_weight == peso_customizado(PESO_CUSTO_31):
            return "peso_31"
        if class_weight == peso_customizado(PESO_BALANCEAMENTO_39):
            return "peso_39_11"
        if class_weight is None:
            return "sem_peso"
        return str(class_weight)

    if "scale_pos_weight" in parametros_modelo:
        peso = float(parametros_modelo["scale_pos_weight"])
        if np.isclose(peso, PESO_CUSTO_31):
            return "peso_31"
        if np.isclose(peso, PESO_BALANCEAMENTO_39):
            return "peso_39_11"
        if np.isclose(peso, 1.0):
            return "sem_peso"
        return f"scale_pos_weight_{peso}"

    return "sem_peso"


def _probabilidade_positiva(probabilidades: np.ndarray) -> np.ndarray:
    """Extrai a coluna da classe positiva de qualquer saída `predict_proba`."""
    proba_array = np.asarray(probabilidades, dtype=float)
    if proba_array.ndim == 2:
        return proba_array[:, 1]
    return proba_array


def calcular_metricas_holdout(
    y_true: pd.Series | np.ndarray,
    probabilidades: np.ndarray,
    k: int = CAPACIDADE_MENSAL,
) -> dict[str, float | int]:
    """Calcula métricas oficiais da Parte 1 no holdout temporal."""
    y_array = np.asarray(y_true, dtype=int)
    proba_array = _probabilidade_positiva(probabilidades)
    k_ajustado = min(k, len(y_array))
    if k_ajustado <= 0:
        positivos = int(y_array.sum())
        return {
            "pr_auc": float(average_precision_score(y_array, proba_array)),
            "roc_auc": float(roc_auc_score(y_array, proba_array)),
            "brier": float(brier_score_loss(y_array, proba_array)),
            "precisao_900": 0.0,
            "cobertura_900": 0.0,
            "resultado_900": 0.0,
            "custo_total_900": float(CUSTO_FRAUDE_PAGA * positivos),
            "tp_900": 0,
            "fp_900": 0,
            "fn_900": positivos,
        }
    ordem = np.argsort(-proba_array, kind="mergesort")
    selecionados = ordem[:k_ajustado]

    tp = int(y_array[selecionados].sum())
    fp = int(k_ajustado - tp)
    positivos = int(y_array.sum())
    fn = int(positivos - tp)

    return {
        "pr_auc": float(average_precision_score(y_array, proba_array)),
        "roc_auc": float(roc_auc_score(y_array, proba_array)),
        "brier": float(brier_score_loss(y_array, proba_array)),
        "precisao_900": float(tp / k_ajustado),
        "cobertura_900": float(tp / positivos if positivos else 0.0),
        "resultado_900": float((CUSTO_FRAUDE_PAGA * tp) - (CUSTO_PERICIA * k_ajustado)),
        "custo_total_900": float((CUSTO_FRAUDE_PAGA * fn) + (CUSTO_PERICIA * fp)),
        "tp_900": tp,
        "fp_900": fp,
        "fn_900": fn,
    }


def scorer_resultado_financeiro(
    y_true: pd.Series | np.ndarray,
    probabilidades: np.ndarray,
) -> float:
    """Scorer financeiro para `make_scorer`, com `k` proporcional ao fold."""
    k_fold = calcular_k_proporcional(len(y_true))
    metricas = calcular_metricas_holdout(y_true, probabilidades, k=k_fold)
    return float(metricas["resultado_900"])


def scorer_pr_auc(
    y_true: pd.Series | np.ndarray,
    probabilidades: np.ndarray,
) -> float:
    """PR AUC sobre a coluna da classe positiva."""
    y_array = np.asarray(y_true, dtype=int)
    proba_array = _probabilidade_positiva(probabilidades)
    return float(average_precision_score(y_array, proba_array))


def scorer_roc_auc(
    y_true: pd.Series | np.ndarray,
    probabilidades: np.ndarray,
) -> float:
    """ROC AUC sobre a coluna da classe positiva."""
    y_array = np.asarray(y_true, dtype=int)
    proba_array = _probabilidade_positiva(probabilidades)
    return float(roc_auc_score(y_array, proba_array))


def scorer_precisao_top_k(
    y_true: pd.Series | np.ndarray,
    probabilidades: np.ndarray,
) -> float:
    """Precisão no topo da fila proporcional ao tamanho do fold."""
    k_fold = calcular_k_proporcional(len(y_true))
    metricas = calcular_metricas_holdout(y_true, probabilidades, k=k_fold)
    return float(metricas["precisao_900"])


def scorers_validacao() -> dict[str, str | object]:
    """Pacote oficial de scorers da Parte 2."""
    return {
        "pr_auc": make_scorer(
            scorer_pr_auc,
            response_method="predict_proba",
            greater_is_better=True,
        ),
        "roc_auc": make_scorer(
            scorer_roc_auc,
            response_method="predict_proba",
            greater_is_better=True,
        ),
        "resultado": make_scorer(
            scorer_resultado_financeiro,
            response_method="predict_proba",
            greater_is_better=True,
        ),
        "precisao_k": make_scorer(
            scorer_precisao_top_k,
            response_method="predict_proba",
            greater_is_better=True,
        ),
    }


def executar_validacao_cruzada(
    estimador: BaseEstimator,
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    seed: int = SEED,
    return_estimator: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Executa a CV oficial da fase com o Pipeline inteiro dentro de cada fold."""
    cv = construir_validacao_cruzada(seed=seed)
    pipeline = construir_pipeline(modelo=clone(estimador))
    resultados_cv = cross_validate(
        estimator=pipeline,
        X=X_treino,
        y=y_treino,
        scoring=scorers_validacao(),
        cv=cv,
        return_estimator=return_estimator,
        n_jobs=1,
    )

    tabela_folds = pd.DataFrame(
        {
            "fold": np.arange(1, len(resultados_cv["test_pr_auc"]) + 1),
            "n_validacao": [len(indices) for _, indices in cv.split(X_treino, y_treino)],
            "k_fold": [
                calcular_k_proporcional(len(indices))
                for _, indices in cv.split(X_treino, y_treino)
            ],
            "fit_time": resultados_cv["fit_time"],
            "score_time": resultados_cv["score_time"],
            "pr_auc": resultados_cv["test_pr_auc"],
            "roc_auc": resultados_cv["test_roc_auc"],
            "precisao_k": resultados_cv["test_precisao_k"],
            "resultado": resultados_cv["test_resultado"],
        }
    )
    return tabela_folds, resultados_cv


def resumir_validacao_cruzada(tabela_folds: pd.DataFrame) -> dict[str, float | int]:
    """Resume a CV em média ± desvio e explicita o `k_fold` usado."""
    return {
        "k_fold": int(tabela_folds["k_fold"].iloc[0]),
        "pr_auc_cv_media": float(tabela_folds["pr_auc"].mean()),
        "pr_auc_cv_desvio": float(tabela_folds["pr_auc"].std(ddof=1)),
        "roc_auc_cv_media": float(tabela_folds["roc_auc"].mean()),
        "roc_auc_cv_desvio": float(tabela_folds["roc_auc"].std(ddof=1)),
        "precisao_k_cv_media": float(tabela_folds["precisao_k"].mean()),
        "precisao_k_cv_desvio": float(tabela_folds["precisao_k"].std(ddof=1)),
        "resultado_k_cv_media": float(tabela_folds["resultado"].mean()),
        "resultado_k_cv_desvio": float(tabela_folds["resultado"].std(ddof=1)),
    }


def construir_estimador_modelo(
    familia_modelo: str,
    parametros_modelo: dict | None = None,
    seed: int = SEED,
) -> BaseEstimator:
    """Monta um estimador da família pedida com os hiperparâmetros informados."""
    parametros = dict(parametros_modelo or {})
    if familia_modelo == "decision_tree":
        return DecisionTreeClassifier(random_state=seed, **parametros)
    if familia_modelo == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **parametros)
    if familia_modelo == "xgboost":
        return XGBClassifier(random_state=seed, eval_metric="logloss", n_jobs=1, **parametros)
    if familia_modelo == "lightgbm":
        return LGBMClassifier(random_state=seed, verbose=-1, **parametros)
    raise ValueError(f"Família de modelo não suportada na Parte 3: {familia_modelo}")


def _amostrar_configs_lista(
    configuracoes: list[dict],
    n_iter: int,
    seed: int,
) -> list[dict]:
    """Amostra sem reposição um subconjunto reprodutível de configurações."""
    if len(configuracoes) <= n_iter:
        return configuracoes
    gerador = np.random.default_rng(seed)
    indices = gerador.choice(len(configuracoes), size=n_iter, replace=False)
    return [configuracoes[int(indice)] for indice in indices]


def gerar_buscas_parte3(seed: int = SEED) -> dict[str, list[dict]]:
    """Define as buscas da Parte 3, incluindo pesos 31 e 39,11."""
    configs_decision_tree = list(
        ParameterGrid(
            {
                "max_depth": [*range(1, 21), None],
                "min_samples_leaf": [1, 5, 20, 50],
                "class_weight": [
                    None,
                    "balanced",
                    peso_customizado(PESO_CUSTO_31),
                    peso_customizado(PESO_BALANCEAMENTO_39),
                ],
            }
        )
    )
    configs_random_forest = list(
        ParameterSampler(
            {
                "n_estimators": [300, 500],
                "max_depth": [None, 5, 10, 15, 20],
                "min_samples_leaf": [1, 5, 20],
                "class_weight": [
                    "balanced",
                    "balanced_subsample",
                    peso_customizado(PESO_CUSTO_31),
                    peso_customizado(PESO_BALANCEAMENTO_39),
                ],
            },
            n_iter=N_ITER_RANDOM_SEARCH,
            random_state=seed,
        )
    )
    configs_xgboost = list(
        ParameterSampler(
            {
                "n_estimators": [200, 350, 500],
                "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [3, 4, 5, 6],
                "min_child_weight": [1, 3, 5],
                "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0],
                "gamma": [0.0, 1.0, 3.0],
                "reg_lambda": [1.0, 3.0, 10.0],
                "scale_pos_weight": [1.0, PESO_CUSTO_31, PESO_BALANCEAMENTO_39],
            },
            n_iter=N_ITER_RANDOM_SEARCH,
            random_state=seed,
        )
    )
    pool_lightgbm = list(
        ParameterGrid(
            {
                "num_leaves": [15, 31, 63],
                "learning_rate": [0.03, 0.05, 0.1],
                "n_estimators": [200, 350, 500],
                "min_child_samples": [20, 50, 100],
                "feature_fraction": [0.7, 0.85, 1.0],
                "bagging_fraction": [0.7, 0.85, 1.0],
                "lambda_l2": [0.0, 1.0, 5.0, 10.0],
                "scale_pos_weight": [1.0, PESO_CUSTO_31, PESO_BALANCEAMENTO_39],
                "is_unbalance": [False],
            }
        )
    )
    configs_lightgbm = _amostrar_configs_lista(pool_lightgbm, N_ITER_RANDOM_SEARCH, seed)
    return {
        "decision_tree": configs_decision_tree,
        "random_forest": configs_random_forest,
        "xgboost": configs_xgboost,
        "lightgbm": configs_lightgbm,
    }


def extrair_numero_folhas(estimador: BaseEstimator) -> float | None:
    """Resume o tamanho da árvore ajustada para apoiar a leitura da curva de profundidade."""
    if isinstance(estimador, DecisionTreeClassifier):
        return float(estimador.get_n_leaves())
    if isinstance(estimador, RandomForestClassifier):
        folhas = [arvore.get_n_leaves() for arvore in estimador.estimators_]
        return float(np.mean(folhas))
    return None


def avaliar_modelo_holdout(
    nome_modelo: str,
    papel: str,
    estimador: BaseEstimator,
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
    seed: int = SEED,
) -> ResultadoModelo:
    """Ajusta o Pipeline, mede o holdout e registra a run no MLflow."""
    pipeline = construir_pipeline(modelo=estimador)
    parametros = {
        "familia_modelo": nome_modelo,
        "papel_modelo": papel,
        "seed": seed,
        "capacidade_mensal": CAPACIDADE_MENSAL,
        "sinistros_por_mes": SINISTROS_POR_MES,
        "features": list(FEATURES),
        **pipeline.named_steps["modelo"].get_params(deep=False),
    }
    with contexto_run(
        nome=nome_run(nome_modelo, "parte1_base", seed),
        parametros=parametros,
        tags={"fase": "4", "parte": "1", "modelo": nome_modelo},
    ) as registro:
        pipeline.fit(X_treino, y_treino)
        probabilidades = pipeline.predict_proba(X_teste)[:, 1]
        metricas = calcular_metricas_holdout(y_teste, probabilidades)
        registro.log_metricas(metricas, prefixo="holdout_")
        registro.log_json(metricas, "metricas_holdout.json")
        uri_modelo = registro.log_pipeline(pipeline, X_treino.head(1))
        return ResultadoModelo(
            modelo=nome_modelo,
            papel=papel,
            pr_auc_holdout=float(metricas["pr_auc"]),
            roc_auc_holdout=float(metricas["roc_auc"]),
            brier_holdout=float(metricas["brier"]),
            precisao_900_holdout=float(metricas["precisao_900"]),
            cobertura_900_holdout=float(metricas["cobertura_900"]),
            resultado_900_holdout=float(metricas["resultado_900"]),
            custo_total_900_holdout=float(metricas["custo_total_900"]),
            tp_900_holdout=int(metricas["tp_900"]),
            fp_900_holdout=int(metricas["fp_900"]),
            fn_900_holdout=int(metricas["fn_900"]),
            run_id=registro.run_id,
            uri_modelo=uri_modelo,
        )


def avaliar_modelo_validado(
    nome_modelo: str,
    papel: str,
    estimador: BaseEstimator,
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
    seed: int = SEED,
) -> ResultadoModeloValidado:
    """Executa a Parte 2: CV no treino + holdout final no mesmo protocolo."""
    tabela_folds, _ = executar_validacao_cruzada(
        estimador=estimador,
        X_treino=X_treino,
        y_treino=y_treino,
        seed=seed,
    )
    resumo_cv = resumir_validacao_cruzada(tabela_folds)

    pipeline = construir_pipeline(modelo=clone(estimador))
    parametros = {
        "familia_modelo": nome_modelo,
        "papel_modelo": papel,
        "seed": seed,
        "n_splits_cv": N_SPLITS_CV,
        "capacidade_mensal": CAPACIDADE_MENSAL,
        "sinistros_por_mes": SINISTROS_POR_MES,
        "k_fold": resumo_cv["k_fold"],
        "features": list(FEATURES),
        **pipeline.named_steps["modelo"].get_params(deep=False),
    }
    with contexto_run(
        nome=nome_run(nome_modelo, "parte2_cv_holdout", seed),
        parametros=parametros,
        tags={"fase": "4", "parte": "2", "modelo": nome_modelo},
    ) as registro:
        registro.log_json(tabela_folds.to_dict(orient="records"), "tabela_cv.json")
        registro.log_metricas(resumo_cv, prefixo="cv_")

        pipeline.fit(X_treino, y_treino)
        probabilidades_holdout = pipeline.predict_proba(X_teste)[:, 1]
        metricas_holdout = calcular_metricas_holdout(y_teste, probabilidades_holdout)
        registro.log_metricas(metricas_holdout, prefixo="holdout_")
        registro.log_json(metricas_holdout, "metricas_holdout.json")
        uri_modelo = registro.log_pipeline(pipeline, X_treino.head(1))

        return ResultadoModeloValidado(
            modelo=nome_modelo,
            papel=papel,
            k_fold=int(resumo_cv["k_fold"]),
            pr_auc_cv_media=float(resumo_cv["pr_auc_cv_media"]),
            pr_auc_cv_desvio=float(resumo_cv["pr_auc_cv_desvio"]),
            roc_auc_cv_media=float(resumo_cv["roc_auc_cv_media"]),
            roc_auc_cv_desvio=float(resumo_cv["roc_auc_cv_desvio"]),
            precisao_k_cv_media=float(resumo_cv["precisao_k_cv_media"]),
            precisao_k_cv_desvio=float(resumo_cv["precisao_k_cv_desvio"]),
            resultado_k_cv_media=float(resumo_cv["resultado_k_cv_media"]),
            resultado_k_cv_desvio=float(resumo_cv["resultado_k_cv_desvio"]),
            pr_auc_holdout=float(metricas_holdout["pr_auc"]),
            roc_auc_holdout=float(metricas_holdout["roc_auc"]),
            brier_holdout=float(metricas_holdout["brier"]),
            precisao_900_holdout=float(metricas_holdout["precisao_900"]),
            cobertura_900_holdout=float(metricas_holdout["cobertura_900"]),
            resultado_900_holdout=float(metricas_holdout["resultado_900"]),
            custo_total_900_holdout=float(metricas_holdout["custo_total_900"]),
            tp_900_holdout=int(metricas_holdout["tp_900"]),
            fp_900_holdout=int(metricas_holdout["fp_900"]),
            fn_900_holdout=int(metricas_holdout["fn_900"]),
            run_id=registro.run_id,
            uri_modelo=uri_modelo,
        )


def avaliar_configuracao_busca(
    familia_modelo: str,
    parametros_modelo: dict,
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
    iteracao: int,
    seed: int = SEED,
    rotulo_run: str = "parte3",
) -> ResultadoBusca:
    """Avalia uma configuração candidata da Parte 3 com CV e holdout."""
    estimador = construir_estimador_modelo(
        familia_modelo=familia_modelo,
        parametros_modelo=parametros_modelo,
        seed=seed,
    )
    tabela_folds, _ = executar_validacao_cruzada(
        estimador=estimador,
        X_treino=X_treino,
        y_treino=y_treino,
        seed=seed,
        return_estimator=False,
    )
    resumo_cv = resumir_validacao_cruzada(tabela_folds)

    pipeline = construir_pipeline(modelo=clone(estimador))
    parametros_run = {
        "familia_modelo": familia_modelo,
        "iteracao_busca": iteracao,
        "seed": seed,
        "n_splits_cv": N_SPLITS_CV,
        "n_iter_random_search": N_ITER_RANDOM_SEARCH,
        "capacidade_mensal": CAPACIDADE_MENSAL,
        "sinistros_por_mes": SINISTROS_POR_MES,
        "k_fold": resumo_cv["k_fold"],
        "origem_pesos": descrever_peso(parametros_modelo),
        "features": list(FEATURES),
        **parametros_modelo,
    }
    with contexto_run(
        nome=nome_run(familia_modelo, f"{rotulo_run}_iter{iteracao:03d}", seed),
        parametros=parametros_run,
        tags={"fase": "4", "parte": "3", "modelo": familia_modelo, "rotulo": rotulo_run},
    ) as registro:
        registro.log_json(tabela_folds.to_dict(orient="records"), "tabela_cv.json")
        registro.log_metricas(resumo_cv, prefixo="cv_")

        pipeline.fit(X_treino, y_treino)
        probabilidades_holdout = pipeline.predict_proba(X_teste)[:, 1]
        metricas_holdout = calcular_metricas_holdout(y_teste, probabilidades_holdout)
        n_folhas_holdout = extrair_numero_folhas(pipeline.named_steps["modelo"])
        registro.log_metricas(metricas_holdout, prefixo="holdout_")
        registro.log_json(metricas_holdout, "metricas_holdout.json")

        return ResultadoBusca(
            familia_modelo=familia_modelo,
            iteracao=iteracao,
            parametros_modelo=dict(parametros_modelo),
            origem_pesos=descrever_peso(parametros_modelo),
            n_folhas_holdout=n_folhas_holdout,
            k_fold=int(resumo_cv["k_fold"]),
            pr_auc_cv_media=float(resumo_cv["pr_auc_cv_media"]),
            pr_auc_cv_desvio=float(resumo_cv["pr_auc_cv_desvio"]),
            roc_auc_cv_media=float(resumo_cv["roc_auc_cv_media"]),
            roc_auc_cv_desvio=float(resumo_cv["roc_auc_cv_desvio"]),
            precisao_k_cv_media=float(resumo_cv["precisao_k_cv_media"]),
            precisao_k_cv_desvio=float(resumo_cv["precisao_k_cv_desvio"]),
            resultado_k_cv_media=float(resumo_cv["resultado_k_cv_media"]),
            resultado_k_cv_desvio=float(resumo_cv["resultado_k_cv_desvio"]),
            pr_auc_holdout=float(metricas_holdout["pr_auc"]),
            roc_auc_holdout=float(metricas_holdout["roc_auc"]),
            brier_holdout=float(metricas_holdout["brier"]),
            precisao_900_holdout=float(metricas_holdout["precisao_900"]),
            cobertura_900_holdout=float(metricas_holdout["cobertura_900"]),
            resultado_900_holdout=float(metricas_holdout["resultado_900"]),
            custo_total_900_holdout=float(metricas_holdout["custo_total_900"]),
            tp_900_holdout=int(metricas_holdout["tp_900"]),
            fp_900_holdout=int(metricas_holdout["fp_900"]),
            fn_900_holdout=int(metricas_holdout["fn_900"]),
            run_id=registro.run_id,
            uri_modelo=None,
        )


def _expandir_parametros_busca(tabela_resultados: pd.DataFrame) -> pd.DataFrame:
    """Abre o dicionário de hiperparâmetros em colunas para facilitar análise e gráficos."""
    parametros = tabela_resultados["parametros_modelo"].apply(pd.Series).add_prefix("param_")
    tabela = pd.concat([tabela_resultados.drop(columns=["parametros_modelo"]), parametros], axis=1)
    if "param_class_weight" in tabela.columns:
        tabela["param_class_weight"] = tabela["param_class_weight"].astype(str)
    return tabela


def salvar_curvas_profundidade(tabela_busca: pd.DataFrame) -> list[str]:
    """Salva as curvas de profundidade da árvore e da floresta para leitura do comitê."""
    caminhos_salvos: list[str] = []
    for familia_modelo, nome_arquivo in (
        ("decision_tree", "04_01_decision_tree_profundidade.png"),
        ("random_forest", "04_02_random_forest_profundidade.png"),
    ):
        tabela_familia = tabela_busca.loc[tabela_busca["familia_modelo"] == familia_modelo].copy()
        if tabela_familia.empty or "param_max_depth" not in tabela_familia.columns:
            continue
        tabela_familia["param_max_depth_plot"] = tabela_familia["param_max_depth"].fillna(21)
        melhor_por_profundidade = (
            tabela_familia.sort_values("resultado_k_cv_media", ascending=False)
            .drop_duplicates(subset=["param_max_depth_plot"])
            .sort_values("param_max_depth_plot")
        )

        PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
        caminho = PASTA_FIGURAS / nome_arquivo
        figura, eixo_esquerdo = plt.subplots(figsize=(10, 5))
        eixo_direito = eixo_esquerdo.twinx()

        eixo_esquerdo.plot(
            melhor_por_profundidade["param_max_depth_plot"],
            melhor_por_profundidade["resultado_k_cv_media"],
            marker="o",
            color="#1f77b4",
            label="Resultado@k CV",
        )
        eixo_direito.plot(
            melhor_por_profundidade["param_max_depth_plot"],
            melhor_por_profundidade["pr_auc_cv_media"],
            marker="s",
            color="#ff7f0e",
            label="PR AUC CV",
        )
        eixo_esquerdo.set_xlabel("max_depth (21 = None)")
        eixo_esquerdo.set_ylabel("Resultado@k CV (R$)")
        eixo_direito.set_ylabel("PR AUC CV")
        eixo_esquerdo.set_title(f"Curva de profundidade - {familia_modelo}")
        figura.tight_layout()
        figura.savefig(caminho, dpi=150, bbox_inches="tight")
        plt.close(figura)
        caminhos_salvos.append(str(caminho))

    return caminhos_salvos


def rodar_ajuste_hiperparametros(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Executa a Parte 3 para os quatro modelos ajustáveis e devolve ranking + melhores."""
    X_treino, y_treino, X_teste, y_teste = carregar_treino_teste()
    configuracoes_busca = gerar_buscas_parte3(seed=seed)
    resultados: list[ResultadoBusca] = []

    for familia_modelo, configuracoes in configuracoes_busca.items():
        for iteracao, parametros_modelo in enumerate(configuracoes, start=1):
            resultados.append(
                avaliar_configuracao_busca(
                    familia_modelo=familia_modelo,
                    parametros_modelo=parametros_modelo,
                    X_treino=X_treino,
                    y_treino=y_treino,
                    X_teste=X_teste,
                    y_teste=y_teste,
                    iteracao=iteracao,
                    seed=seed,
                )
            )

    tabela_resultados = pd.DataFrame([resultado.__dict__ for resultado in resultados])
    tabela_resultados = _expandir_parametros_busca(tabela_resultados)
    tabela_resultados = tabela_resultados.sort_values(
        by=["resultado_k_cv_media", "pr_auc_cv_media", "resultado_900_holdout"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    melhores_por_familia = (
        tabela_resultados.sort_values(
            by=["familia_modelo", "resultado_k_cv_media", "pr_auc_cv_media"],
            ascending=[True, False, False],
        )
        .drop_duplicates(subset=["familia_modelo"])
        .reset_index(drop=True)
    )
    caminhos_curvas = salvar_curvas_profundidade(tabela_resultados)
    return tabela_resultados, melhores_por_familia, caminhos_curvas


def _desserializar_parametro_expandido(valor: object) -> object:
    """Recupera o valor original de um parâmetro que a tabela expandida virou texto.

    Trata também o caso de leitura via CSV: o pandas converte a string `"None"`
    em valor ausente (`NaN`), que precisa voltar a ser `None` para o estimador.
    """
    if valor is None:
        return None
    if isinstance(valor, float) and pd.isna(valor):
        return None
    if isinstance(valor, str):
        if valor == "None":
            return None
        try:
            return ast.literal_eval(valor)
        except (ValueError, SyntaxError):
            return valor
    return valor


def extrair_configuracao_arvore(tabela_busca: pd.DataFrame) -> dict:
    """Recupera a configuração vencedora da árvore, base para a poda pós-treino."""
    tabela_arvore = tabela_busca.loc[tabela_busca["familia_modelo"] == "decision_tree"]
    if tabela_arvore.empty:
        raise ValueError("A tabela de busca não contém resultados de decision_tree.")

    linha = tabela_arvore.sort_values(
        by=["resultado_k_cv_media", "pr_auc_cv_media"],
        ascending=[False, False],
    ).iloc[0]

    profundidade = _desserializar_parametro_expandido(linha.get("param_max_depth"))
    if profundidade is None or pd.isna(profundidade):
        profundidade = None
    else:
        profundidade = int(profundidade)

    return {
        "max_depth": profundidade,
        "min_samples_leaf": int(linha["param_min_samples_leaf"]),
        "class_weight": _desserializar_parametro_expandido(linha.get("param_class_weight")),
    }


def gerar_grid_ccp_alpha(
    configuracao_base: dict,
    valores: tuple[float, ...] = VALORES_CCP_ALPHA,
) -> list[dict]:
    """Cruza a configuração vencedora da árvore com a varredura de `ccp_alpha`."""
    return [{**configuracao_base, "ccp_alpha": float(valor)} for valor in valores]


def rodar_poda_custo_complexidade(
    tabela_busca: pd.DataFrame,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict]:
    """Executa a poda pós-treino (cost-complexity) sobre a árvore vencedora.

    A varredura é focada de propósito: mantém os hiperparâmetros estruturais da
    melhor árvore encontrada e varia apenas o `ccp_alpha`, que é o critério de
    poda por custo-complexidade. É o par pós-treino da pré-poda já varrida.
    """
    configuracao_base = extrair_configuracao_arvore(tabela_busca)
    X_treino, y_treino, X_teste, y_teste = carregar_treino_teste()

    resultados = [
        avaliar_configuracao_busca(
            familia_modelo="decision_tree",
            parametros_modelo=parametros,
            X_treino=X_treino,
            y_treino=y_treino,
            X_teste=X_teste,
            y_teste=y_teste,
            iteracao=indice,
            seed=seed,
            rotulo_run="parte3_poda",
        )
        for indice, parametros in enumerate(gerar_grid_ccp_alpha(configuracao_base), start=1)
    ]

    tabela = pd.DataFrame([resultado.__dict__ for resultado in resultados])
    tabela = (
        _expandir_parametros_busca(tabela)
        .sort_values(by=["resultado_k_cv_media", "pr_auc_cv_media"], ascending=[False, False])
        .reset_index(drop=True)
    )
    melhor = tabela.iloc[0].to_dict()
    return tabela, melhor


def rodar_comparacao_inicial(seed: int = SEED) -> pd.DataFrame:
    """Roda a comparação inicial da Parte 1 e devolve a tabela consolidada."""
    X_treino, y_treino, X_teste, y_teste = carregar_treino_teste()
    resultados: list[ResultadoModelo] = []

    for nome_modelo, (papel, estimador) in construir_modelos_base(seed=seed).items():
        resultados.append(
            avaliar_modelo_holdout(
                nome_modelo=nome_modelo,
                papel=papel,
                estimador=estimador,
                X_treino=X_treino,
                y_treino=y_treino,
                X_teste=X_teste,
                y_teste=y_teste,
                seed=seed,
            )
        )

    tabela = pd.DataFrame([resultado.__dict__ for resultado in resultados])
    return tabela.sort_values(
        by=["resultado_900_holdout", "pr_auc_holdout", "roc_auc_holdout"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def rodar_protocolo_validacao(seed: int = SEED) -> pd.DataFrame:
    """Roda a Parte 2 para os 6 modelos com CV no treino e holdout no fim."""
    X_treino, y_treino, X_teste, y_teste = carregar_treino_teste()
    resultados: list[ResultadoModeloValidado] = []

    for nome_modelo, (papel, estimador) in construir_modelos_base(seed=seed).items():
        resultados.append(
            avaliar_modelo_validado(
                nome_modelo=nome_modelo,
                papel=papel,
                estimador=estimador,
                X_treino=X_treino,
                y_treino=y_treino,
                X_teste=X_teste,
                y_teste=y_teste,
                seed=seed,
            )
        )

    tabela = pd.DataFrame([resultado.__dict__ for resultado in resultados])
    return tabela.sort_values(
        by=["resultado_k_cv_media", "pr_auc_cv_media", "resultado_900_holdout"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Publicação do essencial da Parte 3 no MLflow remoto
# ---------------------------------------------------------------------------


def reconstruir_parametros_vencedor(linha: pd.Series) -> dict:
    """Recupera os hiperparâmetros de uma família vencedora a partir da tabela.

    A tabela de vencedores guarda os hiperparâmetros já expandidos em colunas
    `param_*`; aqui eles voltam ao dicionário que o estimador espera.

    Cuidado obrigatório: ao ler o CSV, o pandas transforma em `float` toda coluna
    que tenha valor ausente — `max_depth=10` volta como `10.0`, e o sklearn
    rejeita com `InvalidParameterError`. Por isso os parâmetros de contagem
    (`PARAMETROS_INTEIROS`) são convertidos de volta para `int`.
    """
    parametros: dict = {}
    for coluna, valor in linha.items():
        if not coluna.startswith("param_"):
            continue
        nome = coluna.removeprefix("param_")
        convertido = _desserializar_parametro_expandido(valor)
        if convertido is None:
            continue
        if nome in PARAMETROS_INTEIROS and isinstance(convertido, float):
            convertido = int(convertido)
        parametros[nome] = convertido
    return parametros


def publicar_essencial_parte3(
    tabela_vencedores: pd.DataFrame,
    tabela_busca: pd.DataFrame,
    tabela_poda: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    """Publica no MLflow apenas o essencial da Parte 3, sem repetir a busca inteira.

    Motivo: espelhar as 372 configurações run a run se mostrou inviável — a
    latência de rede degradou de ~21 s para ~206 s por configuração. A busca
    segue local (é determinística) e sobem só as **quatro famílias vencedoras**,
    com CV e holdout recalculados no mesmo protocolo, mais um run consolidado
    com as tabelas.

    Devolve o que foi recalculado, para conferência contra a tabela local.
    """
    X_treino, y_treino, X_teste, y_teste = carregar_treino_teste()
    conferencia: list[dict] = []

    for _, linha in tabela_vencedores.iterrows():
        familia = str(linha["familia_modelo"])
        parametros = reconstruir_parametros_vencedor(linha)
        estimador = construir_estimador_modelo(familia, parametros, seed=seed)

        tabela_folds, _ = executar_validacao_cruzada(
            estimador=clone(estimador),
            X_treino=X_treino,
            y_treino=y_treino,
            seed=seed,
            return_estimator=False,
        )
        resumo_cv = resumir_validacao_cruzada(tabela_folds)

        pipeline = construir_pipeline(modelo=clone(estimador))
        with contexto_run(
            nome=nome_run(familia, "parte3_vencedor", seed),
            parametros={
                "familia_modelo": familia,
                "papel_modelo": "vencedor da família na busca local da Parte 3",
                "iteracao_busca": int(linha["iteracao"]),
                "origem_pesos": str(linha["origem_pesos"]),
                "seed": seed,
                "n_splits_cv": N_SPLITS_CV,
                "k_fold": int(resumo_cv["k_fold"]),
                "capacidade_mensal": CAPACIDADE_MENSAL,
                "sinistros_por_mes": SINISTROS_POR_MES,
                "custo_pericia": CUSTO_PERICIA,
                "custo_fraude_paga": CUSTO_FRAUDE_PAGA,
                "features": list(FEATURES),
                **parametros,
            },
            tags={
                "fase": "4",
                "parte": "3",
                "modelo": familia,
                "papel": "vencedor_familia",
                "origem": "busca_local",
            },
        ) as registro:
            registro.log_json(tabela_folds.to_dict(orient="records"), "tabela_cv.json")
            registro.log_json(
                {"total_configuracoes_busca": int(len(tabela_busca))},
                "contexto_busca.json",
            )
            registro.log_metricas(resumo_cv, prefixo="cv_")
            pipeline.fit(X_treino, y_treino)
            probabilidades = pipeline.predict_proba(X_teste)[:, 1]
            metricas_holdout = calcular_metricas_holdout(y_teste, probabilidades)
            registro.log_metricas(metricas_holdout, prefixo="holdout_")
            registro.log_json(metricas_holdout, "metricas_holdout.json")
            registro.log_pipeline(pipeline, X_treino.head(1))

        conferencia.append(
            {
                "familia_modelo": familia,
                "resultado_k_cv_media_tabela": float(linha["resultado_k_cv_media"]),
                "resultado_k_cv_media_recalculado": float(resumo_cv["resultado_k_cv_media"]),
                "resultado_900_holdout_tabela": float(linha["resultado_900_holdout"]),
                "resultado_900_holdout_recalculado": float(metricas_holdout["resultado_900"]),
            }
        )

    with contexto_run(
        nome=nome_run("consolidado", "parte3_tabelas", seed),
        parametros={
            "seed": seed,
            "n_configuracoes_busca": int(len(tabela_busca)),
            "n_configuracoes_poda": int(len(tabela_poda)),
            "n_splits_cv": N_SPLITS_CV,
            "capacidade_mensal": CAPACIDADE_MENSAL,
        },
        tags={"fase": "4", "parte": "3", "papel": "consolidado"},
    ) as registro:
        registro.log_json(tabela_vencedores.to_dict(orient="records"), "vencedores_familia.json")
        registro.log_json(tabela_busca.to_dict(orient="records"), "busca_completa.json")
        registro.log_json(tabela_poda.to_dict(orient="records"), "poda_ccp_alpha.json")

    return pd.DataFrame(conferencia)


# ---------------------------------------------------------------------------
# Parte 4 — undersampling controlado
# ---------------------------------------------------------------------------

PASTA_TABELAS: Path = RAIZ_PROJETO / "reports" / "tabelas"
ARQUIVO_VENCEDORES_PARTE3 = "parte3_vencedores_familia.csv"
ARQUIVO_BUSCA_PARTE3 = "parte3_busca_completa.csv"
ARQUIVO_PODA_PARTE3 = "parte3_poda_ccp_alpha.csv"


@dataclass(frozen=True)
class ResultadoAmostragem:
    """Resultado de uma estratégia de amostragem para uma família vencedora."""

    familia_modelo: str
    estrategia: str
    descricao_estrategia: str
    proporcao_treino: float
    k_fold: int
    pr_auc_cv_media: float
    pr_auc_cv_desvio: float
    roc_auc_cv_media: float
    roc_auc_cv_desvio: float
    precisao_k_cv_media: float
    resultado_k_cv_media: float
    resultado_k_cv_desvio: float
    pr_auc_holdout: float
    roc_auc_holdout: float
    brier_holdout: float
    precisao_900_holdout: float
    cobertura_900_holdout: float
    resultado_900_holdout: float
    tp_900_holdout: int
    fp_900_holdout: int
    fn_900_holdout: int
    run_id: str | None


def carregar_tabela_parte3(nome_arquivo: str) -> pd.DataFrame:
    """Lê uma tabela consolidada da Parte 3 guardada em `reports/tabelas`.

    As tabelas ficam versionadas no repositório para que a Parte 4 e a Parte 5 não
    dependam de pasta temporária nem exijam refazer a busca de 34 minutos.
    """
    caminho = PASTA_TABELAS / nome_arquivo
    if not caminho.exists():
        raise FileNotFoundError(
            f"Tabela da Parte 3 ausente: {caminho}. Rode a Parte 3 antes da Parte 4."
        )
    return pd.read_csv(caminho)


def avaliar_estrategia_amostragem(
    familia_modelo: str,
    parametros_modelo: dict,
    estrategia: EstrategiaAmostragem,
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
    seed: int = SEED,
) -> ResultadoAmostragem:
    """Avalia uma estratégia de amostragem com a CV oficial e o holdout temporal.

    A reamostragem ocorre dentro do `fit` de cada fold (ver `quilha.amostragem`),
    então o holdout nunca é tocado — é o mesmo protocolo da Parte 3.
    """
    interno = construir_estimador_modelo(
        familia_modelo=familia_modelo,
        parametros_modelo=parametros_modelo,
        seed=seed,
    )
    estimador = ModeloComSubamostragem(
        estimador=interno,
        estrategia=estrategia,
        seed=seed,
    )

    tabela_folds, _ = executar_validacao_cruzada(
        estimador=estimador,
        X_treino=X_treino,
        y_treino=y_treino,
        seed=seed,
        return_estimator=False,
    )
    resumo_cv = resumir_validacao_cruzada(tabela_folds)

    pipeline = construir_pipeline(modelo=clone(estimador))
    pipeline.fit(X_treino, y_treino)
    probabilidades_holdout = pipeline.predict_proba(X_teste)[:, 1]
    metricas_holdout = calcular_metricas_holdout(y_teste, probabilidades_holdout)

    passo_modelo = pipeline.named_steps["modelo"]
    with contexto_run(
        nome=nome_run(familia_modelo, f"parte4_{estrategia.rotulo}", seed),
        parametros={
            "familia_modelo": familia_modelo,
            "estrategia_amostragem": estrategia.rotulo,
            "descricao_estrategia": estrategia.descricao(),
            "origem_pesos": descrever_peso(parametros_modelo),
            "tipo_modelo_pesos": _tipo_de_peso(parametros_modelo),
            "proporcao_treino_pos_fold": round(passo_modelo.proporcao_treino_, 4),
            "seed": seed,
            "n_splits_cv": N_SPLITS_CV,
            "k_fold": int(resumo_cv["k_fold"]),
            "capacidade_mensal": CAPACIDADE_MENSAL,
            "custo_pericia": CUSTO_PERICIA,
            "custo_fraude_paga": CUSTO_FRAUDE_PAGA,
            "aviso_prevalencia": (
                "undersampling altera a prevalencia: probabilidade crua exige recalibragem"
            ),
            "features": list(FEATURES),
            **parametros_modelo,
        },
        tags={
            "fase": "4",
            "parte": "4",
            "modelo": familia_modelo,
            "estrategia": estrategia.rotulo,
        },
    ) as registro:
        registro.log_json(tabela_folds.to_dict(orient="records"), "tabela_cv.json")
        registro.log_metricas(resumo_cv, prefixo="cv_")
        registro.log_metricas(metricas_holdout, prefixo="holdout_")
        registro.log_json(metricas_holdout, "metricas_holdout.json")

    return ResultadoAmostragem(
        familia_modelo=familia_modelo,
        estrategia=estrategia.rotulo,
        descricao_estrategia=estrategia.descricao(),
        proporcao_treino=float(passo_modelo.proporcao_treino_),
        k_fold=int(resumo_cv["k_fold"]),
        pr_auc_cv_media=float(resumo_cv["pr_auc_cv_media"]),
        pr_auc_cv_desvio=float(resumo_cv["pr_auc_cv_desvio"]),
        roc_auc_cv_media=float(resumo_cv["roc_auc_cv_media"]),
        roc_auc_cv_desvio=float(resumo_cv["roc_auc_cv_desvio"]),
        precisao_k_cv_media=float(resumo_cv["precisao_k_cv_media"]),
        resultado_k_cv_media=float(resumo_cv["resultado_k_cv_media"]),
        resultado_k_cv_desvio=float(resumo_cv["resultado_k_cv_desvio"]),
        pr_auc_holdout=float(metricas_holdout["pr_auc"]),
        roc_auc_holdout=float(metricas_holdout["roc_auc"]),
        brier_holdout=float(metricas_holdout["brier"]),
        precisao_900_holdout=float(metricas_holdout["precisao_900"]),
        cobertura_900_holdout=float(metricas_holdout["cobertura_900"]),
        resultado_900_holdout=float(metricas_holdout["resultado_900"]),
        tp_900_holdout=int(metricas_holdout["tp_900"]),
        fp_900_holdout=int(metricas_holdout["fp_900"]),
        fn_900_holdout=int(metricas_holdout["fn_900"]),
        run_id=registro.run_id,
    )


def _tipo_de_peso(parametros_modelo: dict) -> str:
    """Classifica o peso usado: nenhum, balanceado, custo 31 ou balanceamento 39,11."""
    if "scale_pos_weight" in parametros_modelo:
        return f"scale_pos_weight={parametros_modelo['scale_pos_weight']}"
    peso = parametros_modelo.get("class_weight")
    if peso is None:
        return "sem_peso"
    if peso == "balanced":
        return "class_weight=balanced"
    return "class_weight=custo31" if 31.0 in peso.values() else "class_weight=outro"


def rodar_experimento_undersampling(seed: int = SEED) -> pd.DataFrame:
    """Executa a Parte 4: as cinco estratégias sobre as quatro famílias vencedoras.

    O controle de cada família é exatamente a configuração vencedora da Parte 3 sem
    reamostragem — de propósito, para que os números do controle tenham de bater
    com os da Parte 3 e provem que o experimento não mudou o protocolo.
    """
    X_treino, y_treino, X_teste, y_teste = carregar_treino_teste()
    vencedores = carregar_tabela_parte3(ARQUIVO_VENCEDORES_PARTE3)
    estrategias = estrategias_parte4()

    resultados: list[ResultadoAmostragem] = []
    for _, linha in vencedores.iterrows():
        familia_modelo = str(linha["familia_modelo"])
        parametros_modelo = reconstruir_parametros_vencedor(linha)
        for estrategia in estrategias:
            resultados.append(
                avaliar_estrategia_amostragem(
                    familia_modelo=familia_modelo,
                    parametros_modelo=parametros_modelo,
                    estrategia=estrategia,
                    X_treino=X_treino,
                    y_treino=y_treino,
                    X_teste=X_teste,
                    y_teste=y_teste,
                    seed=seed,
                )
            )

    tabela = pd.DataFrame([resultado.__dict__ for resultado in resultados])
    return tabela.sort_values(
        by=["familia_modelo", "resultado_k_cv_media"],
        ascending=[True, False],
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Parte 5 — campeão, função de custo 31:1 e explicabilidade
# ---------------------------------------------------------------------------

SEMENTES_CAMPEAO: tuple[int, ...] = (42, 7, 2026, 123, 99)
NOME_MODELO_REGISTRADO = "quilha-fraude-campeao"
ALIAS_CAMPEAO = "champion"
ARQUIVO_AMOSTRAGEM_PARTE4 = "parte4_amostragem.csv"

# Corte teórico ótimo: períciar só quando p > 700 / 21.700. É o mesmo piso da
# Fase 1 visto do outro lado — o custo da perícia dividido pelo custo de uma
# fraude paga.
LIMIAR_CORTE_TEORICO: float = CUSTO_PERICIA / CUSTO_FRAUDE_PAGA


@dataclass(frozen=True)
class CandidatoCampeao:
    """Um candidato a campeão, com o que é preciso para reproduzi-lo."""

    origem: str
    familia_modelo: str
    estrategia: str
    resultado_900_holdout: float
    pr_auc_holdout: float
    tp_900_holdout: int
    fp_900_holdout: int
    fn_900_holdout: int
    parametros_modelo: dict


def montar_ranking_candidatos() -> pd.DataFrame:
    """Une Partes 3 e 4 num único ranking, ordenado pelo Resultado Líquido@900.

    Materializa o critério de campeão declarado no prompt: **dinheiro primeiro**.
    As quatro famílias vencedoras da Parte 3 entram como candidatas `controle`; as
    estratégias de amostragem da Parte 4 entram com o resultado que já mediram. As
    linhas de `controle` da Parte 4 repetem, por construção, a Parte 3 — ficam de
    fora para não duplicar candidato.
    """
    vencedores = carregar_tabela_parte3(ARQUIVO_VENCEDORES_PARTE3)
    amostragem = carregar_tabela_parte3(ARQUIVO_AMOSTRAGEM_PARTE4)
    parametros_por_familia = {
        str(linha["familia_modelo"]): reconstruir_parametros_vencedor(linha)
        for _, linha in vencedores.iterrows()
    }

    candidatos: list[CandidatoCampeao] = []

    def _registrar(origem: str, linha: pd.Series, estrategia: str) -> None:
        familia = str(linha["familia_modelo"])
        candidatos.append(
            CandidatoCampeao(
                origem=origem,
                familia_modelo=familia,
                estrategia=estrategia,
                resultado_900_holdout=float(linha["resultado_900_holdout"]),
                pr_auc_holdout=float(linha["pr_auc_holdout"]),
                tp_900_holdout=int(linha["tp_900_holdout"]),
                fp_900_holdout=int(linha["fp_900_holdout"]),
                fn_900_holdout=int(linha["fn_900_holdout"]),
                parametros_modelo=parametros_por_familia[familia],
            )
        )

    for _, linha in vencedores.iterrows():
        _registrar("parte3", linha, "controle")
    for _, linha in amostragem.iterrows():
        estrategia = str(linha["estrategia"])
        if estrategia == "controle":
            continue
        _registrar("parte4", linha, estrategia)

    tabela = pd.DataFrame([candidato.__dict__ for candidato in candidatos])
    return tabela.sort_values(
        by=["resultado_900_holdout", "pr_auc_holdout"],
        ascending=[False, False],
    ).reset_index(drop=True)


def estimador_do_candidato(candidato: pd.Series, seed: int = SEED) -> BaseEstimator:
    """Monta o estimador de um candidato, com reamostragem quando a estratégia pede."""
    interno = construir_estimador_modelo(
        familia_modelo=str(candidato["familia_modelo"]),
        parametros_modelo=dict(candidato["parametros_modelo"]),
        seed=seed,
    )
    rotulo = str(candidato["estrategia"])
    if rotulo == "controle":
        return interno
    for estrategia in estrategias_parte4():
        if estrategia.rotulo == rotulo:
            return ModeloComSubamostragem(estimador=interno, estrategia=estrategia, seed=seed)
    raise ValueError(f"Estratégia de amostragem desconhecida: {rotulo}")


def ajustar_candidato(
    candidato: pd.Series,
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    seed: int = SEED,
) -> Pipeline:
    """Ajusta o Pipeline do candidato no treino — o único `fit` que vê o treino."""
    pipeline = construir_pipeline(modelo=estimador_do_candidato(candidato, seed=seed))
    pipeline.fit(X_treino, y_treino)
    return pipeline


def avaliar_campeao_sementes(
    candidato: pd.Series,
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
    sementes: tuple[int, ...] = SEMENTES_CAMPEAO,
) -> pd.DataFrame:
    """Roda o campeão no holdout com as cinco sementes e registra cada run.

    Uma semente só conta uma história; cinco mostram a dispersão. A primeira
    semente também registra o Pipeline no Model Registry com o alias `champion` —
    é o ponteiro que o app consome depois.
    """
    familia = str(candidato["familia_modelo"])
    linhas: list[dict] = []

    for seed in sementes:
        estimador = estimador_do_candidato(candidato, seed=seed)
        pipeline = construir_pipeline(modelo=estimador)
        with contexto_run(
            nome=nome_run(familia, "parte5_campeao", seed),
            parametros={
                "familia_modelo": familia,
                "estrategia_amostragem": str(candidato["estrategia"]),
                "origem_candidato": str(candidato["origem"]),
                "papel_modelo": "campeao (maior Resultado Liquido@900 no holdout)",
                "seed": seed,
                "capacidade_mensal": CAPACIDADE_MENSAL,
                "custo_pericia": CUSTO_PERICIA,
                "custo_fraude_paga": CUSTO_FRAUDE_PAGA,
                "limiar_corte_teorico": LIMIAR_CORTE_TEORICO,
                "features": list(FEATURES),
                **dict(candidato["parametros_modelo"]),
            },
            tags={"fase": "4", "parte": "5", "modelo": familia, "papel": "campeao"},
        ) as registro:
            pipeline.fit(X_treino, y_treino)
            probabilidades = pipeline.predict_proba(X_teste)[:, 1]
            metricas = calcular_metricas_holdout(y_teste, probabilidades)
            registro.log_metricas(metricas, prefixo="holdout_")
            registro.log_json(metricas, "metricas_holdout.json")
            uri_modelo = registro.log_pipeline(pipeline, X_treino.head(1))
            if seed == sementes[0]:
                registro.registrar_modelo(
                    NOME_MODELO_REGISTRADO,
                    descricao=(
                        "Campeao da Fase 4: maior Resultado Liquido@900 no holdout, "
                        "validado em 5 sementes."
                    ),
                    tags={"fase": "4", "parte": "5", "familia_modelo": familia},
                    alias=ALIAS_CAMPEAO,
                )

        linhas.append({"seed": seed, **metricas, "run_id": registro.run_id, "uri": uri_modelo})

    return pd.DataFrame(linhas)


def resumir_campeao(tabela: pd.DataFrame) -> dict[str, float | int | list[int]]:
    """Média ± desvio das cinco sementes — o veredito de robustez do campeão."""
    return {
        "sementes": [int(s) for s in tabela["seed"]],
        "resultado_900_media": float(tabela["resultado_900"].mean()),
        "resultado_900_desvio": float(tabela["resultado_900"].std(ddof=1)),
        "pr_auc_media": float(tabela["pr_auc"].mean()),
        "pr_auc_desvio": float(tabela["pr_auc"].std(ddof=1)),
        "roc_auc_media": float(tabela["roc_auc"].mean()),
        "roc_auc_desvio": float(tabela["roc_auc"].std(ddof=1)),
        "brier_media": float(tabela["brier"].mean()),
        "tp_900_media": float(tabela["tp_900"].mean()),
    }


def decompor_custo_31_1(
    y_true: pd.Series | np.ndarray,
    probabilidades: np.ndarray,
    k: int = CAPACIDADE_MENSAL,
    limiar: float = LIMIAR_CORTE_TEORICO,
) -> dict:
    """Decompõe a função de custo 31:1 nos dois cortes: teórico e prático.

    A conta, escrita por extenso para o comitê:

        Custo        = 21.700 x FN + 700 x FP
        Custo evitado = 21.700 x TP
        Resultado líquido = Custo evitado - 700 x (nº de perícias)

    O **corte teórico** perícia todo sinistro com p >= 3,2258% (o ponto em que a
    perícia se paga). O **corte prático** perícia os `k` sinistros mais arriscados
    — é o que a capacidade de 900/mês permite.
    """
    y = np.asarray(y_true, dtype=int)
    proba = _probabilidade_positiva(probabilidades)
    positivos = int(y.sum())

    def _corte(selecao: np.ndarray) -> dict[str, float | int]:
        tp = int(y[selecao].sum())
        n_periciados = int(selecao.sum())
        fp = n_periciados - tp
        fn = positivos - tp
        return {
            "n_periciados": n_periciados,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "custo_falso_negativo": float(CUSTO_FRAUDE_PAGA * fn),
            "custo_falso_positivo": float(CUSTO_PERICIA * fp),
            "custo_total": float(CUSTO_FRAUDE_PAGA * fn + CUSTO_PERICIA * fp),
            "custo_evitado": float(CUSTO_FRAUDE_PAGA * tp),
            "resultado_liquido": float(CUSTO_FRAUDE_PAGA * tp - CUSTO_PERICIA * n_periciados),
            "precisao": float(tp / n_periciados) if n_periciados else 0.0,
            "cobertura": float(tp / positivos) if positivos else 0.0,
        }

    ordem = np.argsort(-proba, kind="mergesort")
    selecao_pratica = np.zeros(len(y), dtype=bool)
    selecao_pratica[ordem[: min(k, len(y))]] = True

    return {
        "limiar_teorico": float(limiar),
        "n_fraudes_holdout": positivos,
        "custo_total_sem_modelo": float(CUSTO_FRAUDE_PAGA * positivos),
        "corte_teorico": _corte(proba >= limiar),
        "corte_pratico": _corte(selecao_pratica),
    }


def rodar_analise_custo_familias(seed: int = SEED) -> pd.DataFrame:
    """Responde à pergunta obrigatória: o custo 31:1 de cada família vencedora.

    Uma única passada por família: ajusta no treino, mede o holdout e decompõe o
    custo nos dois cortes. É a tabela que sustenta (ou derruba) a afirmação de que
    o XGBoost é o modelo que mais reduz a função de custo.
    """
    X_treino, y_treino, X_teste, y_teste = carregar_treino_teste()
    vencedores = carregar_tabela_parte3(ARQUIVO_VENCEDORES_PARTE3)
    linhas: list[dict] = []

    for _, linha in vencedores.iterrows():
        familia = str(linha["familia_modelo"])
        parametros = reconstruir_parametros_vencedor(linha)
        pipeline = construir_pipeline(
            modelo=construir_estimador_modelo(familia, parametros, seed=seed)
        )
        with contexto_run(
            nome=nome_run(familia, "parte5_custo31", seed),
            parametros={
                "familia_modelo": familia,
                "papel_modelo": "analise da funcao de custo 31:1",
                "seed": seed,
                "capacidade_mensal": CAPACIDADE_MENSAL,
                "custo_pericia": CUSTO_PERICIA,
                "custo_fraude_paga": CUSTO_FRAUDE_PAGA,
                "limiar_corte_teorico": LIMIAR_CORTE_TEORICO,
                **parametros,
            },
            tags={"fase": "4", "parte": "5", "modelo": familia, "analise": "custo_31_1"},
        ) as registro:
            pipeline.fit(X_treino, y_treino)
            probabilidades = pipeline.predict_proba(X_teste)[:, 1]
            decomposicao = decompor_custo_31_1(y_teste, probabilidades)
            registro.log_json(decomposicao, "custo_31_1.json")
            for corte, prefixo in (("corte_teorico", "teorico_"), ("corte_pratico", "pratico_")):
                registro.log_metricas(decomposicao[corte], prefixo=prefixo)

        teorico = decomposicao["corte_teorico"]
        pratico = decomposicao["corte_pratico"]
        linhas.append(
            {
                "familia_modelo": familia,
                "limiar_teorico": decomposicao["limiar_teorico"],
                "n_fraudes_holdout": decomposicao["n_fraudes_holdout"],
                "custo_total_sem_modelo": decomposicao["custo_total_sem_modelo"],
                "teorico_n_periciados": teorico["n_periciados"],
                "teorico_tp": teorico["tp"],
                "teorico_fp": teorico["fp"],
                "teorico_fn": teorico["fn"],
                "teorico_custo_total": teorico["custo_total"],
                "teorico_custo_evitado": teorico["custo_evitado"],
                "teorico_resultado_liquido": teorico["resultado_liquido"],
                "pratico_n_periciados": pratico["n_periciados"],
                "pratico_tp": pratico["tp"],
                "pratico_fp": pratico["fp"],
                "pratico_fn": pratico["fn"],
                "pratico_custo_total": pratico["custo_total"],
                "pratico_custo_evitado": pratico["custo_evitado"],
                "pratico_resultado_liquido": pratico["resultado_liquido"],
                "run_id": registro.run_id,
            }
        )

    return pd.DataFrame(linhas).sort_values(
        by="pratico_resultado_liquido", ascending=False
    ).reset_index(drop=True)


def curva_resultado_k(
    y_true: pd.Series | np.ndarray,
    probabilidades: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Curva Resultado(k) para k de 0 até o tamanho do holdout.

    Devolve a tabela `(k, tp, precisao, resultado_liquido, custo_total,
    probabilidade_do_k-esimo)` e o resumo com o k que maximiza o resultado, o k do
    corte teórico (onde a probabilidade cruza os 3,2258%) e o k de 900.
    """
    y = np.asarray(y_true, dtype=int)
    proba = _probabilidade_positiva(probabilidades)
    ordem = np.argsort(-proba, kind="mergesort")
    y_ordenado = y[ordem]
    proba_ordenada = proba[ordem]
    n = len(y)

    k = np.arange(0, n + 1)
    tp = np.concatenate([[0], np.cumsum(y_ordenado)])
    resultado = CUSTO_FRAUDE_PAGA * tp - CUSTO_PERICIA * k
    custo = CUSTO_FRAUDE_PAGA * (int(y.sum()) - tp) + CUSTO_PERICIA * (k - tp)
    precisao = np.divide(tp, k, out=np.zeros(n + 1, dtype=float), where=k > 0)

    tabela = pd.DataFrame(
        {
            "k": k,
            "tp": tp,
            "precisao": precisao,
            "resultado_liquido": resultado,
            "custo_total": custo,
            "probabilidade_do_k_esimo": np.concatenate([[np.nan], proba_ordenada]),
        }
    )

    indice_otimo = int(np.argmax(resultado))
    k_teorico = int((proba >= LIMIAR_CORTE_TEORICO).sum())
    ainda_lucrativo = np.nonzero(resultado > 0)[0]
    resumo: dict[str, float | int] = {
        "k_otimo_empirico": int(k[indice_otimo]),
        "resultado_maximo": float(resultado[indice_otimo]),
        "k_corte_teorico": k_teorico,
        "resultado_no_corte_teorico": float(resultado[k_teorico]),
        "resultado_em_900": float(resultado[min(CAPACIDADE_MENSAL, n)]),
        "k_maximo_ainda_lucrativo": int(ainda_lucrativo.max()) if len(ainda_lucrativo) else 0,
    }
    return tabela, resumo


def salvar_curva_resultado_k(
    tabela: pd.DataFrame,
    resumo: dict[str, float | int],
    caminho: Path | str | None = None,
) -> Path:
    """Gera a figura da curva Resultado(k), com o corte teórico e o de 900 marcados."""
    if caminho is None:
        caminho = PASTA_FIGURAS / "parte5_curva_resultado_k.png"
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    figura, eixo = plt.subplots(figsize=(11, 6))
    eixo.plot(
        tabela["k"],
        tabela["resultado_liquido"] / 1000.0,
        color="#1f4e79",
        lw=2,
        label="Resultado líquido (R$ mil)",
    )
    eixo.axvline(
        int(resumo["k_otimo_empirico"]),
        color="#2e7d32",
        ls="--",
        lw=1.5,
        label=f"k* empírico = {int(resumo['k_otimo_empirico'])}",
    )
    eixo.axvline(
        CAPACIDADE_MENSAL,
        color="#c62828",
        ls=":",
        lw=1.5,
        label=f"corte prático = {CAPACIDADE_MENSAL}",
    )
    eixo.axvline(
        int(resumo["k_corte_teorico"]),
        color="#f9a825",
        ls="-.",
        lw=1.5,
        label=f"corte teórico p* = {int(resumo['k_corte_teorico'])}",
    )
    eixo.axhline(0.0, color="0.4", lw=1)
    eixo.set_xlabel("k — sinistros periciados, por ordem decrescente de risco")
    eixo.set_ylabel("Resultado líquido (R$ mil)")
    eixo.set_title("Parte 5 — curva Resultado(k) do campeão no holdout")
    eixo.grid(alpha=0.3)
    eixo.legend(loc="lower left")

    # Eixo secundário: probabilidade do k-ésimo sinistro, com a linha de 3,2258%.
    eixo_prob = eixo.twinx()
    mascara = tabela["k"] > 0
    eixo_prob.plot(
        tabela.loc[mascara, "k"],
        tabela.loc[mascara, "probabilidade_do_k_esimo"] * 100.0,
        color="0.6",
        lw=1,
        alpha=0.8,
    )
    eixo_prob.axhline(
        LIMIAR_CORTE_TEORICO * 100.0,
        color="#8e24aa",
        ls="--",
        lw=1.2,
        label="precisão marginal 3,2258%",
    )
    eixo_prob.set_ylabel("Probabilidade do k-ésimo (%)", color="0.4")
    eixo_prob.set_yscale("log")
    eixo_prob.legend(loc="upper right")

    figura.tight_layout()
    figura.savefig(caminho, dpi=150)
    plt.close(figura)
    return caminho


def avaliar_calibracao(
    candidato: pd.Series,
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compara a escala de probabilidade do campeão com isotônica e Platt.

    O campeão não usa peso de classe, então a probabilidade já sai na escala
    natural (prevalência de ~2,47%) — a calibração **não** é obrigatória. Ainda
    assim medimos: o app entrega a probabilidade ao analista, e é preciso saber se
    ela significa o que diz. Cada calibrador é ajustado **dentro do treino**
    (`cross_val_predict` do próprio `CalibratedClassifierCV`), nunca no holdout.

    Devolve `(tabela_comparativa, curva_de_confiabilidade)`.
    """
    validacao = construir_validacao_cruzada(seed=seed)

    def _brier(probabilidades: np.ndarray) -> float:
        return float(brier_score_loss(y_teste, probabilidades))

    linhas: list[dict] = []
    pipeline = ajustar_candidato(candidato, X_treino, y_treino, seed=seed)
    proba_padrao = pipeline.predict_proba(X_teste)[:, 1]
    linhas.append(
        {
            "modelo": "campeao (sem calibracao)",
            "brier": _brier(proba_padrao),
            "n_acima_do_limiar": int((proba_padrao >= LIMIAR_CORTE_TEORICO).sum()),
        }
    )

    for metodo in ("isotonic", "sigmoid"):
        calibrado = CalibratedClassifierCV(
            estimator=construir_pipeline(modelo=estimador_do_candidato(candidato, seed=seed)),
            method=metodo,
            cv=validacao,
        )
        calibrado.fit(X_treino, y_treino)
        proba_calibrada = calibrado.predict_proba(X_teste)[:, 1]
        linhas.append(
            {
                "modelo": f"campeao + {metodo}",
                "brier": _brier(proba_calibrada),
                "n_acima_do_limiar": int((proba_calibrada >= LIMIAR_CORTE_TEORICO).sum()),
            }
        )

    fracao_positivos, probabilidade_media = calibration_curve(
        y_teste, proba_padrao, n_bins=10, strategy="quantile"
    )
    curva = pd.DataFrame(
        {"probabilidade_media": probabilidade_media, "fracao_positivos": fracao_positivos}
    )
    return pd.DataFrame(linhas), curva


def salvar_curva_calibracao(curva: pd.DataFrame, caminho: Path | str | None = None) -> Path:
    """Diagrama de confiabilidade: probabilidade prevista contra a observada."""
    caminho = Path(caminho) if caminho is not None else PASTA_FIGURAS / "parte5_calibracao.png"
    caminho.parent.mkdir(parents=True, exist_ok=True)

    figura, eixo = plt.subplots(figsize=(7, 7))
    eixo.plot([0, 1], [0, 1], ls="--", color="0.5", label="calibração perfeita")
    eixo.plot(
        curva["probabilidade_media"],
        curva["fracao_positivos"],
        marker="o",
        color="#1f4e79",
        label="campeão",
    )
    eixo.set_xlabel("Probabilidade prevista (média do bin)")
    eixo.set_ylabel("Fração de fraudes observada")
    eixo.set_title("Parte 5 — curva de calibração do campeão no holdout")
    eixo.grid(alpha=0.3)
    eixo.legend(loc="upper left")
    figura.tight_layout()
    figura.savefig(caminho, dpi=150)
    plt.close(figura)
    return caminho


def avaliar_ablacao(
    familia_modelo: str,
    parametros_modelo: dict,
    colunas_removidas: tuple[str, ...],
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
    seed: int = SEED,
) -> dict[str, float]:
    """ROC AUC e PR AUC no holdout com as colunas indicadas fora do modelo.

    É a ablação exigida pela Pergunta 1: medir o quanto `uf` e `idade_segurado`
    acrescentam de fato, em vez de supor.
    """
    estimador = construir_estimador_modelo(familia_modelo, parametros_modelo, seed=seed)
    pipeline = construir_pipeline(modelo=estimador, colunas_excluidas=tuple(colunas_removidas))
    pipeline.fit(X_treino, y_treino)
    probabilidades = pipeline.predict_proba(X_teste)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_teste, probabilidades)),
        "pr_auc": float(average_precision_score(y_teste, probabilidades)),
    }


def rodar_ablacao_campeao(
    candidato: pd.Series,
    X_treino: pd.DataFrame,
    y_treino: pd.Series,
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
    colunas: tuple[str, ...] = ("uf", "idade_segurado"),
    seed: int = SEED,
) -> pd.DataFrame:
    """Roda a ablação do campeão com e sem as colunas pedidas."""
    familia = str(candidato["familia_modelo"])
    parametros = dict(candidato["parametros_modelo"])

    completo = avaliar_ablacao(familia, parametros, (), X_treino, y_treino, X_teste, y_teste, seed)
    sem_colunas = avaliar_ablacao(
        familia, parametros, colunas, X_treino, y_treino, X_teste, y_teste, seed
    )
    linhas = [
        {"cenario": "completo", **completo},
        {"cenario": f"sem {' + '.join(colunas)}", **sem_colunas},
        {
            "cenario": "variacao",
            "roc_auc": sem_colunas["roc_auc"] - completo["roc_auc"],
            "pr_auc": sem_colunas["pr_auc"] - completo["pr_auc"],
        },
    ]
    return pd.DataFrame(linhas)


def importancia_shap(
    pipeline: Pipeline,
    fundo: pd.DataFrame,
    amostra: pd.DataFrame,
    n_fundo: int = 100,
    semente: int = SEED,
) -> pd.Series:
    """Importância média absoluta do SHAP, agregada por feature original."""
    contribuicoes = contribuicoes_agregadas(
        pipeline=pipeline,
        fundo=fundo,
        amostra=amostra,
        n_fundo=n_fundo,
        semente=semente,
    )
    return contribuicoes.abs().mean(axis=0).sort_values(ascending=False)


def salvar_shap_summary(importancia: pd.Series, caminho: Path | str | None = None) -> Path:
    """Barras horizontais da importância SHAP média absoluta por feature original."""
    caminho = caminho if caminho is not None else PASTA_FIGURAS / "parte5_shap_summary.png"
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    ordenada = importancia.sort_values()
    figura, eixo = plt.subplots(figsize=(9, 0.42 * len(ordenada) + 2))
    eixo.barh(ordenada.index, ordenada.to_numpy(), color="#1f4e79")
    eixo.set_xlabel("Contribuição SHAP média absoluta (variação na probabilidade de fraude)")
    eixo.set_title("Parte 5 — SHAP summary do campeão, agregado por feature original")
    eixo.grid(axis="x", alpha=0.3)
    figura.tight_layout()
    figura.savefig(caminho, dpi=150)
    plt.close(figura)
    return caminho


def salvar_shap_local(
    contribuicoes: pd.Series,
    titulo: str,
    caminho: Path | str,
    maximo_features: int = 10,
) -> Path:
    """Barras da contribuição SHAP de UM sinistro, com o texto de negócio no título."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    recorte = contribuicoes.head(maximo_features).iloc[::-1]
    cores = ["#c62828" if valor > 0 else "#2e7d32" for valor in recorte.to_numpy()]

    figura, eixo = plt.subplots(figsize=(9, 0.5 * len(recorte) + 2))
    eixo.barh(recorte.index, recorte.to_numpy(), color=cores)
    eixo.axvline(0.0, color="0.4", lw=1)
    eixo.set_xlabel(
        "Contribuição SHAP (variação na probabilidade de fraude) — vermelho empurra para fraude"
    )
    eixo.set_title(titulo)
    eixo.grid(axis="x", alpha=0.3)
    figura.tight_layout()
    figura.savefig(caminho, dpi=150)
    plt.close(figura)
    return caminho


# ---------------------------------------------------------------------------
# Parte 5 — SHAP local dos três casos narrados de negócio (item 7)
# ---------------------------------------------------------------------------

CASOS_SHAP_LOCAIS: tuple[str, ...] = ("fraude_pega", "fraude_perdida", "legitimo_topo")

TITULOS_CASOS_SHAP: dict[str, str] = {
    "fraude_pega": "Fraude pega — periciado no corte de 900 e confirmado",
    "fraude_perdida": "Fraude perdida — fraude real fora das 900 perícias",
    "legitimo_topo": "Falso positivo — sinistro legítimo no topo da fila",
}


def selecionar_casos_shap(
    y_teste: pd.Series | np.ndarray,
    probabilidades: np.ndarray,
    k: int = CAPACIDADE_MENSAL,
) -> dict[str, int]:
    """Localiza no holdout os três sinistros que a narrativa de negócio pede.

    * `fraude_pega`    — fraude real (y=1) dentro do corte de 900, a de maior risco;
    * `fraude_perdida` — fraude real (y=1) fora do corte de 900, a de maior risco;
    * `legitimo_topo`  — sinistro legítimo (y=0) dentro do corte de 900, o de maior risco.

    Devolve o índice posicional no holdout. Se um caso não existir, devolve -1 — falha
    visível, nunca silenciosa.
    """
    y = np.asarray(y_teste, dtype=int)
    proba = _probabilidade_positiva(probabilidades)
    ordem = np.argsort(-proba, kind="mergesort")
    selecao = np.zeros(len(y), dtype=bool)
    selecao[ordem[: min(k, len(y))]] = True

    def _de_maior_risco(mascara: np.ndarray) -> int:
        # `ordem` já vem do maior risco para o menor: o primeiro que casa é o campeão do grupo.
        candidatos = ordem[mascara[ordem]]
        return int(candidatos[0]) if candidatos.size else -1

    return {
        "fraude_pega": _de_maior_risco(selecao & (y == 1)),
        "fraude_perdida": _de_maior_risco((~selecao) & (y == 1)),
        "legitimo_topo": _de_maior_risco(selecao & (y == 0)),
    }


def _narrativa_caso(caso: str, probabilidade: float, contribuicoes: pd.Series, n: int = 3) -> str:
    """Frase de negócio montada a partir das contribuições SHAP — nada digitado à mão."""
    aberturas = {
        "fraude_pega": (
            "Periciado no corte de 900 e confirmado como fraude "
            f"(probabilidade {probabilidade:.4f})"
        ),
        "fraude_perdida": (
            f"Fraude real que ficou fora das 900 perícias (probabilidade {probabilidade:.4f})"
        ),
        "legitimo_topo": (
            "Sinistro legítimo que entrou na fila e consumiu perícia "
            f"(probabilidade {probabilidade:.4f})"
        ),
    }
    principais = contribuicoes.head(n)
    trecho = "; ".join(f"{nome} {valor:+.4f}" for nome, valor in principais.items())
    return f"{aberturas[caso]}. Sinais que mais empurraram a decisão: {trecho}."


def explicar_casos_locais(
    pipeline: Pipeline,
    X_treino: pd.DataFrame,
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
    probabilidades: np.ndarray,
    pasta: Path | str | None = None,
    n_fundo: int = 100,
    semente: int = SEED,
) -> tuple[pd.DataFrame, list[Path]]:
    """SHAP local de 3 sinistros reais, com a narrativa do porquê (item 7 da Parte 5).

    Executa o `explicar_sinistro` — o caminho que o app percorre ao explicar um chamado
    novo — para um fraude pega, um fraude perdida e um legítimo no topo da fila. Devolve
    `(tabela_dos_casos, lista_de_figuras)`.
    """
    pasta_figuras = Path(pasta) if pasta is not None else PASTA_FIGURAS
    pasta_figuras.mkdir(parents=True, exist_ok=True)

    proba = _probabilidade_positiva(probabilidades)
    y_array = np.asarray(y_teste, dtype=int)
    indices = selecionar_casos_shap(y_array, proba)

    linhas: list[dict] = []
    figuras: list[Path] = []
    for caso in CASOS_SHAP_LOCAIS:
        indice = indices[caso]
        if indice < 0:
            raise RuntimeError(f"Holdout não contém um caso '{caso}' para explicar.")
        contribuicoes = explicar_sinistro(
            pipeline=pipeline,
            linha=X_teste.iloc[[indice]],
            fundo=X_treino,
            n_fundo=n_fundo,
            semente=semente,
        )
        caminho = pasta_figuras / f"parte5_shap_local_{caso}.png"
        salvar_shap_local(contribuicoes, TITULOS_CASOS_SHAP[caso], caminho)
        figuras.append(caminho)
        linhas.append(
            {
                "caso": caso,
                "indice_holdout": indice,
                "probabilidade": float(proba[indice]),
                "fraude_real": int(y_array[indice]),
                "top_features": ", ".join(contribuicoes.head(3).index),
                "narrativa": _narrativa_caso(caso, float(proba[indice]), contribuicoes),
            }
        )
    return pd.DataFrame(linhas), figuras

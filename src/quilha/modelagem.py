"""Modelagem da Fase 4: baselines, validação e métricas de negócio.

Esta primeira camada da Fase 4 resolve a Parte 1 do prompt:

* define os 6 modelos da comparação inicial;
* implementa a Regra Atual da Quilha como estimador compatível com scikit-learn;
* roda todos pelo mesmo protocolo sobre o holdout temporal;
* aplica a validação cruzada oficial da fase, sem vazamento;
* registra métricas e o Pipeline inteiro no MLflow.

As etapas de CV, busca de hiperparâmetros, under-sampling e campeão entram nas
partes seguintes da fase e serão construídas por cima deste módulo.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, make_scorer, roc_auc_score
from sklearn.model_selection import (
    ParameterGrid,
    ParameterSampler,
    StratifiedKFold,
    cross_validate,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from quilha.dados import PASTA_FIGURAS, carregar_base_crua, dividir_temporal
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

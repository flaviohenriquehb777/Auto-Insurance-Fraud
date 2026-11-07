# %% [markdown]
# # Fase 4 — Modelagem
#
# **Projeto:** Detecção de fraude em sinistros de automóvel — Quilha Seguros
# **Semente:** `42` · **Holdout:** agosto/2026 · **Capacidade:** 900 perícias/mês
#
# Este notebook integra as Partes 0 a 6. As buscas custosas das Partes 3 a 5 são
# consumidas das tabelas versionadas, em vez de serem repetidas a cada execução.
# A Regra Atual tem sua CV revalidada no treino porque o run original não persistiu
# as métricas dos folds. Ao final, o campeão é reajustado no treino completo,
# avaliado uma vez no holdout, salvo para o aplicativo e verificado após recarga.
#
# MLflow usa DagsHub quando há credenciais e SQLite local quando não há; nenhuma
# célula exige token para concluir.

# %%
# =============================================================================
# Raiz do projeto, imports e reprodutibilidade
# =============================================================================
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from IPython.display import Image, Markdown, display

SEED = 42
warnings.filterwarnings("ignore", message="IProgress not found")
warnings.filterwarnings("ignore", message="Saving scikit-learn models in the pickle or cloudpickle format")
warnings.filterwarnings("ignore", message="Failed to resolve installed pip version")
warnings.filterwarnings("ignore", message="Credenciais do MLflow ausentes")


def localizar_raiz(inicio: Path) -> Path:
    """Sobe diretórios até localizar o pyproject.toml da raiz."""
    for pasta in [inicio.resolve(), *inicio.resolve().parents]:
        if (pasta / "pyproject.toml").exists():
            return pasta
    raise FileNotFoundError("Raiz do projeto não encontrada (pyproject.toml ausente).")


RAIZ = localizar_raiz(Path.cwd())
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from quilha.dados import DATA_CORTE, carregar_base_crua
from quilha.mlops import contexto_run, credenciais_presentes, nome_run
from quilha.modelagem import (
    ALIAS_CAMPEAO,
    CAPACIDADE_MENSAL,
    LIMIAR_CORTE_TEORICO,
    NOME_MODELO_REGISTRADO,
    SEED as SEED_MODELAGEM,
    RegraAtualQuilha,
    ajustar_candidato,
    calcular_metricas_holdout,
    carregar_treino_teste,
    executar_validacao_cruzada,
    montar_ranking_candidatos,
    resumir_validacao_cruzada,
    separar_entrada_alvo,
)
from quilha.pipeline import CAMINHO_PIPELINE, carregar_pipeline, salvar_pipeline
from quilha.pipeline import construir_pipeline

assert SEED == SEED_MODELAGEM
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)
print("Raiz do projeto localizada pelo pyproject.toml.")
print(f"SEED={SEED} | k holdout={CAPACIDADE_MENSAL} | corte teórico={LIMIAR_CORTE_TEORICO:.4%}")
print(f"Credenciais MLOps presentes: {credenciais_presentes()}")
if not credenciais_presentes():
    print("Sem credenciais remotas: as runs usam o fallback local em mlflow.db.")

# %% [markdown]
# ## 1. Revalidar a Regra Atual pela CV oficial
#
# Os outros cinco baselines já têm seus resultados de cinco folds registrados no
# MLflow e congelados em `reports/tabelas/parte2_cv_base_mlflow.csv`. A Regra Atual
# não tinha métricas CV persistidas; recalculamos somente essa linha usando os
# mesmos cinco folds estratificados do treino e `k=720`. O holdout não participa.

# %%
base_crua_treino = carregar_base_crua()
mascara_treino = pd.to_datetime(base_crua_treino["data_abertura"]) < DATA_CORTE
base_treino_cv = base_crua_treino.loc[mascara_treino].copy()
X_treino_cv, y_treino_cv = separar_entrada_alvo(base_treino_cv)

with contexto_run(
    nome=nome_run("regra_atual_quilha", "parte2_cv_revalidacao", SEED),
    parametros={
        "familia_modelo": "regra_atual_quilha",
        "seed": SEED,
        "n_splits_cv": 5,
        "k_fold": 720,
        "origem": "revalidacao_integracao_parte6",
    },
    tags={"fase": "4", "parte": "2", "modelo": "regra_atual_quilha"},
) as registro_cv_regra:
    tabela_cv_regra, _ = executar_validacao_cruzada(
        estimador=RegraAtualQuilha(),
        X_treino=X_treino_cv,
        y_treino=y_treino_cv,
        seed=SEED,
        return_estimator=False,
    )
    resumo_cv_regra = resumir_validacao_cruzada(tabela_cv_regra)
    registro_cv_regra.log_metricas(resumo_cv_regra, prefixo="cv_")
    registro_cv_regra.log_json(
        tabela_cv_regra.to_dict(orient="records"),
        "tabela_cv_regra_atual.json",
    )
    pipeline_regra = construir_pipeline(modelo=clone(RegraAtualQuilha()))
    pipeline_regra.fit(X_treino_cv, y_treino_cv)
    registro_cv_regra.log_pipeline(pipeline_regra, X_treino_cv.head(1))

id_run_regra = registro_cv_regra.run_id or "MLflow indisponível"
linha_cv_regra = {
    "familia_modelo": "regra_atual_quilha",
    **resumo_cv_regra,
    "run_id_mlflow": id_run_regra,
}
tabela_cv_base = pd.read_csv(RAIZ / "reports" / "tabelas" / "parte2_cv_base_mlflow.csv")
tabela_cv_modelos = pd.concat(
    [tabela_cv_base, pd.DataFrame([linha_cv_regra])],
    ignore_index=True,
).sort_values("resultado_k_cv_media", ascending=False)
caminho_cv = RAIZ / "reports" / "tabelas" / "parte2_cv_seis_modelos.csv"
tabela_cv_modelos.to_csv(caminho_cv, index=False)

print(f"Treino usado na CV: {len(X_treino_cv):,} linhas; fraudes={int(y_treino_cv.sum()):,}.")
print(f"Run da revalidação da Regra Atual: {id_run_regra}")
display(tabela_cv_modelos)

# %% [markdown]
# ## 2. Resultados congelados das Partes 3 a 5
#
# A seleção prioriza Resultado Líquido@900, não acurácia. As tabelas são artefatos
# das execuções já concluídas; nenhuma busca de hiperparâmetros é repetida aqui.

# %%
pasta_tabelas = RAIZ / "reports" / "tabelas"
vencedores_parte3 = pd.read_csv(pasta_tabelas / "parte3_vencedores_familia.csv")
amostragem_parte4 = pd.read_csv(pasta_tabelas / "parte4_amostragem.csv")
ranking_candidatos = montar_ranking_candidatos()
custo_familias = pd.read_csv(pasta_tabelas / "parte5_custo31_familias.csv")
sementes_campeao = pd.read_csv(pasta_tabelas / "parte5_sementes_campeao.csv")
ablacao = pd.read_csv(pasta_tabelas / "parte5_ablacao.csv")
calibracao = pd.read_csv(pasta_tabelas / "parte5_calibracao.csv")
casos_shap = pd.read_csv(pasta_tabelas / "parte5_shap_casos.csv")
importancia_shap = pd.read_csv(pasta_tabelas / "parte5_shap_importancia.csv")

print("Vencedores da busca e poda da Parte 3:")
display(vencedores_parte3)
print("Comparação das estratégias de amostragem da Parte 4:")
display(
    amostragem_parte4[
        [
            "familia_modelo",
            "estrategia",
            "pr_auc_holdout",
            "resultado_900_holdout",
            "tp_900_holdout",
            "fp_900_holdout",
            "fn_900_holdout",
        ]
    ].sort_values("resultado_900_holdout", ascending=False)
)
print("Top candidatos por Resultado Líquido@900:")
display(ranking_candidatos.head(10))

# %% [markdown]
# ## 3. Custo 31:1, calibração e explicabilidade
#
# Um falso negativo custa R$ 21.700 e uma perícia custa R$ 700. Logo,
# `21.700 / 700 = 31`: uma perícia precisa evitar, em média, mais de 1/31 de
# fraude paga para compensar seu custo. O corte probabilístico de viabilidade é
# `p* = 700 / 21.700 = 3,2258%`; a capacidade operacional, porém, limita a fila
# prática a 900 perícias mensais.

# %%
print("Decomposição da função de custo por família e por corte:")
display(custo_familias)
print("Robustez do campeão nas cinco sementes:")
display(sementes_campeao)
print("Calibração:")
display(calibracao)
print("Ablação:")
display(ablacao)
print("Casos SHAP locais:")
display(casos_shap)
print("Importância SHAP agregada por feature original:")
display(importancia_shap.head(10))

figuras = [
    "04_01_decision_tree_profundidade.png",
    "04_02_random_forest_profundidade.png",
    "parte5_curva_resultado_k.png",
    "parte5_calibracao.png",
    "parte5_shap_summary.png",
    "parte5_shap_local_fraude_pega.png",
    "parte5_shap_local_fraude_perdida.png",
    "parte5_shap_local_legitimo_topo.png",
]
for nome_figura in figuras:
    caminho_figura = RAIZ / "reports" / "figures" / nome_figura
    if caminho_figura.exists():
        display(Markdown(f"**{nome_figura}**"))
        display(Image(filename=str(caminho_figura)))

# %% [markdown]
# ## 4. Reajuste final, holdout e artefato do aplicativo
#
# Só depois de fechar CV, busca, amostragem e seleção do campeão acessamos o
# holdout temporal. O campeão continua sendo escolhido pelo ranking versionado;
# o Pipeline completo é então ajustado nos quatro meses de treino e testado uma
# única vez em agosto.

# %%
campeao = ranking_candidatos.iloc[0]
assert campeao["familia_modelo"] == "xgboost"
assert campeao["estrategia"] == "controle"

del base_crua_treino, base_treino_cv, X_treino_cv, y_treino_cv
X_treino, y_treino, X_teste, y_teste = carregar_treino_teste()

with contexto_run(
    nome=nome_run("xgboost", "parte6_integracao", SEED),
    parametros={
        "familia_modelo": str(campeao["familia_modelo"]),
        "estrategia_amostragem": str(campeao["estrategia"]),
        "seed": SEED,
        "capacidade_mensal": CAPACIDADE_MENSAL,
        "limiar_corte_teorico": LIMIAR_CORTE_TEORICO,
        **dict(campeao["parametros_modelo"]),
    },
    tags={"fase": "4", "parte": "6", "modelo": "xgboost", "papel": "integracao_campeao"},
) as registro_campeao:
    pipeline_campeao = ajustar_candidato(campeao, X_treino, y_treino, seed=SEED)
    probabilidades_holdout = pipeline_campeao.predict_proba(X_teste)[:, 1]
    metricas_holdout = calcular_metricas_holdout(y_teste, probabilidades_holdout)
    registro_campeao.log_metricas(metricas_holdout, prefixo="holdout_")
    registro_campeao.log_json(metricas_holdout, "metricas_holdout.json")
    uri_modelo = registro_campeao.log_pipeline(pipeline_campeao, X_treino.head(1))
    versao_registro = registro_campeao.registrar_modelo(
        NOME_MODELO_REGISTRADO,
        descricao=(
            "Campeao da Fase 4: XGBoost controle, selecionado por Resultado Liquido@900."
        ),
        tags={"fase": "4", "parte": "6", "familia_modelo": "xgboost"},
        alias=ALIAS_CAMPEAO,
    )

caminho_salvo = salvar_pipeline(pipeline_campeao)
pipeline_recarregado = carregar_pipeline(caminho_salvo)
probabilidades_recarregadas = pipeline_recarregado.predict_proba(X_teste)[:, 1]
np.testing.assert_allclose(
    probabilidades_recarregadas,
    probabilidades_holdout,
    rtol=0,
    atol=1e-12,
)

print(f"Campeão: {campeao['familia_modelo']} / {campeao['estrategia']}")
print(f"Run de integração: {registro_campeao.run_id}")
print(f"URI resolvível do modelo logado: {uri_modelo}")
print(f"Registro: {NOME_MODELO_REGISTRADO} | alias={ALIAS_CAMPEAO} | versão={versao_registro}")
print(f"Artefato regravado: {caminho_salvo.relative_to(RAIZ)}")
print(f"Tamanho do artefato: {caminho_salvo.stat().st_size:,} bytes")
print(f"Paridade antes/depois da recarga: {np.max(np.abs(probabilidades_holdout - probabilidades_recarregadas)):.2e}")
display(pd.DataFrame([metricas_holdout]))

# %% [markdown]
# ## 5. Critérios de integração
#
# O teste automatizado cobre o scorer financeiro, a Regra Atual, o corte teórico,
# `k` proporcional e a paridade de predição do artefato salvo. A checagem final
# abaixo confirma que o modelo carregado pelo aplicativo é o XGBoost campeão e
# que todos os três componentes do Pipeline estão presentes.

# %%
assert list(pipeline_recarregado.named_steps) == ["preparo", "imputacao", "modelo"]
assert type(pipeline_recarregado.named_steps["modelo"]).__name__ == "XGBClassifier"
assert len(tabela_cv_modelos) == 6
assert tabela_cv_modelos["k_fold"].nunique() == 1
assert int(tabela_cv_modelos["k_fold"].iloc[0]) == 720
assert int(metricas_holdout["tp_900"]) == 201
assert int(metricas_holdout["fp_900"]) == 699
assert int(metricas_holdout["fn_900"]) == 80
assert float(metricas_holdout["resultado_900"]) == 3_731_700.0
print("ACEITE: seis modelos com CV comparável; Pipeline campeão salvo e validado.")

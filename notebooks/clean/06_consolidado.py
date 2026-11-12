# %% [markdown]
# # Notebook consolidado — Detecção de fraude em sinistros de automóvel (CRISP-DM, Fases 1 a 6)
#
# **Projeto:** Quilha Seguros · **Semente:** `42` · **Base:** 60.000 sinistros (abr–ago/2026)
#
# Este é o **arquivo único** que conta o projeto inteiro — da pergunta de negócio à política
# de perícia em uso — e prova, com o número ao lado, cada uma das **5 respostas do comitê**.
#
# Ele é escrito para rodar **sozinho**: sem credencial, sem infraestrutura e sem caminho fixo.
# O tracking de MLOps é envolvido em `contexto_run`, que degrada para o SQLite local
# (`sqlite:///mlflow.db`) quando não há token — e nenhuma célula quebra por causa disso.
#
# **Seções:** 0 (setup) · 1 (negócio) · 2 (dados e colunas — P1) · 3 (descritiva — P2) ·
# 4 (modelagem — P3) · 5 (avaliação — P4/P5) · 6 (implantação) · 7 (Quadro de Respostas).
#
# Toda a lógica vem dos módulos de `src/quilha` — este notebook **usa**, não reimplementa.

# %%
# =============================================================================
# 0 · SETUP — raiz tolerante ao caminho, semente, versões e cronômetro
# =============================================================================
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Cronômetro iniciado ANTES de qualquer import pesado: é o tempo real de execução.
INICIO = time.perf_counter()

# Semente única de toda a aleatoriedade do projeto (reexecução determinística).
SEED = 42


def localizar_raiz(inicio: Path) -> Path:
    """Sobe diretórios a partir de `inicio` até localizar o `pyproject.toml` (a raiz).

    Tolerante ao diretório de execução: funciona rodando da raiz, de `notebooks/` ou de
    qualquer subpasta. Se a raiz não for encontrada, a mensagem diz o que fazer — o
    notebook nunca quebra por causa do `cwd`.
    """
    for pasta in [inicio.resolve(), *inicio.resolve().parents]:
        if (pasta / "pyproject.toml").exists():
            return pasta
    raise FileNotFoundError(
        "Raiz do projeto não encontrada (nenhum pyproject.toml acima). "
        "Coloque o notebook dentro da árvore do repositório ou execute-o a partir da raiz."
    )


RAIZ = localizar_raiz(Path.cwd())
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

# --- funções do projeto (reaproveitadas, nunca duplicadas) --------------------
from quilha.avaliacao import (
    CAMPOS_QUADRO_RESPOSTAS,
    N_REAMOSTRAGENS_BOOTSTRAP,
    PERGUNTA_DE_DECISAO,
    PERICIAS_POR_PERITO,
    PERITOS_ATUAIS,
    aplicar_politica,
    bootstrap_resultado,
    construir_politica_campeao,
    construir_referencias,
    custo_sem_pericia,
    escore_positivo,
    executar_backtest,
    grade_sensibilidade,
    k_marginal_cruza_piso,
    medir_politicas_no_mes,
    montar_quadro_respostas,
    piso_viabilidade,
    pontos_de_decisao_k,
    regra_atual_do_case,
    resultado_economico,
    resultado_liquido,
    resumo_parametros_decisao,
    sensibilidade_taxa_fraude,
    taxa_fraude_de_break_even,
    tp_acumulado_por_k,
    triagem_disparidade,
)
from quilha.app import (
    MENSAGEM_PERMANENTE,
    PayloadRecusado,
    avaliar_fila,
    drift_das_features,
    faixa_prioridade,
    hash_do_artefato,
    k_do_lote,
    medir_valor_da_politica,
    montar_fila,
    painel_valor,
    pontuar_lote,
    resumo_fila,
    saude_do_modelo,
)
from quilha.contrato import CAMPOS_ABERTURA, validar_payload
from quilha.dados import (
    CAMINHO_BASE_CRUA,
    RAIZ_PROJETO,
    carregar_base_crua,
    dividir_temporal,
)
from quilha.mlops import contexto_run, credenciais_presentes, nome_run
from quilha.modelagem import (
    CAPACIDADE_MENSAL,
    CUSTO_FRAUDE_PAGA,
    CUSTO_PERICIA,
    LIMIAR_CORTE_TEORICO,
    LIMIAR_REGRA_ATUAL,
    PASTA_TABELAS,
    SEED as SEED_MODELAGEM,
    SINISTROS_POR_MES,
    calcular_metricas_holdout,
    curva_resultado_k,
    importancia_shap,
    separar_entrada_alvo,
)
from quilha.monitoramento import alertas_imputacao, resumo_imputacoes
from quilha.pipeline import CAMINHO_PIPELINE, carregar_pipeline, construir_pipeline
from quilha.preparacao import (
    BOLETIM_NAO_INFORMADO,
    FEATURES,
    ROTULOS_APOLICE,
    ROTULOS_VALOR,
    preparar,
)
from sklearn.base import clone
from xgboost import XGBClassifier

assert SEED == SEED_MODELAGEM, "A semente do notebook e a do projeto precisam bater."

# --- versões do ambiente (rastreabilidade) ------------------------------------
import sklearn  # noqa: E402
import xgboost  # noqa: E402

try:
    import mlflow

    versao_mlflow = mlflow.__version__
except ImportError:  # pragma: no cover - só ocorre sem o pacote instalado
    versao_mlflow = "não instalado"

print("=" * 78)
print("0 · SETUP")
print("=" * 78)
print(f"  raiz do projeto : {RAIZ}")
print(f"  base crua       : {CAMINHO_BASE_CRUA.relative_to(RAIZ)}")
print(f"  artefato        : {CAMINHO_PIPELINE.relative_to(RAIZ)}")
print(f"  semente (SEED)  : {SEED}")
print(
    "  versões         : "
    f"python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__} | "
    f"sklearn {sklearn.__version__} | xgboost {xgboost.__version__} | mlflow {versao_mlflow}"
)
print(f"  credenciais MLOps: {credenciais_presentes()} "
      f"-> {'DagsHub' if credenciais_presentes() else 'SQLite local (mlflow.db)'}")

pd.set_option("display.float_format", lambda valor: f"{valor:,.6f}")

# %% [markdown]
# ## 1 · Negócio — a decisão, o critério econômico e o piso de viabilidade
#
# Os parâmetros do Anexo 1 do case governam tudo: uma perícia custa **R$ 700** e evita
# **R$ 21.700** de fraude paga. Daí sai o **piso de viabilidade** `p* = 700/21.700 = 3,2258%`
# — abaixo dele, periciar ao acaso **perde** dinheiro. Como a taxa de fraude da base (~2,47%)
# está abaixo do piso, a política só cria valor por **concentração**: empurrar fraudes para o
# topo da fila com precisão acima de `p*`.

# %%
# =============================================================================
# 1 · NEGÓCIO — parâmetros do case, piso e a pergunta que encerra tudo
# =============================================================================
parametros_caso = resumo_parametros_decisao()
piso = piso_viabilidade()

print("=" * 78)
print("1 · PARÂMETROS DA DECISÃO (Anexo 1 do case)")
print("=" * 78)
print(f"  sinistros por mês            : {parametros_caso['sinistros_por_mes']:,}")
print(f"  capacidade de perícia        : {parametros_caso['capacidade_mensal']} perícias/mês "
      f"({PERITOS_ATUAIS} peritos × {PERICIAS_POR_PERITO} perícias)")
print(f"  custo de uma perícia         : R$ {CUSTO_PERICIA:,.2f}")
print(f"  custo da fraude paga         : R$ {CUSTO_FRAUDE_PAGA:,.2f}")
print(f"  regra atual                  : valor reclamado > R$ {LIMIAR_REGRA_ATUAL:,.0f} -> corte por valor")
print()
print(f"  piso de viabilidade p*       : {piso:.6f}  ({piso * 100:.4f}%)")
print(f"  razão de custo               : {parametros_caso['razao_custo_fraude_pericia']:.2f} : 1 "
      f"(1 fraude perdida = 31 perícias)")
print()
print("  Critério econômico: Resultado(k) = 21.700 × TP(k) − 700 × k,  sujeito a k ≤ 900/mês")
print(f"  Conferência da matemática: resultado_liquido(tp=1, k=31) = R$ {resultado_liquido(1, 31):,.2f} (deve dar 0)")
print()
print("-" * 78)
print("PERGUNTA QUE O PROJETO ENCERRA:")
print("-" * 78)
print(PERGUNTA_DE_DECISAO)

# %% [markdown]
# ## 2 · Dados e colunas — leitura crua e a decisão de features (Pergunta 1)
#
# A base é lida **crua** (`dtype=str`), como entregue. O tratamento inteiro mora na função
# pura `preparar`, que nada aprende do treino. Duas colunas **saem** de propósito:
# `valor_indenizado` (é preenchida no encerramento — vazamento) e `terceiro_identificado`
# (redundante: o sinal útil já vira `tem_terceiro_identificado`). As chaves de controle
# (`id_sinistro`, datas, `hora_ocorrencia`) nunca entram — o tempo entra por atributo derivado.

# %%
# =============================================================================
# 2 · DADOS E COLUNAS — leitura crua, tipagem e decisão de features (P1)
# =============================================================================
base_crua = carregar_base_crua()
fraude_numerica = pd.to_numeric(base_crua["fraude"], errors="raise").astype(int)
X_completo, y_completo = separar_entrada_alvo(base_crua)

print("=" * 78)
print("2 · LEITURA CRUA DA BASE (P1)")
print("=" * 78)
print(f"  formato          : {base_crua.shape[0]:,} linhas × {base_crua.shape[1]} colunas")
print(f"  colunas cruas    : {', '.join(base_crua.columns)}")
print(f"  features do modelo: {len(FEATURES)} (ordem canônica de preparacao.FEATURES)")

colunas_fora = sorted(set(base_crua.columns) - set(FEATURES) - {"fraude"})
print()
print(f"  COLUNAS FORA DO MODELO ({len(colunas_fora)}):")
for coluna in colunas_fora:
    motivo = {
        "valor_indenizado": "vazamento (preenchida no encerramento)",
        "terceiro_identificado": "redundante (vira tem_terceiro_identificado)",
        "id_sinistro": "chave de controle",
        "data_abertura": "data crua (o tempo entra por atributo derivado)",
        "data_ocorrencia": "data crua (o tempo entra por atributo derivado)",
        "hora_ocorrencia": "texto cru (vira hora + periodo_dia)",
    }.get(coluna, "coluna de controle")
    print(f"    - {coluna:<22} {motivo}")
print()
print(f"  AS {len(FEATURES)} FEATURES DO MODELO:")
for posicao, nome in enumerate(FEATURES, start=1):
    print(f"    {posicao:>2}. {nome}")

# %%
# =============================================================================
# 2 · P1 — Idade e UF: usar ou não usar? (ablação com o mesmo protocolo)
# =============================================================================
# Reajusta o campeão SEM `uf` e `idade_segurado`, com os MESMOS hiperparâmetros e a mesma
# divisão temporal (treino abr–jul, teste ago, SEED=42), e mede o preço em reais da remoção.
treino_cru, teste_cru = dividir_temporal(base_crua)
X_treino, y_treino = separar_entrada_alvo(treino_cru)
X_teste, y_teste = separar_entrada_alvo(teste_cru)

HIERPARAMETROS_CAMPEAO = {
    "max_depth": 4,
    "n_estimators": 350,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.7,
    "gamma": 3,
    "min_child_weight": 5,
    "reg_lambda": 1,
    "scale_pos_weight": 1,
}

COLUNAS_ABLADAS = ("uf", "idade_segurado")
pipeline_sem_idade_uf = construir_pipeline(
    modelo=XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=1, **HIERPARAMETROS_CAMPEAO),
    colunas_excluidas=COLUNAS_ABLADAS,
)
pipeline_sem_idade_uf.fit(X_treino, y_treino)
prob_sem_idade_uf = pipeline_sem_idade_uf.predict_proba(X_teste)[:, 1]
metricas_sem_idade_uf = calcular_metricas_holdout(y_teste, prob_sem_idade_uf)

# O campeão auditado roda no MESMO holdout, para a comparação ser maçã-com-maçã.
pipeline_auditado = carregar_pipeline(CAMINHO_PIPELINE)
metricas_com_idade_uf = calcular_metricas_holdout(
    y_teste, escore_positivo(pipeline_auditado, X_teste)
)

print("P1 — ABLAÇÃO DE `uf` + `idade_segurado` (holdout de agosto, k=900)")
print(f"  com idade e UF  : ROC AUC {metricas_com_idade_uf['roc_auc']:.6f} · "
      f"PR AUC {metricas_com_idade_uf['pr_auc']:.6f}")
print(f"  sem idade e UF  : ROC AUC {metricas_sem_idade_uf['roc_auc']:.6f} · "
      f"PR AUC {metricas_sem_idade_uf['pr_auc']:.6f}")
print(f"  Resultado@900 com as duas: R$ {float(metricas_com_idade_uf['resultado_900']):,.2f} · "
      f"sem as duas: R$ {float(metricas_sem_idade_uf['resultado_900']):,.2f}")

# %% [markdown]
# ## 3 · Análise descritiva — o que o negócio pergunta (Pergunta 2)
#
# Tudo aqui é **calculado de fato** a partir de `carregar_base_crua` + `preparar` — nenhum
# número digitado à mão. O achado central: a fraude **se concentra abaixo** do limiar de
# R$ 50 mil (a faixa de R$ 40–50 mil tem ~25% de fraude), e é por isso que a regra de valor
# perde dinheiro e a fila é ordenada por **escore**.

# %%
# =============================================================================
# 3 · DESCRITIVA (P2) — taxas por grupo, calculadas da base crua
# =============================================================================
base_analise = preparar(base_crua).assign(fraude=y_completo.to_numpy())


def numero_pt(valor: int) -> str:
    """Milhar com ponto: `27970` -> `27.970`."""
    return f"{valor:,}".replace(",", ".")


def pct_pt(taxa: float) -> str:
    """Percentual com vírgula e duas casas: `0.0373` -> `3,73%`."""
    return f"{taxa * 100:.2f}%".replace(".", ",")


def brl(valor: float) -> str:
    """Reais no padrão brasileiro: `3731700.0` -> `R$ 3.731.700,00`."""
    texto = f"{abs(valor):,.2f}".replace(",", "@").replace(".", ",").replace("@", ".")
    return f"{'-' if valor < 0 else ''}R$ {texto}"


def resumo_categoria(coluna: str, ordem: list[str] | None = None) -> str:
    """Contagem e taxa de fraude por categoria, no padrão 'Categoria: n (taxa)'."""
    agrupado = base_analise.groupby(coluna, observed=True)["fraude"].agg(n="size", taxa="mean")
    sequencia = ordem if ordem is not None else list(agrupado.sort_values("n", ascending=False).index)
    return " · ".join(
        f"{rotulo}: {numero_pt(int(agrupado.loc[rotulo, 'n']))} sinistros "
        f"({pct_pt(float(agrupado.loc[rotulo, 'taxa']))} de fraude)"
        for rotulo in sequencia
        if rotulo in agrupado.index
    )


taxa_base = float(y_completo.mean())
n_fraudes_base = int(y_completo.sum())

print("=" * 78)
print("3 · ANÁLISE DESCRITIVA DA BASE (P2)")
print("=" * 78)
print(f"  taxa de fraude da base : {pct_pt(taxa_base)} — {numero_pt(n_fraudes_base)} fraudes "
      f"em {numero_pt(len(y_completo))} sinistros")
print()
print(f"  por tipo de sinistro   : {resumo_categoria('tipo_sinistro')}")
print()
print(f"  por canal de abertura  : {resumo_categoria('canal_abertura')}")
print()
print(f"  por boletim de ocorrência: {resumo_categoria('boletim_ocorrencia')}")
print()
print(f"  por tempo de apólice   : {resumo_categoria('faixa_apolice', list(ROTULOS_APOLICE))}")
print()
print(f"  por valor reclamado    : {resumo_categoria('faixa_valor_reclamado', list(ROTULOS_VALOR))}")

# %%
# =============================================================================
# 3 · P2 — O boletim em branco: caso de negócio, nunca imputação
# =============================================================================
boletim_branco = base_analise.loc[base_analise["boletim_ocorrencia"] == BOLETIM_NAO_INFORMADO, "fraude"]
n_boletim_branco = len(boletim_branco)
share_boletim_branco = n_boletim_branco / len(base_analise)
taxa_boletim_branco = float(boletim_branco.mean())

print("P2 — TRATAMENTO DO BOLETIM EM BRANCO")
print(f"  registros em branco : {numero_pt(n_boletim_branco)} ({pct_pt(share_boletim_branco)} da base)")
print(f"  taxa de fraude      : {pct_pt(taxa_boletim_branco)} "
      f"(média da base {pct_pt(taxa_base)}) -> ACIMA da média")
print("  DECISÃO: o branco vira a categoria própria 'Não informado' — nunca imputação (nem pela moda).")

# %% [markdown]
# ## 4 · Modelagem — comparação, escolha do campeão e métricas (Pergunta 3)
#
# Os números de treino vêm da **validação cruzada de 5 folds** congelada nas tabelas
# versionadas; os de teste, do **holdout temporal** de agosto. Os modelos custosos não são
# re-treinados aqui — as tabelas são lidas, com a procedência declarada.

# %%
# =============================================================================
# 4 · MODELAGEM (P3) — seis modelos, campeão e métricas (tabelas versionadas)
# =============================================================================
tabela_seis_modelos = pd.read_csv(PASTA_TABELAS / "parte2_cv_seis_modelos.csv")
tabela_vencedores = pd.read_csv(PASTA_TABELAS / "parte3_vencedores_familia.csv")

linha_dummy = tabela_seis_modelos.loc[tabela_seis_modelos["familia_modelo"] == "dummy_prior"].iloc[0]
linha_campeao_cv = tabela_vencedores.loc[tabela_vencedores["familia_modelo"] == "xgboost"].iloc[0]

ROTULOS_MODELOS = {
    "dummy_prior": "Dummy prior",
    "regra_atual_quilha": "Regra atual Quilha",
    "decision_tree": "Árvore de decisão",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

print("=" * 78)
print("4 · MODELAGEM (P3) — procedência: reports/tabelas/parte2_cv_seis_modelos.csv")
print("=" * 78)
print("  SEIS MODELOS COMPARADOS (validação cruzada de 5 folds, k=720, no TREINO):")
ordem = tabela_seis_modelos.sort_values("resultado_k_cv_media", ascending=False)
for _, linha in ordem.iterrows():
    nome = ROTULOS_MODELOS.get(linha["familia_modelo"], linha["familia_modelo"])
    print(f"    {nome:<22} PR AUC {linha['pr_auc_cv_media']:.6f} · "
          f"ROC AUC {linha['roc_auc_cv_media']:.6f} · Resultado@720 R$ {linha['resultado_k_cv_media']:,.2f}")
print()
print(f"  MODELO RECOMENDADO     : XGBoost (família 'xgboost')")
print(f"  referência simples     : ROC AUC {linha_dummy['roc_auc_cv_media']:.6f} · "
      f"PR AUC {linha_dummy['pr_auc_cv_media']:.6f} (DummyClassifier prior)")
print()
print("  HIPERPARÂMETROS DO CAMPEÃO (tabela parte3_vencedores_familia.csv):")
for nome, valor in HIERPARAMETROS_CAMPEAO.items():
    print(f"    {nome:<18} = {valor}")

# %%
# =============================================================================
# 4 · P3 — Períodos do treino/teste e métricas de treino (CV) x teste (holdout)
# =============================================================================
abertura_treino = pd.to_datetime(X_treino["data_abertura"])
abertura_teste = pd.to_datetime(X_teste["data_abertura"])

pipeline = carregar_pipeline(CAMINHO_PIPELINE)
escore_campeao_teste = escore_positivo(pipeline, X_teste)
metricas_teste = calcular_metricas_holdout(y_teste, escore_campeao_teste)

print("P3 — PERÍODOS E MÉTRICAS DO CAMPEÃO")
print(f"  treino : {abertura_treino.min():%d/%m/%Y} a {abertura_treino.max():%d/%m/%Y} "
      f"({numero_pt(len(y_treino))} sinistros, {pct_pt(float(y_treino.mean()))} de fraude)")
print(f"  teste  : {abertura_teste.min():%m/%Y} "
      f"({numero_pt(len(y_teste))} sinistros, {pct_pt(float(y_teste.mean()))} de fraude)")
print()
print(f"  TREINO (média da CV de 5 folds): ROC AUC {linha_campeao_cv['roc_auc_cv_media']:.6f} · "
      f"PR AUC {linha_campeao_cv['pr_auc_cv_media']:.6f}")
print(f"  TESTE  (holdout de agosto)     : ROC AUC {metricas_teste['roc_auc']:.6f} · "
      f"PR AUC {metricas_teste['pr_auc']:.6f}")
print(f"  Resultado@900 no teste         : TP={metricas_teste['tp_900']} · "
      f"FP={metricas_teste['fp_900']} · FN={metricas_teste['fn_900']} · "
      f"{brl(float(metricas_teste['resultado_900']))}")

# %% [markdown]
# ## 5 · Avaliação — política de perícia, economia e dimensionamento (P4 e P5)
#
# A decisão é `ordenar pelo escore → cortar em k → periciar`. Aqui o escore vira dinheiro no
# mês de teste, sempre contra as três referências (nada, regra de R$ 50 mil e acaso), e a
# recomendação é submetida ao bootstrap e à grade de sensibilidade antes de ser afirmada.

# %%
# =============================================================================
# 5 · AVALIAÇÃO (P4) — as políticas no mês de teste, sob o mesmo critério
# =============================================================================
referencias = construir_referencias(k=CAPACIDADE_MENSAL)
referencias["regra_atual"].estimador.fit(X_treino, y_treino)
referencias["aleatoria"].estimador.fit(X_treino, y_treino)

politicas_mes = {"campeao": construir_politica_campeao(pipeline, k=CAPACIDADE_MENSAL), **referencias}
tabela_politicas = medir_politicas_no_mes(politicas_mes, X_teste, y_teste)

print("=" * 78)
print("5 · POLÍTICAS NO MÊS DE TESTE (agosto/2026) — mesma capacidade, mesmo critério")
print("=" * 78)
for _, linha in tabela_politicas.iterrows():
    print(f"  {linha['politica']:<28} k={linha['n_periciados']:>3}  TP={linha['tp']:>3}  "
          f"precisão={linha['precisao'] * 100:>6.2f}%  economia={brl(linha['economia'])}")

economia_campeao = float(tabela_politicas.loc[tabela_politicas["chave"] == "campeao", "economia"].iloc[0])
economia_regra = float(tabela_politicas.loc[tabela_politicas["chave"] == "regra_atual", "economia"].iloc[0])
tp_campeao_900 = int(tabela_politicas.loc[tabela_politicas["chave"] == "campeao", "tp"].iloc[0])
tp_regra_900 = int(tabela_politicas.loc[tabela_politicas["chave"] == "regra_atual", "tp"].iloc[0])
ganho_sobre_regra = economia_campeao - economia_regra

print()
print(f"  Ganho da política sobre a regra atual: {brl(ganho_sobre_regra)}/mês")

# %%
# =============================================================================
# 5 · P4 — A regra atual exatamente como o case a descreve
# =============================================================================
regra_case = regra_atual_do_case(X_teste, y_teste, limiar=LIMIAR_REGRA_ATUAL, k=CAPACIDADE_MENSAL)

print("P4 — REGRA ATUAL COMO O CASE A DESCREVE (valor_reclamado > R$ 50 mil -> corte em 900)")
print(f"  pedidos acima de R$ {regra_case['limiar']:,.0f}: {regra_case['n_acima_limiar']} "
      f"(o case fala em ~950 — divergência de arredondamento do case)")
print(f"  periciados (corte em {CAPACIDADE_MENSAL}): {regra_case['n_periciados']}")
print(f"  fraudes encontradas (TP): {regra_case['tp']} · precisão {regra_case['precisao'] * 100:.2f}% · "
      f"economia {brl(regra_case['economia'])}")

# %%
# =============================================================================
# 5 · P4/P5 — Curva Resultado(k) e os pontos de decisão (dimensionamento)
# =============================================================================
tabela_curva, resumo_curva = curva_resultado_k(y_teste, escore_campeao_teste)
pontos_decisao = pontos_de_decisao_k(tabela_curva, resumo_curva)

k_otimo = int(resumo_curva["k_otimo_empirico"])
k_teorico = int(resumo_curva["k_corte_teorico"])
k_lucrativo_max = int(resumo_curva["k_maximo_ainda_lucrativo"])
k_cruza_piso = k_marginal_cruza_piso(tabela_curva)
resultado_900 = float(tabela_curva.loc[tabela_curva["k"] == CAPACIDADE_MENSAL, "resultado_liquido"].iloc[0])
resultado_otimo = float(tabela_curva.loc[tabela_curva["k"] == k_otimo, "resultado_liquido"].iloc[0])
ganho_alem_de_900 = resultado_otimo - resultado_900
tp_1800 = int(tabela_curva.loc[tabela_curva["k"] == 2 * CAPACIDADE_MENSAL, "tp"].iloc[0])
multiplo_real = tp_1800 / tp_campeao_900

print("P4/P5 — PONTOS DE DECISÃO SOBRE A CURVA Resultado(k)")
for k, valor, rotulo in pontos_decisao:
    print(f"  {rotulo:<22} k={k:>5}  ->  {brl(valor)}")
print()
print(f"  ótimo empírico k={numero_pt(k_otimo)} (~{round(k_otimo / PERICIAS_POR_PERITO)} peritos) · "
      f"ganho além das 900 = {brl(ganho_alem_de_900)}/mês")
print(f"  última perícia que ainda se paga k={k_teorico} · primeira que não se paga k={k_cruza_piso} "
      f"({'vizinhas — confere' if k_cruza_piso == k_teorico + 1 else 'DIVERGE'})")
print(f"  fim do lucro: k={numero_pt(k_lucrativo_max)}")
print()
print(f"  tese do Otávio (dobrar perícias dobraria a fraude): {tp_campeao_900} -> {tp_1800} fraudes; "
      f"múltiplo real {multiplo_real:.2f}× (a promessa de 2× exigiria {2 * tp_campeao_900})")

# %%
# =============================================================================
# 5 · P5 — Sensibilidade (um fator por vez), bootstrap e break-even
# =============================================================================
tp_campeao_acumulado = tp_acumulado_por_k(y_teste, escore_campeao_teste)
tp_regra_acumulado = tp_acumulado_por_k(y_teste, escore_positivo(referencias["regra_atual"].estimador, X_teste))
grade = grade_sensibilidade(tp_campeao_acumulado, tp_regra_acumulado)

print("P5 — GRADE DE SENSIBILIDADE ECONÔMICA")
print(f"  células: {len(grade)} · ganho de {brl(grade['ganho'].min())} a {brl(grade['ganho'].max())}")
print(f"  células em que o campeão NÃO vence a regra: "
      f"{int((~grade['vence_regra']).sum())} de {len(grade)}")
print()
print("  Um fator por vez — capacidade (fraude R$ 21.700, perícia R$ 700):")
visao_capacidade = grade[(grade["custo_fraude_paga"] == CUSTO_FRAUDE_PAGA) & (grade["custo_pericia"] == CUSTO_PERICIA)]
for _, linha in visao_capacidade.iterrows():
    print(f"    capacidade {int(linha['capacidade']):>5} · TP campeão {int(linha['tp_campeao']):>3} · "
          f"ganho {brl(linha['ganho'])} · vence a regra: {bool(linha['vence_regra'])}")
print()
taxa_equilibrio = taxa_fraude_de_break_even(y_teste, escore_campeao_teste, k=CAPACIDADE_MENSAL)
print(f"  taxa de fraude de equilíbrio (abaixo dela a política para de se pagar): "
      f"{taxa_equilibrio * 100:.4f}%")
print(f"  piso de viabilidade absoluto: {piso * 100:.4f}%")

# %%
# =============================================================================
# 5 · P5 — Bootstrap pareado do mês de teste (IC 95% da diferença)
# =============================================================================
tabela_bootstrap, resumo_bootstrap = bootstrap_resultado(
    y_teste,
    {"campeao": escore_campeao_teste, "regra_atual": escore_positivo(referencias["regra_atual"].estimador, X_teste)},
    k=CAPACIDADE_MENSAL,
)
linha_dif = tabela_bootstrap.set_index("metrica").loc["diferenca"]

print(f"P5 — BOOTSTRAP DO MÊS DE TESTE ({resumo_bootstrap['n_reamostragens']:,} reamostragens, SEED={SEED})")
for _, linha in tabela_bootstrap.iterrows():
    print(f"  {linha['metrica']:<12} média {brl(linha['media'])} · "
          f"IC 95% [{brl(linha['ic_inferior'])}; {brl(linha['ic_superior'])}]")
print()
print(f"  diferença pareada (campeão − regra): {brl(resumo_bootstrap['diferenca_media'])} "
      f"[{brl(resumo_bootstrap['diferenca_ic_inferior'])}; {brl(resumo_bootstrap['diferenca_ic_superior'])}]")
print(f"  o IC 95% exclui o zero? {'SIM — o ganho é distinguível do ruído' if resumo_bootstrap['exclui_zero'] else 'NÃO'}")

# %%
# =============================================================================
# 5 · P5 — Backtest walk-forward (estimativa fora da amostra, 4 janelas)
# =============================================================================
base_datas = pd.to_datetime(base_crua["data_abertura"])
modelo_campeao = pipeline.named_steps["modelo"]
tabela_backtest = executar_backtest(X_completo, y_completo, base_datas, modelo_campeao)
pivot_backtest = tabela_backtest.pivot(index="janela", columns="politica", values="resultado_k")
diferenca_janelas = pivot_backtest["campeao"] - pivot_backtest["regra_atual"]

print("P5 — BACKTEST WALK-FORWARD (treino expansivo, avalia no mês seguinte)")
for janela, linha in pivot_backtest.iterrows():
    print(f"  janela {janela}  campeão {brl(linha['campeao']):>16} | regra atual {brl(linha['regra_atual']):>16}")
print()
print(f"  diferença média campeão − regra: {brl(diferenca_janelas.mean())} ± {brl(diferenca_janelas.std())}")
print(f"  pior janela da diferença       : {brl(diferenca_janelas.min())}")

# %%
# =============================================================================
# 5 · P4 — Disparidade da fila periciada (sinal para revisão, não veredito)
# =============================================================================
tabela_disparidade = triagem_disparidade(y_teste, escore_campeao_teste, X_teste, k=CAPACIDADE_MENSAL)
disparidade_idade = tabela_disparidade[tabela_disparidade["dimensao"] == "idade"]

print("P4 — TRIAGEM DE DISPARIDADE DA FILA (por faixa de idade, k=900)")
print(disparidade_idade[["grupo", "n_grupo", "n_selecionado", "razao_selecao", "precisao"]].to_string(index=False))
print("  NOTA: a fila é PLANA por idade (razão de seleção próxima de 1×); os extremos por UF")
print("        são ruído de amostra pequena — achado registrado, não veredito.")

# %% [markdown]
# ## 6 · Implantação (Fase 6) — contrato, app e monitoramento
#
# O app **não decide fraude**: prioriza a fila, explica e mostra o valor. Antes de qualquer
# predição, o **contrato de dados** recusa payload inválido com erro explícito — a "falha
# silenciosa da vírgula" é o pior modo de falha possível. Nada é re-treinado aqui.

# %%
# =============================================================================
# 6 · IMPLANTAÇÃO — identidade do artefato e contrato na fronteira (camada 2)
# =============================================================================
hash_artefato = hash_do_artefato(CAMINHO_PIPELINE)
CAMINHO_VERSOES = CAMINHO_PIPELINE.with_name("pipeline_fraude.versoes.json")
versoes = json.loads(CAMINHO_VERSOES.read_text(encoding="utf-8"))

print("=" * 78)
print("6 · IDENTIDADE DO ARTEFATO IMPLANTADO")
print("=" * 78)
print(f"  arquivo  : {CAMINHO_PIPELINE.relative_to(RAIZ)}")
print(f"  SHA-256  : {hash_artefato}")
print(f"  gerado_em: {versoes['gerado_em']}")
print(f"  etapas   : {list(pipeline.named_steps)} · estimador {type(pipeline.named_steps['modelo']).__name__}")
print()
print(f"  MENSAGEM PERMANENTE DO APP: {MENSAGEM_PERMANENTE}")

# %%
# =============================================================================
# 6 · Contrato na fronteira — um payload inválido NÃO chega ao modelo
# =============================================================================
linha_valida = teste_cru.iloc[[0]][list(CAMPOS_ABERTURA)].reset_index(drop=True)
payload_ruim = linha_valida.copy()
payload_ruim.loc[:, "valor_reclamado"] = "21.700,00"

erros_ruim = validar_payload(payload_ruim)
erros_bom = validar_payload(linha_valida)

print("6 · CONTRATO DE DADOS NA FRONTEIRA (camada 2 da vírgula)")
print(f"  payload válido  -> {len(erros_bom)} erro(s) (esperado: 0)")
print(f"  payload com '21.700,00' em valor_reclamado -> {len(erros_ruim)} erro(s):")
for erro in erros_ruim:
    print(f"    {erro}")
print()
try:
    pontuar_lote(payload_ruim, pipeline, k=1)
    print("  FALHA: o payload inválido foi pontuado (não deveria).")
except PayloadRecusado as recusa:
    print(f"  pontuar_lote RECUSOU o lote inteiro (fail closed): {recusa}")
    print(f"  erros em tabela: {recusa.como_tabela().to_dict(orient='records')}")

# %%
# =============================================================================
# 6 · App — fila do mês, valor realizado e governança
# =============================================================================
# A capacidade proporcional ao lote (decisão D4) difere do corte auditado de 900: em um lote
# de 11.904 sinistros, k = round(900 × 11.904 / 12.024). As duas leituras são mostradas.
k_proporcional_teste = k_do_lote(len(teste_cru))
fila_teste = pontuar_lote(teste_cru, pipeline, k=CAPACIDADE_MENSAL)
acerto = avaliar_fila(fila_teste, teste_cru["fraude"].to_numpy())
valor = medir_valor_da_politica(pipeline, teste_cru)
painel = painel_valor()

print("6 · APP — FILA DO MÊS (mês de teste de agosto, 100% dos sinistros)")
print(f"  {resumo_fila(fila_teste, acerto)}")
print()
print(f"  k auditado (capacidade mensal)     : {CAPACIDADE_MENSAL}")
print(f"  k proporcional ao lote (regra D4)  : {k_proporcional_teste} "
      f"(= round(900 × {len(teste_cru):,} / {SINISTROS_POR_MES:,}))")
print(f"  valor realizado pelo app           : {valor}")
print(f"  confere com o resultado auditado   : {'SIM' if valor['confere'] else 'NÃO'}")
print()
print("  PAINEL DE VALOR (diretoria):")
print(f"    política    {brl(painel['politica'])}/mês")
print(f"    regra atual {brl(painel['regra_atual'])}/mês")
print(f"    ganho       {brl(painel['ganho'])}/mês "
      f"(IC 95% [{brl(painel['ganho_ic_inferior'])}; {brl(painel['ganho_ic_superior'])}])")
print()
print("  Semáforo de prioridade por posição (nunca por culpa):")
for posicao in (1, CAPACIDADE_MENSAL // 2, CAPACIDADE_MENSAL, 2 * CAPACIDADE_MENSAL, 3 * CAPACIDADE_MENSAL):
    print(f"    posição {posicao:>5} -> {faixa_prioridade(posicao, CAPACIDADE_MENSAL)}")

# %%
# =============================================================================
# 6 · Governança — saúde do modelo, imputações (camada 3) e drift
# =============================================================================
saude = saude_do_modelo(pipeline, teste_cru)
resumo_imp = resumo_imputacoes(pipeline, teste_cru)
alertas = alertas_imputacao(resumo_imp)
total_imputados = int(resumo_imp["n_imputados"].sum())

tabela_drift = drift_das_features(treino_cru, teste_cru)

print("6 · GOVERNANÇA E MONITORAMENTO")
print(f"  hash do artefato : {saude['hash']}")
print(f"  linhas no lote   : {saude['n_linhas']:,}")
print(f"  imputações (camada 3): {total_imputados} linha(s) imputada(s) no total "
      f"(limiar {saude['limiar_imputacao']:.0%} por feature)")
print(f"  alertas de imputação : {alertas if alertas else 'nenhum'}")
print()
print("  DRIFT DAS FEATURES PRINCIPAIS (PSI treino -> teste):")
for _, linha in tabela_drift.iterrows():
    print(f"    {linha['feature']:<22} PSI {linha['psi']:.6f} · {linha['veredito']}")

# %%
# =============================================================================
# 6 · Explicabilidade (SHAP global) e registro MLOps (degrada sem credencial)
# =============================================================================
amostra_shap = X_teste.sample(n=min(300, len(X_teste)), random_state=SEED)
importancia = importancia_shap(pipeline, X_treino, amostra_shap, semente=SEED)

print("6 · EXPLICABILIDADE (SHAP médio absoluto por feature original, 300 sinistros)")
for nome, valor in importancia.head(5).items():
    print(f"    {nome:<24} {valor:.6f}")
print(f"    ... {len(importancia)} features ao todo")
print()

with contexto_run(
    nome=nome_run("fase6", "notebook_consolidado", SEED),
    parametros={
        "capacidade_mensal": CAPACIDADE_MENSAL,
        "custo_pericia": CUSTO_PERICIA,
        "custo_fraude_paga": CUSTO_FRAUDE_PAGA,
        "seed": SEED,
    },
    tags={"fase": "6", "artefato": "notebook_consolidado", "hash_artefato": hash_artefato},
) as registro:
    try:
        registro.log_metricas(
            {
                "resultado_900": float(valor["resultado"]),
                "tp_900": int(valor["tp"]),
                "ganho_sobre_regra": float(ganho_sobre_regra),
                "diff_ic_inferior": float(resumo_bootstrap["diferenca_ic_inferior"]),
                "diff_ic_superior": float(resumo_bootstrap["diferenca_ic_superior"]),
            }
        )
        print(f"  run MLOps registrada ({registro.uri}) — hash do artefato nas tags.")
    except Exception as erro:  # noqa: BLE001 - o tracking NUNCA derruba o notebook
        print(f"  Aviso: registro no MLflow falhou ({erro!r}); seguindo sem tracking.")

# %% [markdown]
# ## 7 · Quadro de Respostas do comitê — os 24 campos, medidos e impressos
#
# Fecha o projeto: consolida as **Perguntas 1 a 5** a partir dos números **medidos acima** —
# nada digitado à mão. O quadro nasce de `montar_quadro_respostas`, a mesma função que gera o
# CSV e o documento da fase, então as três saídas não podem divergir.

# %%
# =============================================================================
# 7 · QUADRO DE RESPOSTAS — 24 campos (Perguntas 1 a 5)
# =============================================================================
resultado_com_uf_idade = float(metricas_teste["resultado_900"])
resultado_sem_uf_idade = float(metricas_sem_idade_uf["resultado_900"])
preco_de_remover = resultado_com_uf_idade - resultado_sem_uf_idade

respostas_quadro = {
    "colunas_modelo": f"{len(FEATURES)} features: " + ", ".join(FEATURES),
    "colunas_fora": ", ".join(colunas_fora)
    + " (chaves de controle, vazamento e a original redundante `terceiro_identificado`)",
    "idade_uf_decisao": (
        f"Manter: remover `uf` + `idade_segurado` vale {brl(abs(preco_de_remover))}/mês A FAVOR "
        f"da remoção (empate técnico) e o PR AUC cai sem elas."
    ),
    "auc_teste_idade_uf": (
        f"Com idade e UF — ROC {metricas_teste['roc_auc']:.6f} · PR {metricas_teste['pr_auc']:.6f}; "
        f"sem as duas — ROC {metricas_sem_idade_uf['roc_auc']:.6f} · "
        f"PR {metricas_sem_idade_uf['pr_auc']:.6f}."
    ),
    "taxa_base": (
        f"{pct_pt(taxa_base)} — {numero_pt(n_fraudes_base)} fraudes em "
        f"{numero_pt(len(y_completo))} sinistros."
    ),
    "por_tipo": resumo_categoria("tipo_sinistro"),
    "por_canal": resumo_categoria("canal_abertura"),
    "por_boletim": resumo_categoria("boletim_ocorrencia"),
    "boletim_branco": (
        f"Categoria própria '{BOLETIM_NAO_INFORMADO}' — {numero_pt(n_boletim_branco)} sinistros "
        f"({pct_pt(share_boletim_branco)} da base), taxa de fraude {pct_pt(taxa_boletim_branco)}: "
        f"acima da média, então NUNCA imputar (nem pela moda)."
    ),
    "por_faixa_apolice": resumo_categoria("faixa_apolice", list(ROTULOS_APOLICE)),
    "por_faixa_valor": resumo_categoria("faixa_valor_reclamado", list(ROTULOS_VALOR)),
    "modelos_comparados": (
        "6 modelos (Dummy prior · Regra atual Quilha · Árvore de decisão · Random Forest · "
        "XGBoost · LightGBM) — recomendado: XGBoost."
    ),
    "periodos": (
        f"Treino {abertura_treino.min():%d/%m/%Y} a {abertura_treino.max():%d/%m/%Y} "
        f"({numero_pt(len(y_treino))} sinistros, {pct_pt(float(y_treino.mean()))} de fraude) · "
        f"teste {abertura_teste.min():%m/%Y} ({numero_pt(len(y_teste))} sinistros, "
        f"{pct_pt(float(y_teste.mean()))} de fraude)."
    ),
    "referencia_simples": (
        f"ROC AUC {linha_dummy['roc_auc_cv_media']:.6f} · PR AUC "
        f"{linha_dummy['pr_auc_cv_media']:.6f} (DummyClassifier prior)."
    ),
    "auc_recomendado": (
        f"Treino — ROC {linha_campeao_cv['roc_auc_cv_media']:.6f} · PR "
        f"{linha_campeao_cv['pr_auc_cv_media']:.6f}; teste — ROC {metricas_teste['roc_auc']:.6f} · "
        f"PR {metricas_teste['pr_auc']:.6f}."
    ),
    "politica_recomendada": (
        f"Escore do campeão, corte em top-{numero_pt(CAPACIDADE_MENSAL)} (900 perícias/mês)."
    ),
    "economia_politica": (
        f"{tp_campeao_900} fraudes · economia de {brl(economia_campeao)}/mês "
        f"(precisão {pct_pt(tp_campeao_900 / CAPACIDADE_MENSAL)})."
    ),
    "economia_regra": (
        f"{tp_regra_900} fraudes · economia de {brl(economia_regra)}/mês "
        f"(perde até para periciar ao acaso)."
    ),
    "ganho_sobre_regra": f"{brl(ganho_sobre_regra)}/mês no mês de teste.",
    "acao_regra": (
        "Extinguir o critério de R$ 50 mil como regra de triagem; substituí-lo pela política de "
        f"escore do campeão, mantidas as {numero_pt(CAPACIDADE_MENSAL)} perícias/mês."
    ),
    "acerto_minimo": f"{pct_pt(piso)} (700 / 21.700 = 1/31).",
    "k_otimo_equipe": (
        f"k={numero_pt(k_otimo)} (~{round(k_otimo / PERICIAS_POR_PERITO)} peritos, "
        f"{brl(resultado_otimo)}); ganho além das 900 = {brl(ganho_alem_de_900)}/mês."
    ),
    "fraudes_1800": (
        f"{tp_1800} fraudes com 1.800 perícias (múltiplo real {multiplo_real:.2f}× sobre as "
        f"{tp_campeao_900} das 900; a promessa de 2× exigiria {2 * tp_campeao_900})."
    ),
    "acao_equipe": (
        f"Manter os {PERITOS_ATUAIS} peritos ({numero_pt(CAPACIDADE_MENSAL)} perícias/mês); só "
        f"ampliar se a Precisão@900 seguir acima de {pct_pt(piso)} com folga."
    ),
}

quadro_respostas, quadro_markdown = montar_quadro_respostas(respostas_quadro)
CAMINHO_QUADRO = PASTA_TABELAS / "avaliacao_quadro_respostas.csv"

print("=" * 78)
print("7 · QUADRO DE RESPOSTAS DO COMITÊ (24 campos)")
print("=" * 78)
print()
for _, linha in quadro_respostas.iterrows():
    print(f"{linha['pergunta']} | {linha['campo']} | {linha['resposta']}")
print()
preenchidos = int((quadro_respostas["resposta"] != "não informado").sum())
print(f"QUADRO: {preenchidos} de {len(quadro_respostas)} campos preenchidos "
      f"(esperado: {len(CAMPOS_QUADRO_RESPOSTAS)}).")

# %%
# =============================================================================
# 7 · Resposta final, tempo real de execução e divergências honestas
# =============================================================================
FIM = time.perf_counter()
tempo_total = FIM - INICIO
minutos, segundos = divmod(tempo_total, 60)

print("=" * 78)
print("RESPOSTA FINAL DO PROJETO")
print("=" * 78)
print(
    f"Implantar a política de {CAPACIDADE_MENSAL} perícias/mês no escore do campeão: no mês de "
    f"teste ela rende {brl(economia_campeao)} contra {brl(economia_regra)} da regra de R$ 50 mil "
    f"({brl(ganho_sobre_regra)} de ganho), e a vantagem sobrevive ao IC 95% do bootstrap "
    f"[{brl(resumo_bootstrap['diferenca_ic_inferior'])}; {brl(resumo_bootstrap['diferenca_ic_superior'])}] "
    f"e à grade de sensibilidade ({int((~grade['vence_regra']).sum())} de {len(grade)} cenários perdem)."
)
print()
print("-" * 78)
print("DIVERGÊNCIAS HONESTAS (números computados x tabelas salvas)")
print("-" * 78)
print(f"  1. Capacidade do lote: `k_do_lote({len(teste_cru)})` = {k_proporcional_teste}, enquanto o")
print(f"     corte auditado usa as {CAPACIDADE_MENSAL} perícias/mês. Motivo: a regra D4 aplica")
print(f"     `round(900 × n / 12.024)`; o corte auditado fixa a capacidade mensal cheia.")
print(f"  2. `regra_atual_do_case` gera {regra_case['n_acima_limiar']} pedidos acima do limiar,")
print("     enquanto o case fala em ~950 — arredondamento do texto do case.")
print(f"  3. Métricas do treino: este notebook usa a MÉDIA DA CV de 5 folds "
      f"(ROC {linha_campeao_cv['roc_auc_cv_media']:.6f} · PR {linha_campeao_cv['pr_auc_cv_media']:.6f}),")
print("     não o resultado direto no treino — procedência distinta, ambas válidas.")
print()
print("-" * 78)
print(f"TEMPO REAL DE EXECUÇÃO: {tempo_total:.2f} s ({int(minutos):02d}:{segundos:05.2f})")
print("-" * 78)

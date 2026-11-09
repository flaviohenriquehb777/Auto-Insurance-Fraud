# %% [markdown]
# # Fase 5 — Avaliação (CRISP-DM)
#
# **Projeto:** Detecção de fraude em sinistros de automóvel — Quilha Seguros
# **Semente:** `42` · **Mês de avaliação:** agosto/2026 · **Capacidade:** 900 perícias/mês
#
# Esta fase não treina modelo: carrega o campeão aprovado na Fase 4 e o converte em
# **política de perícia**. A decisão é `ordenar pelo escore → cortar em k → periciar`,
# e o critério é o resultado líquido `21.700 × TP − 700 × k`.
#
# Partes: 0 (reconciliação com o case) · 1 (auditoria do artefato) · 2 (backtest) ·
# 3 (política e economia — Pergunta 4) · 4 (dimensionamento — Pergunta 5) ·
# 5 (sensibilidade e incerteza) · 6 (aceitabilidade e risco) · 7 (processo e próximos
# passos) · 8 (consolidação).
#
# MLflow usa DagsHub quando há credenciais e SQLite local quando não há; nenhuma
# célula exige token para concluir.

# %%
# =============================================================================
# Raiz do projeto e imports
# =============================================================================
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone

SEED = 42


def localizar_raiz(inicio: Path) -> Path:
    """Sobe diretórios até localizar o pyproject.toml da raiz."""
    for pasta in [inicio.resolve(), *inicio.resolve().parents]:
        if (pasta / "pyproject.toml").exists():
            return pasta
    raise FileNotFoundError("Raiz do projeto não encontrada (pyproject.toml ausente).")


RAIZ = localizar_raiz(Path.cwd())
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from quilha.avaliacao import (
    N_REAMOSTRAGENS_BOOTSTRAP,
    PERGUNTA_DE_DECISAO,
    aplicar_politica,
    construir_referencias,
    custo_sem_pericia,
    escore_positivo,
    executar_backtest,
    janelas_walk_forward,
    medir_janela,
    piso_viabilidade,
    resultado_economico,
    resultado_liquido,
    resumo_parametros_decisao,
)
from quilha.dados import carregar_base_crua
from quilha.modelagem import (
    CAPACIDADE_MENSAL,
    PASTA_TABELAS,
    calcular_k_proporcional,
    calcular_metricas_holdout,
    carregar_treino_teste,
    separar_entrada_alvo,
)
from quilha.pipeline import CAMINHO_PIPELINE, carregar_pipeline, construir_pipeline
from quilha.preparacao import FEATURES

pd.set_option("display.float_format", lambda valor: f"{valor:,.6f}")

# %% [markdown]
# ## Parte 0 — Reconciliação com o case e travamento do critério de decisão
#
# Antes de qualquer análise: fixar os parâmetros do Anexo 1, derivar o piso de
# viabilidade e declarar por escrito qual é a decisão desta fase e o que a encerra.
# Tudo sai de constantes nomeadas — nenhum número solto no código.

# %%
# =============================================================================
# Parte 0 — Parâmetros do case, piso de viabilidade e pergunta de decisão
# =============================================================================
parametros = resumo_parametros_decisao()

# O piso é derivado na célula, não copiado: 700 / 21.700.
piso = piso_viabilidade()

print("=" * 78)
print("PARÂMETROS DA DECISÃO (Anexo 1 do case)")
print("=" * 78)
for nome, valor in parametros.items():
    if nome == "pergunta_de_decisao":
        continue
    if isinstance(valor, float):
        print(f"  {nome:<34} {valor:,.4f}")
    else:
        print(f"  {nome:<34} {valor}")

print()
print("PISO DE VIABILIDADE (derivado nesta célula)")
print(f"  custo da perícia        = R$ {parametros['custo_pericia']:,.2f}")
print(f"  custo da fraude paga    = R$ {parametros['custo_fraude_paga']:,.2f}")
print(f"  p* = 700 / 21.700       = {piso:.6f}  ({piso * 100:.4f}%)")
print(f"  razão de custo          = {parametros['razao_custo_fraude_pericia']:.2f} : 1")
print()
print("MATEMÁTICA DA POLÍTICA")
print("  Resultado(k) = 21.700 × TP(k) − 700 × k,   sujeito a k ≤ 900/mês")
print("  A k-ésima perícia só se paga enquanto a precisão marginal ≥ 3,2258%")
print()
print("REFERÊNCIA: custo de não periciar ninguém")
print(f"  21.700 × 281 fraudes de agosto            = R$ {custo_sem_pericia(281):,.2f}")
print()
print("CONFERÊNCIA: conta manual de um caso sintético")
print(f"  resultado_liquido(tp=1, k=31) = R$ {resultado_liquido(tp=1, k=31):,.2f}  (deve dar 0)")
print()
print("-" * 78)
print("PERGUNTA QUE ENCERRA A FASE")
print("-" * 78)
print(PERGUNTA_DE_DECISAO)
print()
print(f"Bootstrap da Parte 5: {N_REAMOSTRAGENS_BOOTSTRAP:,} reamostragens (SEED={SEED})")
print(f"Tabelas da fase: {PASTA_TABELAS.relative_to(RAIZ)}")

# %% [markdown]
# **Leitura da Parte 0.** O piso de viabilidade (`3,2258%`) está **acima** da taxa de
# fraude da base (`2,47%`) — periciar ao acaso dá prejuízo. A política só cria valor
# por **concentração**: ela precisa empurrar fraudes para o topo da fila com precisão
# acima do piso. É esse o objeto medido daqui em diante. A conferência
# `resultado_liquido(tp=1, k=31) = 0` fecha o ciclo: 31 perícias custam exatamente uma
# fraude paga, que é a mesma conta vista de dois lados.

# %% [markdown]
# ## Parte 1 — Auditoria do que está sendo avaliado
#
# Provar que a política avaliada é **exatamente** o artefato aprovado na Fase 4 — nem
# outro modelo, nem outra lista de features. Primeiro o tipo do estimador e os
# hiperparâmetros, depois o hash SHA-256 que liga a avaliação ao artefato auditável e,
# por fim, o **teste de fumaça**: o artefato tem de reproduzir o `Resultado@900`
# da Fase 4. Não reproduzir invalida tudo o que vem depois.

# %%
# =============================================================================
# Parte 1 — Tipo do estimador final, hiperparâmetros e features
# =============================================================================
pipeline = carregar_pipeline(CAMINHO_PIPELINE)
modelo_final = pipeline.named_steps["modelo"]

print("ETAPAS DO PIPELINE:", list(pipeline.named_steps))
print("TIPO DO ESTIMADOR FINAL:", type(modelo_final).__name__)
print()

# Tabela de hiperparâmetros de docs/04_modelagem.md, §3 ("Parâmetros do XGBoost campeão").
ESPERADO_FASE4 = {
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
parametros_artefato = modelo_final.get_params()
print("HIPERPARÂMETROS — artefato x Fase 4")
hiperparametros_ok = True
for nome, esperado in ESPERADO_FASE4.items():
    atual = parametros_artefato.get(nome)
    confere = abs(float(atual) - float(esperado)) < 1e-12
    hiperparametros_ok = hiperparametros_ok and confere
    print(f"  {nome:<18} artefato={atual!r:<10} Fase4={esperado!r:<10} {'OK' if confere else 'DIVERGE'}")
print(f"  -> todos conferem: {hiperparametros_ok}")
print()

features_artefato = list(FEATURES)
print(f"FEATURES ({len(features_artefato)}) — ordem canônica de preparacao.FEATURES")
for posicao, nome in enumerate(features_artefato, start=1):
    print(f"  {posicao:>2}. {nome}")

# %%
# =============================================================================
# Parte 1 — Hash SHA-256 do artefato e metadados de versão
# =============================================================================
CAMINHO_VERSOES = CAMINHO_PIPELINE.with_name("pipeline_fraude.versoes.json")
hash_pipeline = hashlib.sha256(CAMINHO_PIPELINE.read_bytes()).hexdigest()
versoes = json.loads(CAMINHO_VERSOES.read_text(encoding="utf-8"))

print("IDENTIDADE DO ARTEFATO SOB AVALIAÇÃO")
print(f"  arquivo  : {CAMINHO_PIPELINE.relative_to(RAIZ)}")
print(f"  tamanho  : {CAMINHO_PIPELINE.stat().st_size:,} bytes")
print(f"  SHA-256  : {hash_pipeline}")
print(f"  gerado_em: {versoes['gerado_em']}")
print(
    "  versões  : "
    f"python {versoes['python']} | sklearn {versoes['scikit_learn']} | "
    f"pandas {versoes['pandas']} | numpy {versoes['numpy']}"
)
print(f"  features no versoes.json: {len(versoes['features'])}")
print(f"  mesma ordem de preparacao.FEATURES: {versoes['features'] == features_artefato}")

# %%
# =============================================================================
# Parte 1 — Teste de fumaça: o artefato reproduz o Resultado@900 da Fase 4?
# =============================================================================
X_treino, y_treino, X_teste, y_teste = carregar_treino_teste()
probabilidades = pipeline.predict_proba(X_teste)[:, 1]
metricas_fumaca = calcular_metricas_holdout(y_teste, probabilidades)

REFERENCIA_FASE4 = {"resultado_900": 3_731_700.0, "tp_900": 201, "fp_900": 699, "fn_900": 80}

print("TESTE DE FUMAÇA (holdout de agosto, corte de 900)")
print(f"  sinistros no teste: {len(y_teste):,}")
print(f"  fraudes no teste  : {int(y_teste.sum())}")
print(f"  TP@900: {metricas_fumaca['tp_900']:>4}  (Fase 4: {REFERENCIA_FASE4['tp_900']})")
print(f"  FP@900: {metricas_fumaca['fp_900']:>4}  (Fase 4: {REFERENCIA_FASE4['fp_900']})")
print(f"  FN@900: {metricas_fumaca['fn_900']:>4}  (Fase 4: {REFERENCIA_FASE4['fn_900']})")
print(
    f"  Resultado@900: R$ {metricas_fumaca['resultado_900']:,.2f}  "
    f"(Fase 4: R$ {REFERENCIA_FASE4['resultado_900']:,.2f})"
)
print(f"  PR AUC: {metricas_fumaca['pr_auc']:.6f} | ROC AUC: {metricas_fumaca['roc_auc']:.6f}")

reproduz = (
    metricas_fumaca["tp_900"] == REFERENCIA_FASE4["tp_900"]
    and metricas_fumaca["fp_900"] == REFERENCIA_FASE4["fp_900"]
    and metricas_fumaca["fn_900"] == REFERENCIA_FASE4["fn_900"]
    and abs(metricas_fumaca["resultado_900"] - REFERENCIA_FASE4["resultado_900"]) < 0.5
)
print()
if reproduz:
    print("VEREDITO: REPRODUZ EXATAMENTE — a avaliação é sobre o artefato aprovado na Fase 4.")
else:
    raise AssertionError("O artefato NÃO reproduz o Resultado@900 da Fase 4: PARAR a fase.")

# %%
# =============================================================================
# Parte 1 — As três referências obrigatórias, medidas no holdout
# =============================================================================
# Os estimadores das referências (b) e (c) são ajustados só no TREINO; (a) não tem
# estimador. Medir as três aqui é o mínimo para "fixá-las": a economia de cada uma
# também é o piso contra o qual a política será comparada nas Partes 3 e 5.
referencias = construir_referencias(k=CAPACIDADE_MENSAL)
referencias["regra_atual"].estimador.fit(X_treino, y_treino)
referencias["aleatoria"].estimador.fit(X_treino, y_treino)

linhas_referencia = []
for chave, politica in referencias.items():
    economia = resultado_economico(y_teste, aplicar_politica(politica, X_teste))
    linhas_referencia.append({"referencia": chave, "politica": politica.nome, **economia})

tabela_referencias = pd.DataFrame(linhas_referencia)
print("AS TRÊS REFERÊNCIAS NO HOLDOUT DE AGOSTO")
print(
    tabela_referencias[
        ["referencia", "n_periciados", "tp", "fp", "fn", "precisao", "economia", "custo_total"]
    ].to_string(index=False)
)
print()
for _, linha in tabela_referencias.iterrows():
    print(
        f"  {linha['politica']:<26} economia = R$ {linha['economia']:>14,.2f} | "
        f"custo total = R$ {linha['custo_total']:>13,.2f} | TP={linha['tp']}"
    )

# %% [markdown]
# **Leitura da Parte 1.** O artefato é o `XGBClassifier` da Fase 4 (hiperparâmetros e as 18
# features conferem) e **reproduz** `Resultado@900` com `TP=201`, `FP=699`, `FN=80` —
# o hash SHA-256 acima é o que amarra esta fase a esse arquivo exato. As três referências
# ficam fixadas como objetos: **não periciar ninguém** (economia 0; custo `21.700 × 281`),
# a **regra atual** de R$ 50 mil e a **referência simples**.
#
# A **regra atual é uma política legítima e pode já ser lucrativa** — ela não é o "zero".
# Por isso o critério da fase é **superá-la**, não apenas ficar positivo (alerta (b) da
# Fase 1). É contra esse número que o campeão será comparado daqui em diante.

# %% [markdown]
# ## Parte 2 — Estimativa honesta fora da amostra
#
# O `R$ 3.731.700` de agosto é evidência de **seleção**: agosto foi usado para escolher o
# campeão. Uma decisão de comitê não pode se apoiar só nisso. Aqui a avaliação vira
# **origem móvel (walk-forward)**: treina em `abr..mn`, avalia no mês **seguinte**, avança o
# treino — sem embaralhar o tempo. O campeão é **reajustado** em cada janela com os
# hiperparâmetros **congelados** (`clone`) — não há reabertura de tuning; só os dados mudam.
# Cada janela é uma amostra que o modelo não viu.

# %%
# =============================================================================
# Parte 2 — Janelas cronológicas de origem móvel
# =============================================================================
base_completa = carregar_base_crua()
datas_abertura = pd.to_datetime(base_completa["data_abertura"])
X_completo, y_completo = separar_entrada_alvo(base_completa)
modelo_campeao = pipeline.named_steps["modelo"]

janelas = janelas_walk_forward(datas_abertura, minimo_meses_treino=1)

linhas_janela = []
for janela in janelas:
    y_aval = y_completo.loc[janela["mascara_avaliacao"]]
    linhas_janela.append(
        {
            "treino": janela["treino_rotulo"],
            "avaliacao": janela["aval_rotulo"],
            "n_sinistros": len(y_aval),
            "n_fraudes": int(y_aval.sum()),
            "taxa_fraude": float(y_aval.mean()),
            "k_proporcional": calcular_k_proporcional(len(y_aval)),
        }
    )
tabela_janelas = pd.DataFrame(linhas_janela)

print(f"JANELAS DE ORIGEM MÓVEL ({len(tabela_janelas)} avaliações, treino expansivo)")
print(tabela_janelas.to_string(index=False))

# %%
# =============================================================================
# Parte 2 — Backtest: campeão x regra atual x referência simples
# =============================================================================
tabela_backtest = executar_backtest(X_completo, y_completo, datas_abertura, modelo_campeao)

colunas_backtest = [
    "janela", "treino", "politica", "n_sinistros", "n_fraudes", "taxa_fraude", "k",
    "tp", "fp", "precisao_k", "resultado_k", "pr_auc", "roc_auc",
]
print("BACKTEST POR JANELA E POLÍTICA")
print(tabela_backtest[colunas_backtest].to_string(index=False))

# %%
# =============================================================================
# Parte 2 — Conferência interna: o reajuste reproduz o artefato auditado?
# =============================================================================
# Se o reajuste da última janela (abr–jul -> ago), feito com `clone`, não casar com o
# artefato, o backtest não estaria medindo o mesmo modelo que está na prateleira.
ultima = janelas[-1]
X_treino_ultima = X_completo.loc[ultima["mascara_treino"]]
y_treino_ultima = y_completo.loc[ultima["mascara_treino"]]
X_aval_ultima = X_completo.loc[ultima["mascara_avaliacao"]]
y_aval_ultima = y_completo.loc[ultima["mascara_avaliacao"]]

campeao_reajustado = construir_pipeline(modelo=clone(modelo_campeao))
campeao_reajustado.fit(X_treino_ultima, y_treino_ultima)

escore_reajuste = escore_positivo(campeao_reajustado, X_aval_ultima)
escore_artefato = escore_positivo(pipeline, X_aval_ultima)
diferenca_maxima = float(np.max(np.abs(escore_reajuste - escore_artefato)))
metricas_reajuste_900 = medir_janela(y_aval_ultima, escore_reajuste, k=900)

print(f"última janela: treino {ultima['treino_rotulo']} -> avaliação {ultima['aval_rotulo']}")
print(f"diferença máxima de probabilidade reajuste x artefato: {diferenca_maxima:.3e}")
print(
    f"Resultado@900 do reajuste: R$ {metricas_reajuste_900['resultado_k']:,.2f} "
    f"(Fase 4: R$ 3.731.700,00) | TP={metricas_reajuste_900['tp']} "
    f"FP={metricas_reajuste_900['fp']} FN={metricas_reajuste_900['fn']}"
)

# %%
# =============================================================================
# Parte 2 — Média ± desvio entre janelas e diferença pareada
# =============================================================================
pivot_resultado = tabela_backtest.pivot(index="janela", columns="politica", values="resultado_k")
pivot_resultado = pivot_resultado[["campeao", "regra_atual", "aleatoria"]]
pivot_resultado["campeao_menos_regra"] = (
    pivot_resultado["campeao"] - pivot_resultado["regra_atual"]
)

print("RESULTADO@k POR JANELA (R$)")
print(pivot_resultado.to_string())
print()
for coluna in ["campeao", "regra_atual", "aleatoria", "campeao_menos_regra"]:
    serie = pivot_resultado[coluna]
    print(f"  {coluna:<20} média R$ {serie.mean():>14,.2f}  ±  R$ {serie.std():>13,.2f}")

diferenca = pivot_resultado["campeao_menos_regra"]
print()
print(
    f"  diferença pareada: R$ {diferenca.mean():,.2f} ± R$ {diferenca.std():,.2f}  ->  "
    + ("RUÍDO entre janelas" if diferenca.std() > diferenca.mean() else "acima do ruído")
)

# %%
# =============================================================================
# Parte 2 — Degradação temporal e frase de decisão
# =============================================================================
print("DEGRADAÇÃO TEMPORAL — campeão x regra atual, em ordem cronológica")
for janela, linha in pivot_resultado.iterrows():
    print(
        f"  {janela}  campeão R$ {linha['campeao']:>13,.2f}  |  "
        f"regra atual R$ {linha['regra_atual']:>13,.2f}"
    )
valor_inicial = pivot_resultado["campeao"].iloc[0]
valor_final = pivot_resultado["campeao"].iloc[-1]
print(
    f"  primeira janela R$ {valor_inicial:,.2f}  ->  última janela R$ {valor_final:,.2f}  |  "
    f"variação R$ {valor_final - valor_inicial:,.2f}"
)

CAMINHO_BACKTEST = PASTA_TABELAS / "avaliacao_backtest.csv"
tabela_backtest.to_csv(CAMINHO_BACKTEST, index=False)
print()
print(f"Tabela salva em {CAMINHO_BACKTEST.relative_to(RAIZ)} ({len(tabela_backtest)} linhas).")

n_janelas = len(pivot_resultado)
print()
if diferenca.mean() > 0:
    print(
        f"DECISÃO: fora da amostra, em {n_janelas} janelas, o campeão supera a regra atual em "
        f"R$ {diferenca.mean():,.2f} ± R$ {diferenca.std():,.2f} por mês."
    )
else:
    print(
        f"DECISÃO: fora da amostra, em {n_janelas} janelas, o campeão NÃO supera a regra atual "
        f"(diferença R$ {diferenca.mean():,.2f} ± R$ {diferenca.std():,.2f} por mês)."
    )

# %% [markdown]
# **Leitura da Parte 2.** O backtest responde ao limite herdado: agora existe uma estimativa em
# janelas que o modelo **não viu**. O `k` de cada janela é proporcional à capacidade mensal
# (`round(900 × n / 12.024)`), porque os meses variam de tamanho — e isso é dito, não escondido
# (guardrail da unidade de tempo). O reajuste da última janela tem de casar com o artefato
# auditado: se as probabilidades divergirem, o backtest não mediria o modelo da prateleira.
#
# A leitura de negócio é a **diferença pareada** campeão − regra atual janela a janela: é ela
# que entra na decisão, e é ela que a Parte 5 submeterá ao intervalo de confiança. Sinais
# menores que o erro de amostragem são **ruído**, não ganho.

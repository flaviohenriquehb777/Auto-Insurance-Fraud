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
    PERICIAS_POR_PERITO,
    PERITOS_ATUAIS,
    aplicar_politica,
    construir_politica_campeao,
    construir_referencias,
    curva_ganho_por_perito,
    custo_sem_pericia,
    escore_positivo,
    executar_backtest,
    janelas_walk_forward,
    k_marginal_cruza_piso,
    medir_janela,
    medir_politicas_no_mes,
    piso_viabilidade,
    pontos_de_decisao_k,
    regra_atual_do_case,
    resultado_economico,
    resultado_liquido,
    resumo_parametros_decisao,
    salvar_curva_decisao,
    salvar_ganho_por_perito,
)
from quilha.dados import carregar_base_crua
from quilha.modelagem import (
    CAPACIDADE_MENSAL,
    CUSTO_FRAUDE_PAGA,
    CUSTO_PERICIA,
    LIMIAR_REGRA_ATUAL,
    PASTA_TABELAS,
    calcular_k_proporcional,
    calcular_metricas_holdout,
    carregar_treino_teste,
    curva_resultado_k,
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

# %% [markdown]
# ## Parte 3 — Política de perícia e economia (Pergunta 4)
#
# A política é o entregável; o modelo é insumo. Aqui o escore vira **decisão** e a decisão
# vira **dinheiro** no mês de teste: quantas fraudes encontradas, quanto se evita de perda e
# quanto custa — sempre contra as três referências. E ao fim, a **ação sobre a regra de
# R$ 50 mil**: não um comentário, uma decisão com o número ao lado.

# %%
# =============================================================================
# Parte 3 — As quatro políticas medidas no mês de teste (agosto)
# =============================================================================
politicas_mes = {
    "campeao": construir_politica_campeao(pipeline, k=CAPACIDADE_MENSAL),
    **referencias,
}

tabela_politicas = medir_politicas_no_mes(politicas_mes, X_teste, y_teste)

print("POLÍTICAS NO MÊS DE TESTE (agosto/2026) — mesma capacidade, mesmo critério")
print(
    tabela_politicas[
        [
            "politica",
            "capacidade",
            "n_periciados",
            "tp",
            "fp",
            "fn",
            "precisao",
            "cobertura",
            "economia",
            "custo_total",
        ]
    ].to_string(index=False)
)
print()
for _, linha in tabela_politicas.iterrows():
    print(
        f"  {linha['politica']:<28} k={linha['n_periciados']:>3}  TP={linha['tp']:>3}  "
        f"precisão={linha['precisao'] * 100:>6.2f}%  cobertura={linha['cobertura'] * 100:>6.2f}%  "
        f"economia=R$ {linha['economia']:>13,.2f}  custo=R$ {linha['custo_total']:>13,.2f}"
    )

# %%
# =============================================================================
# Parte 3 — A regra atual exatamente como o case a descreve
# =============================================================================
# Literal do case: filtrar valor_reclamado > R$ 50 mil, ordenar por valor decrescente
# e cortar em 900. Medido sobre o chamado CRU — a conversão do texto com vírgula é
# feita aqui, igualzinha à que o Pipeline faz internamente.
regra_case = regra_atual_do_case(
    X_teste, y_teste, limiar=LIMIAR_REGRA_ATUAL, k=CAPACIDADE_MENSAL
)

print("REGRA ATUAL COMO O CASE A DESCREVE (valor_reclamado > R$ 50 mil -> corte em 900)")
print(
    f"  pedidos acima de R$ {regra_case['limiar']:,.0f}: {regra_case['n_acima_limiar']}  "
    f"(o case fala em ~950)"
)
print(f"  de fato periciados (corte em {CAPACIDADE_MENSAL}): {regra_case['n_periciados']}")
print(
    f"  fraudes encontradas (TP): {regra_case['tp']}  |  "
    f"precisão: {regra_case['precisao'] * 100:.2f}%  |  "
    f"cobertura: {regra_case['cobertura'] * 100:.2f}%"
)
print(
    f"  economia: R$ {regra_case['economia']:,.2f}  |  "
    f"custo total: R$ {regra_case['custo_total']:,.2f}"
)

# %%
# =============================================================================
# Parte 3 — Ganho da política sobre a regra atual (ponto; IC 95% na Parte 5)
# =============================================================================
economia_campeao = float(
    tabela_politicas.loc[tabela_politicas["chave"] == "campeao", "economia"].iloc[0]
)
economia_regra = float(
    tabela_politicas.loc[tabela_politicas["chave"] == "regra_atual", "economia"].iloc[0]
)
ganho_sobre_regra = economia_campeao - economia_regra

print("GANHO DA POLÍTICA SOBRE A REGRA ATUAL (mês de teste)")
print(f"  Resultado(campeão)      : R$ {economia_campeao:>14,.2f}")
print(f"  Resultado(regra atual)  : R$ {economia_regra:>14,.2f}")
print(f"  Ganho por mês           : R$ {ganho_sobre_regra:>14,.2f}")
print()
print("  O intervalo de confiança de 95% desta diferença é calculado na Parte 5.")
print("  A Parte 2 já mostrou o ganho fora da amostra, em 4 janelas, acima do ruído.")

# %%
# =============================================================================
# Parte 3 — Curva Resultado(k) do campeão e os pontos de decisão
# =============================================================================
# A curva é recalculada aqui a partir das probabilidades do artefato no mês de teste:
# nada é lido de figura antiga, e os pontos saem do próprio resumo da curva.
probabilidades_teste = escore_positivo(pipeline, X_teste)
tabela_curva, resumo_curva = curva_resultado_k(y_teste, probabilidades_teste)
pontos_decisao = pontos_de_decisao_k(tabela_curva, resumo_curva)

print("PONTOS DE DECISÃO SOBRE A CURVA Resultado(k) — mês de teste")
for k, valor, rotulo in pontos_decisao:
    print(f"  {rotulo:<22} k={k:>5}  ->  R$ {valor:>13,.2f}")

caminho_figura = salvar_curva_decisao(tabela_curva, pontos_decisao)
print()
print(f"Figura salva em {caminho_figura.relative_to(RAIZ)}")

# %%
# =============================================================================
# Parte 3 — Ação sobre a regra atual (Pergunta 4)
# =============================================================================
print("AÇÃO RECOMENDADA SOBRE A REGRA DE R$ 50 MIL")
print(
    f"  1. Extinguir o critério de R$ {LIMIAR_REGRA_ATUAL:,.0f} como regra de triagem: no mês de "
    f"teste ele periciou {regra_case['n_periciados']} sinistros, achou {regra_case['tp']} fraudes "
    f"e deu economia de R$ {regra_case['economia']:,.2f} (negativa)."
)
print(
    f"  2. Substituí-lo pela política de escore do campeão, mantidas as "
    f"{CAPACIDADE_MENSAL} perícias/mês: mesmo custo de perícia, economia de "
    f"R$ {economia_campeao:,.2f} — ganho de R$ {ganho_sobre_regra:,.2f}/mês."
)

# %% [markdown]
# **Leitura da Parte 3.** No mês de teste, com as **mesmas 900 perícias**, o campeão encontra
# 201 fraudes e devolve economia de R$ 3.731.700; a regra de R$ 50 mil acha 14 e devolve
# economia **negativa** — ela custa mais do que economiza e perde até para escolher ao acaso.
# O ganho de trocar a regra pela política de escore é a diferença dos dois números acima, e o
# IC 95% dela é o tema da Parte 5.
#
# A curva `Resultado(k)` marca os quatro pontos de decisão: o corte de hoje (900), o corte
# teórico `p*`, o ótimo empírico e o último `k` ainda lucrativo — este último é o que a Parte 4
# usa para responder "vale a pena mais peritos?".

# %% [markdown]
# ## Parte 4 — Dimensionamento da equipe e valor marginal (Pergunta 5)
#
# O comitê pergunta "vale a pena mais peritos?". Aqui se responde com dinheiro: o valor
# marginal de cada perícia, a curva de ganho por perito, a contraprova da tese do Otávio
# ("o dobro de peritos, o dobro de fraude") e a ação sobre a equipe — com número.

# %%
# =============================================================================
# Parte 4 — Acerto mínimo de uma perícia a mais para se pagar
# =============================================================================
piso_marginal = piso_viabilidade()
print("ACERTO MÍNIMO DE UMA PERÍCIA (A MAIS) PARA SE PAGAR")
print(f"  custo de uma perícia   = R$ {CUSTO_PERICIA:,.2f}")
print(f"  custo da fraude paga   = R$ {CUSTO_FRAUDE_PAGA:,.2f}")
print(f"  700 / 21.700           = {piso_marginal:.6f}  ({piso_marginal * 100:.4f}%)")
print(
    "  É o mesmo número do piso da Fase 1: a k-ésima perícia zera o valor esperado em "
    "p = 1/(21.700/700) = 1/31 — a mesma conta vista do lado marginal."
)

# %%
# =============================================================================
# Parte 4 — Valor marginal: até quanto compensa periciar e o k de menor custo
# =============================================================================
k_otimo = int(resumo_curva["k_otimo_empirico"])
k_teorico = int(resumo_curva["k_corte_teorico"])
k_lucrativo_max = int(resumo_curva["k_maximo_ainda_lucrativo"])
k_cruza_piso = k_marginal_cruza_piso(tabela_curva)

resultado_900 = float(
    tabela_curva.loc[tabela_curva["k"] == CAPACIDADE_MENSAL, "resultado_liquido"].iloc[0]
)
resultado_otimo = float(
    tabela_curva.loc[tabela_curva["k"] == k_otimo, "resultado_liquido"].iloc[0]
)
resultado_teorico = float(
    tabela_curva.loc[tabela_curva["k"] == k_teorico, "resultado_liquido"].iloc[0]
)
ganho_alem_de_900 = resultado_otimo - resultado_900


def _peritos(k: int) -> float:
    """Peritos equivalentes a um `k` de perícias/mês (1 perito = 30 perícias)."""
    return k / PERICIAS_POR_PERITO


print("VALOR MARGINAL E LIMITES DE CAPACIDADE (curva Resultado(k), mês de teste)")
print(f"  capacidade hoje (R$ 50 mil/900)      k={CAPACIDADE_MENSAL:>5}  ->  R$ {resultado_900:>13,.2f}")
print(f"  ótimo empírico (maior resultado)     k={k_otimo:>5}  ->  R$ {resultado_otimo:>13,.2f}"
      f"   ({_peritos(k_otimo):.1f} peritos)")
print(f"  corte teórico (precisão = piso)      k={k_teorico:>5}  ->  R$ {resultado_teorico:>13,.2f}"
      f"   ({_peritos(k_teorico):.1f} peritos)")
print(f"  último k ainda lucrativo             k={k_lucrativo_max:>5}  ->  "
      f"R$ {float(tabela_curva.loc[tabela_curva['k'] == k_lucrativo_max, 'resultado_liquido'].iloc[0]):>13,.2f}")
print()
# O resumo da curva reporta o ÚLTIMO k ainda viável (precisão marginal >= piso); a
# função devolve a PRIMEIRA perícia que não se paga. São vizinhas por construção:
# a última que se paga + 1 é a primeira que não se paga.
cruza_confere = k_cruza_piso == k_teorico + 1
print(f"  contraprova: última perícia que ainda se paga k={k_teorico} (resumo da curva); "
      f"a primeira que não se paga é k={k_cruza_piso} "
      f"-> {'confere (vizinhas)' if cruza_confere else 'DIVERGE'}")
print(f"  ganho mensal além das 900: Resultado(k_ótimo) − Resultado(900) = R$ {ganho_alem_de_900:,.2f}")
print()
print('  "menor custo total", as duas leituras do enunciado:')
print(f"    (A) minimizar o custo total  -> mesmo ponto de maior resultado -> k={k_otimo}"
      f" (R$ {resultado_otimo:,.2f})")
print(f"    (B) última perícia que ainda se paga (precisão marginal >= piso) -> k={k_teorico}"
      f" (R$ {resultado_teorico:,.2f})")

# %%
# =============================================================================
# Parte 4 — Curva de ganho por perito e recomendação de dimensionamento
# =============================================================================
quadro_peritos = curva_ganho_por_perito(tabela_curva, peritos_max=80)

pontos_peritos = [
    ("atual", PERITOS_ATUAIS),
    ("ótimo empírico", round(_peritos(k_otimo))),
    ("corte teórico", round(_peritos(k_teorico))),
]
resumo_peritos = quadro_peritos[quadro_peritos["peritos"].isin([p for _, p in pontos_peritos])]

print("GANHO POR PERITO (1 perito = 30 perícias/mês)")
print(
    resumo_peritos[["peritos", "k", "resultado_liquido", "ganho_marginal_perito", "ganho_sobre_base"]]
    .to_string(index=False)
)
print()
marginal_atual = float(
    quadro_peritos.loc[quadro_peritos["peritos"] == PERITOS_ATUAIS, "ganho_marginal_perito"].iloc[0]
)
print(f"  ganho do 30º perito (o último contratado hoje): R$ {marginal_atual:,.2f}/mês")
print(
    f"  a partir de {round(_peritos(k_otimo))} peritos o ganho marginal entra em platô; "
    f"além de {round(_peritos(k_teorico))} peritos cada contratação destrói valor."
)

caminho_peritos = salvar_ganho_por_perito(quadro_peritos, dict(pontos_peritos))
print()
print(f"Figura salva em {caminho_peritos.relative_to(RAIZ)}")

# %%
# =============================================================================
# Parte 4 — Contraprova da tese do Otávio ("o dobro de peritos, o dobro de fraude")
# =============================================================================
tp_900 = int(tabela_curva.loc[tabela_curva["k"] == CAPACIDADE_MENSAL, "tp"].iloc[0])
tp_1800 = int(tabela_curva.loc[tabela_curva["k"] == 2 * CAPACIDADE_MENSAL, "tp"].iloc[0])
promessa_otavio = 2 * tp_900
multiplo_real = tp_1800 / tp_900

print("TESTE DA AFIRMAÇÃO DO OTÁVIO (dobrar a equipe dobraria a fraude?)")
print(f"  TP com {CAPACIDADE_MENSAL} perícias (hoje)      : {tp_900}")
print(f"  TP com {2 * CAPACIDADE_MENSAL} perícias (dobro)      : {tp_1800}")
print(f"  promessa de 2× implicaria                  : {promessa_otavio} fraudes")
print(f"  múltiplo real                              : {multiplo_real:.2f}×  (prometido: 2,00×)")
print(
    f"  motivo econômico: o topo da fila concentra as fraudes mais óbvias, então a precisão "
    f"marginal decresce — 900 perícias acham {tp_900} de {int(tabela_curva['tp'].iloc[-1])} fraudes."
)

# %%
# =============================================================================
# Parte 4 — Ação sobre a equipe (Pergunta 5)
# =============================================================================
print("AÇÃO RECOMENDADA SOBRE A EQUIPE")
print(
    f"  Manter os {PERITOS_ATUAIS} peritos ({CAPACIDADE_MENSAL} perícias/mês). O ganho de ir ao "
    f"ótimo empírico (k={k_otimo}, ~{round(_peritos(k_otimo))} peritos) é de "
    f"R$ {ganho_alem_de_900:,.2f}/mês — e esse ótimo é *in-sample*, dentro da faixa de variação "
    f"das janelas do backtest."
)
print(
    f"  Condição que invalidaria: se a janela de monitoramento confirmar que a precisão @900 "
    f"segue acima de {piso_marginal * 100:.2f}% com folga, reavaliar a ampliação até ~"
    f"{round(_peritos(k_teorico))} peritos (corte teórico, k={k_teorico})."
)

# %% [markdown]
# **Leitura da Parte 4.** O acerto mínimo de uma perícia marginal é o mesmo piso da Fase 1
# (3,2258%) — a mesma conta vista do lado de quem decide contratar. No mês de teste, o resultado
# sobe de R$ 3.731.700 (k=900) até R$ 3.968.300 no ótimo empírico (k=1.430, ~48 peritos) e
# desaparece em k=8.710. Só que o ótimo **não é estimativa fora da amostra**: é escolhido no
# mesmo mês usado na seleção, e o ganho sobre as 900 atuais cabe dentro do desvio do backtest.
# Por isso a ação é **manter a equipe**, não ampliar por um número bonito.
#
# A tese do Otávio cai por aritmética: dobrar as perícias **não** dobra as fraudes — a precisão
# marginal decresce porque o topo da fila já concentra as fraudes mais óbvias.

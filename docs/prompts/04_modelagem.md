# FASE 4 — MODELAGEM (CRISP-DM) · Prompt de execução

**Projeto:** Detecção de fraude em sinistros de automóvel — Quilha Seguros
**Fase:** 4 de 6 (CRISP-DM — Modelagem)
**Status:** concluída (Partes 0 a 6 executadas; evidências em `docs/04_modelagem.md`)
**Pré-requisito:** Fases 1 a 3 concluídas e verificadas (`docs/01_entendimento_negocio.md`,
`docs/02_entendimento_dados.md`, `docs/03_preparacao_dados.md`)

---

## Como executar este documento

Para dar início a esta fase, basta pedir:

> **"Fase 4 — execute o prompt em `docs/prompts/04_modelagem.md`, parte por parte, começando pela Parte 0."**

A frase aponta para este arquivo de propósito: mesmo que o contexto da conversa se perca,
o documento é autocontido e traz todo o estado herdado, os números do negócio e as decisões
já fechadas nas fases anteriores.

**Partes:** 0 (MLOps) · 1 (modelos) · 2 (validação) · 3 (hiperparâmetros e desbalanceamento) ·
4 (undersampling) · 5 (campeão, custo 31:1 e SHAP) · 6 (integração).

---

## Contexto herdado (não reabrir)

**Estado:** Fases 1 a 3 concluídas. Existem `quilha.dados` (`carregar_base_crua`,
`dividir_temporal`, `encontrar_raiz`, `PASTA_MODELOS`), `quilha.preparacao` (`preparar`,
`FEATURES` com 18 features), `quilha.pipeline` (`construir_pipeline(modelo=...)`,
`salvar_pipeline`, `carregar_pipeline`, `versoes`, `CAMINHO_PIPELINE`) e `quilha.explicacao`
(`mapear_features`, `agregar_contribuicoes`, `explicar_sinistro`). O `Pipeline` é
`preparo → imputacao → modelo`, com `set_output(transform="pandas")`; o artefato versionado
agora contém o XGBoost campeão. `DummyClassifier` permanece somente como baseline.

**Insumos econômicos e operacionais (fixos):**

- Capacidade de perícia: **900/mês**. Custo da perícia: **R$ 700**. Fraude paga (FN):
  **R$ 21.700** médios.
- Relação de custo: **31:1** (21.700 / 700) — um falso negativo custa 31 falsos positivos.
- Corte teórico ótimo (custo unitário): **p\* = 700 / 21.700 = 3,2258%**. Um sinistro só paga
  a perícia se a probabilidade de fraude passar disso. É o mesmo piso de viabilidade da
  Fase 1.
- Base: 60.000 sinistros, fraude 2,47%. Treino (abr–jul): 48.096 sinistros, 1.199 fraudes
  (2,49%), 12.024/mês. Teste (ago): 11.904 sinistros, 281 fraudes (2,36%).

**Decisões já fechadas que este prompt não reabre:**

- `SEED = 42` em tudo que tem aleatoriedade. **Sem `shuffle` e sem `stratify` no holdout** —
  o corte é cronológico.
- **Acurácia vetada** como métrica de decisão. **Sem SMOTE** (cria fraude que não existe e
  distorce a probabilidade, que é o que o app mostra e o SHAP explica).
- `valor_indenizado`, `id_sinistro`, `data_ocorrencia`, `data_abertura` e `fraude` **nunca**
  entram como atributo.
- `fim_de_semana` fica (medido: ±0,04 p.p., custo zero, derivada da data).
  `add_indicator=True` fica e está testado. `boletim_ocorrencia` fica com a hipótese de
  *timing* registrada.
- `models/pipeline_fraude.versoes.json` não é versionado; o `.joblib` é.

**Métricas oficiais desta fase:**

- **PR AUC** — métrica técnica principal (referência = taxa base 2,47%, **não** 0,5).
- **ROC AUC** — exigida pelo quadro de respostas do case (Perguntas 1 e 3).
- **Resultado Líquido@k = 21.700 × TP − 700 × k** — métrica de decisão do negócio.
- **Precisão@k** e **Cobertura@k** — apoio.
- **Brier score e curva de calibração** — diagnóstico obrigatório (a probabilidade é
  entregue ao analista).

---

## Parte 0 — Infraestrutura de MLOps (primeiro, antes de qualquer modelo)

Não retrofitar depois. Antes de treinar o primeiro modelo:

1. Adicionar `xgboost` e `lightgbm` às dependências do projeto com `uv add` e confirmar que
   o `uv export` (requirements do Streamlit Cloud) os inclui.
2. Criar `src/quilha/mlops.py` com: leitura de `MLFLOW_TRACKING_URI`,
   `MLFLOW_TRACKING_USERNAME` e `MLFLOW_TRACKING_PASSWORD` **por variável de ambiente**
   (nunca em código, nunca no repositório); `experimento()` que devolve
   `"quilha-fraude-fase4"`; convenção de nome de run (`<familia>__<config>__seed<seed>`); e
   `contexto_run(...)` como *context manager*.
3. **Degradação limpa:** se as credenciais não existirem, o módulo emite um aviso claro e o
   treino segue normalmente, gravando em `mlruns/` local. O notebook **tem de rodar de ponta
   a ponta sem token** — é pré-requisito para o notebook consolidado que a equipe vai testar.
4. Logar sempre **o Pipeline inteiro** (`preparo + imputacao + modelo`), nunca só o
   estimador — é o objeto que o app carrega. Use `mlflow.sklearn.log_model` com **assinatura
   inferida sobre o dataframe CRU de entrada** (as 15 colunas que o app envia: as 17 do CSV
   menos `fraude` e `valor_indenizado`) e `input_example` de uma linha crua.
5. Registrar em cada run: hiperparâmetros + `seed` + lista de features; métricas
   (média ± desvio dos 5 folds **e** o valor no holdout); e como artefatos a tabela da CV, o
   `pipeline.joblib`, `versoes.json`, o gráfico SHAP, a curva `Resultado(k)` e a matriz de
   confusão do corte em 900.
6. Logar o `uv.lock` (hash) e o commit do git quando existir — sem isso a run não é
   reproduzível.
7. **Não logar a base inteira** (nem o CSV nem a matriz transformada): logar apenas o
   schema/assinatura. Dado bruto fica fora do tracking.
8. Model Registry: registrar os candidatos com `mlflow.register_model` e promover o campeão
   com **alias** (`champion`) via `MlflowClient.set_registered_model_alias`. Use alias, não
   *stages* — stages estão descontinuados no MLflow atual. Descrição e tags do modelo
   registrado citam as métricas que o elegeram.

---

## Parte 1 — Modelos e baselines (todos pelo mesmo protocolo)

Rode **todos** e devolva a tabela comparativa:

| # | Modelo | Papel |
|---|---|---|
| 1 | `DummyClassifier(strategy="prior")` | Referência simples (exigida na Pergunta 3 do case) |
| 2 | **Regra Atual da Quilha** | Baseline de negócio |
| 3 | `DecisionTreeClassifier` | Interpretabilidade / profundidade ótima |
| 4 | `RandomForestClassifier` | Bagging |
| 5 | `XGBoost` | Boosting |
| 6 | `LightGBM` | Boosting |

**A Regra Atual precisa ser implementada como estimador** (`BaseEstimator` +
`ClassifierMixin`, com `classes_ = [0, 1]` e `predict_proba` devolvendo, na coluna 1, uma
nota **monotônica em `valor_reclamado`**, zerada abaixo de R$ 50 mil). Sem isso ela não entra
na mesma CV nem nas mesmas métricas — e é justamente a comparação que o comitê quer ver.

**Todos** passam por: `construir_pipeline(modelo=...)`, `random_state=42`, mesmo split,
mesmas métricas, mesmo log.

---

## Parte 2 — Protocolo de validação (sem nenhum vazamento)

1. **CV dentro do treino:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
   Aqui pode embaralhar — o holdout é que é cronológico.
2. **O Pipeline inteiro entra na CV.** Nunca passe uma matriz já transformada para o
   `cross_validate`: imputador e `OneHotEncoder` têm de ser ajustados **dentro de cada
   fold**. É a diferença entre uma métrica honesta e uma métrica bonita.
3. **Corte proporcional na CV:** o fold tem ~9.619 sinistros, e a capacidade é mensal
   (12.024 sinistros/mês). Logo **k_fold = round(900 × n_fold / 12.024) ≈ 720**. No holdout
   (11.904 ≈ um mês), **k = 900**. Deixe `CAPACIDADE` e `SINISTROS_POR_MES` como constantes
   nomeadas, não como número solto.
4. **Scorer próprio** para o resultado financeiro (`make_scorer` com
   `response_method="predict_proba"`, maior é melhor), rodando junto com `roc_auc` e
   `average_precision` no mesmo `cross_validate`. Use `return_estimator=True` para ficar com
   os modelos de cada fold.
5. **Nada de `refit` em métrica técnica.** O critério de escolha é dinheiro; PR AUC e
   Precisão@900 desempatam e explicam.
6. Reporte **média ± desvio** dos 5 folds em toda comparação. **Toda diferença menor que o
   desvio da diferença pareada é ruído** — foi assim que fechamos a ablação na Fase 3, e vale
   aqui.
7. `early_stopping` (XGBoost/LightGBM) **jamais** pode ver o holdout: use uma fatia do treino
   do fold.

---

## Parte 3 — Ajuste de hiperparâmetros e tratamento do desbalanceamento

1. **Árvore de decisão:** varra `max_depth` de 1 a 20 **e** `None`, cruzando com
   `min_samples_leaf ∈ {1, 5, 20, 50}` e `class_weight ∈ {None, "balanced"}`. Uma curva de
   `max_depth` × PR AUC e × Resultado@k, com o número de folhas ao lado — o comitê precisa
   ver onde o ganho para de crescer e a variância começa.
2. **Random Forest:** `n_estimators ∈ {300, 500}`, `max_depth ∈ {None, 5, 10, 15, 20}`,
   `min_samples_leaf ∈ {1, 5, 20}`, `class_weight ∈ {"balanced", "balanced_subsample"}`.
3. **XGBoost:** `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`,
   `colsample_bytree`, `gamma`, `reg_lambda`.
4. **LightGBM:** `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`,
   `feature_fraction`, `bagging_fraction`, `lambda_l2`, `is_unbalance`.
5. **`scale_pos_weight` / `class_weight` — teste os dois valores, e diga qual venceu:**
   - **39,11** = balanceamento puro (46.897 legítimos / 1.199 fraudes no treino);
   - **31** = razão de **custo** (21.700 / 700), que é a teoria sob decisão assimétrica.

   Não assuma que o maior é melhor: sob função de custo 31:1, o valor ponderado pelo custo é
   o candidato teoricamente correto. E **registre a consequência**: rebalancear **distorce a
   probabilidade**, e nós entregamos probabilidade ao analista e usamos o corte de 3,23%. Por
   isso o Brier e a curva de calibração desta parte não são opcionais.
6. Busca com `RandomizedSearchCV` (ou equivalente) usando **a mesma**
   `StratifiedKFold(5, shuffle=True, random_state=42)` em todos os modelos, `n_iter` declarado
   e logado. Rode com `scoring` múltiplo e `refit` no scorer financeiro.
7. Se `n_jobs`/threads tornarem a busca instável, fixe `n_jobs` e logue o tempo de cada run.

---

## Parte 4 — UnderSample inteligente (teste controlado)

Aviso antes de executar: **`class_weight`/`scale_pos_weight` resolvem o mesmo problema sem
jogar dado fora**, e a expectativa é que ganhem. Ainda assim, teste e reporte — com número, o
comitê decide.

Teste **dentro da CV** (o sorteio só pode acontecer no treino do fold, nunca no holdout) e
compare com a mesma tabela de métricas:

1. **Todos os positivos mantidos + amostragem do majoritário** em proporções fixas (1:5 e
   1:10), `random_state=42`.
2. **NearMiss-1** (seleção por proximidade das fronteiras) — é o que justifica a palavra
   "inteligente".
3. Controle: sem reamostragem, com peso.

Registre explicitamente que **undersampling altera a prevalência** e, portanto, a escala das
probabilidades: as métricas de ranking (PR AUC, ROC AUC, Resultado@k) continuam comparáveis,
mas a probabilidade crua **não** — se for usar o corte de 3,23%, ela precisa ser recalibrada.

---

## Parte 5 — Campeão, função custo 31:1 e explicabilidade

1. **Critério de campeão:** maior **Resultado Líquido@900** no holdout. PR AUC e Precisão@900
   como apoio e explicação. Nunca acurácia.
2. **Robustez de semente:** o campeão roda no holdout com **5 sementes (42, 7, 2026, 123,
   99)** e reporta média ± desvio. Na Fase 3 uma semente só nos contou uma história falsa; não
   repetimos isso.
3. **Função de custo 31:1, com a decomposição completa:**
   - Custo = 21.700 × FN + 700 × FP, e o custo evitado = 21.700 × TP.
   - **Corte teórico ótimo:** períciar quando p > 700/21.700 = **3,2258%** — o mesmo piso da
     Fase 1. Explique por que os dois números coincidem: é a mesma conta vista de dois lados.
   - **Corte prático:** capacidade fixa de 900 → ranquear e cortar em 900. Mostrar os dois e
     comparar o resultado.
4. **Curva `Resultado(k)`** para k de 0 até o tamanho do holdout, com a linha de **precisão
   marginal = 3,2258%** marcada. É o insumo das Perguntas 4 e 5 (k de menor custo, ganho além
   das 900, cenário de 1.800 perícias) — gere o artefato agora para a Fase 5 não refazer.
5. **Calibração:** se o campeão usar peso de classe, avalie `CalibratedClassifierCV`
   (isotonic/Platt) ajustado **dentro do fold** e reavalie Brier e o corte de 3,23% na escala
   calibrada. A probabilidade que o app exibe tem de significar o que diz.
6. **SHAP com o campeão, ponta a ponta** — é o `explicar_sinistro` que existe e **nunca foi
   executado**: execute-o e feche essa dívida. Use a agregação por feature original (one-hot e
   indicador de ausente voltam para a feature de origem, casamento pelo maior prefixo).
7. Entregue, com nome de negócio:
   - **SHAP summary** agregado por feature original;
   - **SHAP local de 3 sinistros reais**: um fraude pega, um fraude perdida, um legítimo no
     topo da fila — com a narrativa do porquê;
   - a **confirmação (ou não) do `fim_de_semana` como atributo morto** pela importância SHAP;
   - a **ablação de `uf` e `idade_segurado` com o modelo campeão** — os quatro números
     exigidos pela Pergunta 1 (ROC AUC e PR AUC com e sem as duas colunas), que na Fase 3
     saíram preliminares com modelo de apoio.
8. Nenhum número digitado à mão: tudo sai de célula executada.

---

## Parte 6 — Integração com o restante do projeto

1. `models/pipeline_fraude.joblib` passa a conter **o campeão** (substitui o placeholder),
   regravado por `salvar_pipeline`, com o `versoes.json` ao lado.
2. Novo módulo `src/quilha/modelagem.py` (busca, scorers, custo, campeão) e, se necessário,
   `src/quilha/mlops.py` — ambos com docstring em português e sem `fit` fora do treino.
3. `tests/test_modelagem.py` com, no mínimo: o scorer financeiro (conferência manual de
   21.700×TP − 700×k), o estimador da Regra Atual (nota monotônica em `valor_reclamado`, zero
   abaixo de 50 mil), o cálculo do corte p\* = 3,2258%, o `k_fold` proporcional e um teste de
   que o campeão salvo prevê igual ao campeão em memória.
4. Notebook da fase em `notebooks/clean/04_modelagem.py` (fonte, sem outputs) e
   `notebooks/executed/04_modelagem.ipynb` (executado, com outputs) — mesma mecânica da
   Fase 3 (`jupytext`).
5. `docs/04_modelagem.md` com a tabela comparativa, as curvas, a decisão de campeão e o porquê
   de cada parâmetro — em especial a explicação **didática** do 31:1, porque ela vai para o
   relatório do comitê.
6. `ruff check src tests` e `pytest` verdes ao final. (O CI deve ter escopo `src tests`: rodar
   ruff no repositório inteiro acusa os notebooks executados.)
7. **Idempotência e portabilidade**, porque este código vai virar o notebook consolidado que a
   equipe vai executar: nada de caminho absoluto, nada de estado global, tudo relativo à raiz
   encontrada por `encontrar_raiz`, e as células de MLflow funcionando **sem credencial**.

---

## Critérios de aceite

| # | Critério | Evidência exigida |
|---|---|---|
| 1 | Os 6 modelos rodaram pela mesma CV, com o Pipeline dentro do fold | Tabela com média ± desvio das 5 dobras |
| 2 | Acurácia não aparece como métrica de decisão | Ausência no notebook e no documento |
| 3 | Nenhum ajuste viu o holdout | `fit` só no treino; holdout lido uma vez, no fim |
| 4 | `k` proporcional na CV (~720) e 900 no holdout | Constantes nomeadas e impressas |
| 5 | Profundidades: curva da árvore e da floresta, com a escolha justificada | Gráfico + tabela + frase de decisão |
| 6 | `class_weight` e `scale_pos_weight` testados em 31 e 39,11 | Tabela das 4 combinações |
| 7 | UnderSample comparado com o controle | Tabela antes/depois, com o custo em informação declarado |
| 8 | Campeão escolhido por dinheiro, com 5 sementes | Resultado@900 média ± desvio |
| 9 | Função 31:1 decomposta e o corte de 3,2258% explicado | Conta explícita na célula e no documento |
| 10 | SHAP agregado por feature original, com 3 casos narrados | Gráficos + texto |
| 11 | Todos os runs no MLflow, com o Pipeline logado e o alias `champion` | IDs de run citados no documento |
| 12 | Notebook roda de ponta a ponta **sem credenciais de MLOps** | Execução limpa sem env vars |
| 13 | `ruff` e `pytest` verdes; artefato do app atualizado | Saída real dos comandos |

---

## Entregáveis

- `notebooks/clean/04_modelagem.py` e `notebooks/executed/04_modelagem.ipynb`
- `docs/04_modelagem.md`
- `src/quilha/modelagem.py`, `src/quilha/mlops.py`, `tests/test_modelagem.py`
- `models/pipeline_fraude.joblib` = campeão + `models/pipeline_fraude.versoes.json`
- Tabela comparativa final, curvas de profundidade, curva `Resultado(k)`, `shap_summary` e os
  3 SHAP locais em `reports/figures/`
- Runs no MLflow (DagsHub) com o modelo registrado e o alias `champion`

---

## Pergunta obrigatória a responder com número (não antes de medir)

> **Após o ajuste, o XGBoost, além de demonstrar as melhores métricas, é também o modelo que
> mais reduz a função de custo 31:1?**

Responda com: (a) a tabela de custo total e de custo evitado por modelo, decomposta em
**21.700 × FN + 700 × FP**; (b) o corte teórico de **3,2258%** de cada modelo na curva de
precisão; (c) o resultado no corte prático de 900 perícias; e (d) a conclusão — **inclusive se
a resposta for "não"**. Se outro modelo pagar menos, diga qual e por quê. Nada de afirmar antes
de medir, e nada de suavizar o resultado. Essa explicação entra no documento da fase **e** no
SHAP, porque é ela que sustenta a decisão diante do comitê.

---

## Modo de trabalho

Uma etapa por vez, com confirmação antes de avançar. Executar de verdade e colar a **saída
real**; se algo não rodou, dizer que não rodou. Nunca inventar número. Se existir caminho mais
sênior do que o pedido, avise e explique **antes** de executar.

---

## Progresso da execução

| Parte | Descrição | Status |
|---|---|---|
| 0 | Infraestrutura de MLOps (dependências + `mlops.py` + degradação sem credencial) | concluída |
| 1 | Modelos e baselines, mesmo protocolo | concluída |
| 2 | Protocolo de validação (Pipeline dentro do fold, `k` proporcional) | concluída |
| 3 | Hiperparâmetros e desbalanceamento (31 vs 39,11) | concluída |
| 4 | UnderSample inteligente vs controle | concluída |
| 5 | Campeão, função custo 31:1 e SHAP | concluída |
| 6 | Integração (artefato, testes, notebook, docs) | concluída |

**Evidência final (2026-11-07):** notebook executado sem credenciais, fallback local
em `mlflow.db`, Pipeline XGBoost salvo (449.791 bytes), paridade de predição antes/depois
da recarga (diferença máxima `0`), `ruff check src tests` verde e `50 passed` no pytest.
A CV histórica da Regra Atual foi revalidada em cinco folds de treino (`k=720`) e incluída
na tabela comparativa dos seis modelos.

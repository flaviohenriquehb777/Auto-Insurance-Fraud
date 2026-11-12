# FASE 6 — IMPLANTAÇÃO (CRISP-DM) · Prompt de execução

**Projeto:** Detecção de fraude em sinistros de automóvel — Quilha Seguros
**Fase:** 6 de 6 (CRISP-DM — Implantação; a fase que põe o resultado em uso)
**Status:** em andamento — **Parte 0 concluída** (Ramo A: o gerente confirmou a premissa
temporal das três preditoras; sem re-treino). Partes 1 a 5 pendentes, uma a uma, com autorização
a cada passo
**Pré-requisito:** Fase 5 **concluída e mergeada na `main`** (PR #2), com o parecer em
`docs/05_avaliacao.md`, o Quadro de Respostas preenchido (24 campos), o campeão em
`models/pipeline_fraude.joblib` e as tabelas `reports/tabelas/avaliacao_*.csv`.
**Fonte de verdade do negócio:** `docs/case/Auto Insurance Fraud.md` (Anexos 1 a 3, Glossário
e FAQ). Em caso de divergência com qualquer outro documento, **o case manda**.

---

## Como executar este documento

Para dar início a esta fase, basta pedir:

> **"Fase 6 — execute o prompt em `docs/prompts/06_implantacao.md`, parte por parte, começando
> pela Parte 0."**

O documento é autocontido de propósito: mesmo que o contexto da conversa se perca, ele traz os
números econômicos do negócio, as decisões já fechadas nas fases anteriores, as evidências
reais medidas na Fase 5 e **o desenho completo do app** acordado com o responsável do projeto.

**Partes:** 0 (reconciliação e a decisão pendente do gerente) · 1 (blindagem da fronteira — as
camadas 2 e 3 da vírgula) · 2 (o app de triagem) · 3 (o notebook consolidado único) · 4 (o
resumo executivo `.docx`) · 5 (verificação final e encerramento do projeto).

---

## Mandato desta fase (ler antes de qualquer célula)

As Fases 1 a 5 produziram um **modelo** (Fase 4) e uma **decisão de negócio** (Fase 5): a
política de perícia foi escolhida, precificada e defendida. Nada disso existe para a Quilha até
que alguém **use**. A Fase 6 é a fase em que o resultado sai do analista e vira **ferramenta de
trabalho** — e é a última fase do projeto.

Três princípios governam tudo o que for feito aqui:

1. **"O modelo ordena, a política decide, o perito confirma."** Esta frase não é retórica: é a
   regra de interface do app e o guarda-corpo jurídico da empresa. O app **prioriza**; ele nunca
   **acusa**. Periciar é decisão humana, e a fraude só existe depois que o perito a confirma.
2. **Nenhuma linha de código de produção pode produzir uma decisão silenciosamente errada.** O
   app é a fronteira por onde entra dado digitado por gente — é exatamente ali que a "falha
   silenciosa da vírgula" faria estrago. Por isso a Fase 6 fecha as camadas 2 e 3 antes de
   abrir a porta do app.
3. **O entregável do case precisa rodar sozinho, sem infraestrutura e sem credencial.** Quem
   abre o `.ipynb` é outra pessoa, em outra máquina, provavelmente sem token nenhum. O notebook
   consolidado é escrito para esse cenário desde a primeira célula.

**Critério de sucesso da fase (não reabrir):** a Quilha consegue operar a política recomendada
— o app abre, prioriza a fila do mês, explica cada sinistro e mostra o valor em reais; o
contrato de dados recusa payload inválido com erro explícito em vez de prever errado; e os
**dois entregáveis do case** (o `.ipynb` executado e o `.docx` de 1–2 páginas com as 5 respostas
e o Quadro) são produzidos, executam de ponta a ponta e conferem.

---

## Contexto herdado (não reabrir)

**Insumos econômicos e operacionais (Anexo 1 — valores fixos do case):**

| Parâmetro | Valor |
|---|---|
| Sinistros abertos por mês | ~12.000 |
| Capacidade de perícia | **900/mês** (30 peritos × 30 perícias) |
| Custo de uma perícia | **R$ 700** (perito + deslocamento + atraso do cliente honesto) |
| Custo de uma fraude paga sem perícia | **R$ 21.700** (média) |
| Regra atual | valor reclamado **> R$ 50 mil** → ~950 pedidos/mês, cortados em 900 por valor |
| Premissa | havendo fraude, a perícia **sempre** a encontra e o pagamento é negado |

**Piso de viabilidade:** `700 / 21.700 = 3,2258%`. `21.700 / 700 = 31` → um falso negativo
custa 31 falsos positivos.

**O que as Fases 1 a 5 deixaram pronto (números reais — não recalcular por padrão, auditar):**

| Evidência | Valor medido |
|---|---|
| Base completa (01/04–31/08/2026) | 60.000 sinistros · 1.480 fraudes · **2,47%** |
| Treino (abr–jul) | 48.096 sinistros · 1.199 fraudes · 2,49% |
| Teste (ago/2026) | 11.904 sinistros · 281 fraudes · 2,36% |
| Campeão | **XGBoost** `max_depth=4`, `n_estimators=350`, `lr=0,03`, `subsample=0,85`, `colsample_bytree=0,7`, `gamma=3`, `min_child_weight=5`, `reg_lambda=1`, `scale_pos_weight=1` |
| Resultado@900 (teste) | **R$ 3.731.700/mês** · `TP=201`, `FP=699`, `FN=80` · precisão 22,33% |
| PR AUC / ROC AUC (teste) | 0,538556 / 0,936597 |
| Regra atual (>R$ 50 mil), 900 perícias | 14 fraudes · **−R$ 326.200/mês** (perde até para o acaso) |
| Ganho da política sobre a regra atual | **R$ 4.057.900/mês** |
| Bootstrap da diferença (2.000, SEED=42) | R$ 4.090.114 · IC 95% **[R$ 3.537.100; R$ 4.665.500]** |
| Backtest walk-forward (4 janelas) | diferença média **R$ 4.329.150 ± 198.489** |
| Grade de sensibilidade (100 células) | **0 de 100** cenários perdem · break-even R$ 3.134,33 (razão 4,478:1) |
| Sensibilidade à taxa de fraude | equilíbrio em **0,3480%** (≈9,3× abaixo do piso) |
| `k` ótimo empírico | **1.430** perícias (~48 peritos) → R$ 3.968.300 · ganho além das 900 = R$ 236.600/mês |
| 1.800 perícias | 235 fraudes (múltiplo real **1,17×**, não 2×) |
| Acerto mínimo por perícia | **3,23%** (1/31) |
| Preço de não usar `uf` + `idade_segurado` | R$ 43.400/mês a favor da remoção (empate técnico) |
| Hash SHA-256 do artefato auditado | `65c7eb5e…` |
| SHAP (300 sinistros) — topo | `boletim_ocorrencia` 0,014021 · `tipo_sinistro` 0,012628 · `razao_reclamado_fipe` 0,011226 · `dias_de_apolice` 0,005010 · `tem_terceiro_identificado` 0,004004 · `fim_de_semana` 0,000058 (18º de 18) |

**Achado de negócio que o app precisa respeitar (Pergunta 2):** a fraude **se concentra abaixo**
do limiar de R$ 50 mil. A faixa de R$ 40–50 mil tem 2.231 sinistros com **25,15% de fraude**;
acima de R$ 50 mil a taxa desaba. Foi por isso que a regra de valor perde dinheiro — e é por isso
que a fila do app é ordenada por **escore**, não por valor.

**Decisões já fechadas que este prompt não reabre:**

- `SEED = 42` em tudo que tem aleatoriedade. **Acurácia vetada** como métrica. **Sem SMOTE.**
- `valor_indenizado`, `id_sinistro`, `data_ocorrencia`, `data_abertura`, `hora_ocorrencia` e
  `fraude` **nunca** entram no modelo. Toda feature existe **no momento da abertura**.
- As **18 features** do modelo, na ordem canônica de `quilha.preparacao.FEATURES`.
- O `R$ 700` **já inclui** os sete dias de atraso — não contar duas vezes.
- A porta única de entrada é `pipeline.predict` / `pipeline.predict_proba` sobre o **chamado
  cru**: o `preparo` (com o parser de vírgula) mora **dentro** do Pipeline. Nada converte número
  fora dele.
- O `boletim_ocorrencia` em branco vira a categoria própria **"Não informado"** — nunca
  imputado.
- MLOps **degrada graciosamente**: sem credencial cai para SQLite local; o notebook **nunca**
  pode quebrar por causa do tracking.
- Os entregáveis do case são **dois arquivos soltos, sem `.zip`**: o arquivo de análise
  (`.ipynb` **executado**, com saídas visíveis) e o resumo executivo (`.docx`/PDF, 1–2 páginas +
  o Quadro de Respostas). **Não** é preciso enviar a base.

---

## Perguntas do comitê que esta fase encerra

| Pergunta | O que falta após a Fase 5 | Fase que encerra |
|---|---|---|
| **P1 a P5** | Consolidar, em **um** arquivo de análise executado, o percurso completo e os números que sustentam cada resposta | **Fase 6** |
| Acionamento (P4/P5) | Transformar a política em **ferramenta usável** pela triagem, com valor em reais visível | **Fase 6** |

A Fase 5 respondeu. A Fase 6 **entrega e prova** — e o projeto só termina quando o `.ipynb` e o
`.docx` estiverem prontos, enxutos e verificados.

---

## Regras invioláveis (guardrails)

1. **Nada construído sobre número não verificado.** A Parte 0 reaudita o artefato e decide a
   configuração final **antes** de qualquer linha do app.
2. **O app não emite veredito de fraude.** Nenhum rótulo, cor ou texto na tela pode afirmar que
   um sinistro **é** fraude. A precisão é 22,33%: 78% da fila de 900 é honesta e 80 fraudes
   passam — um "FRAUDE" na tela cria passivo jurídico (LGPD/decisão automatizada) e reputacional.
3. **Toda cor é reetiquetada por prioridade, não por culpa** (ver Parte 2).
4. **Fronteira única e validada.** Todo dado entra por um único caminho, que **valida o
   payload** e **recusa com erro explícito** o que não estiver no contrato. Falha silenciosa é
   proibida.
5. **O notebook consolidado roda sem credencial, sem infraestrutura e sem a base em caminho
   fixo.** A primeira célula descobre a raiz ou orienta o usuário; jamais quebra por `cwd`.
6. **Um número, uma fonte.** Todo número do `.docx` aparece **impresso** em alguma célula do
   `.ipynb` (o critério do case é conferir o resumo nas saídas do arquivo).
7. **Nenhum número bonito sem contraprova** e **nenhuma afirmação sem incerteza** — herdado.
8. **Um app, um artefato.** O app carrega **exatamente** `models/pipeline_fraude.joblib` do
   hash auditado. Nada de re-treinar no app.
9. **Uma etapa por vez.** Nunca avançar sozinho. Nenhuma decisão sem evidência.
10. **Nada de reinventar o que existe.** `preparacao`, `explicacao`, `mlops`, `pipeline` e
    `avaliacao` já têm as funções. Esta fase **usa**, não duplica.

---

## Parte 0 — Reconciliação e a decisão pendente (resolver ANTES de tudo)

Esta parte existe por causa de **uma ressalva deixada em aberto pela auditoria** e é a **única**
coisa que pode mudar a configuração final do modelo. Ela roda primeiro e pode ter dois finais.

**Contexto da pendência (ressalva R1 da auditoria, severidade média):** três preditoras têm
premissa temporal que precisa ser confirmada pelo negócio — se estavam **preenchidas com a
informação definitiva no momento exato da triagem**:

- `dias_de_apolice` (3ª preditora mais forte no SHAP uni e multivariado);
- `boletim_ocorrencia` (a **nº 1** no SHAP e a única coluna com branco — 4,83%);
- `terceiro_identificado` (origem da feature `tem_terceiro_identificado`).

A pergunta já foi formulada ao responsável do projeto, com a tabela de leitura da resposta.

**O que fazer na Parte 0:**

1. **Reauditar o artefato** (independe da resposta): `carregar_pipeline(CAMINHO_PIPELINE)`;
   conferir tipo do estimador, hiperparâmetros, as **18 features** e a ordem; registrar
   **SHA-256** do `.joblib` e conferir contra `65c7eb5e…`; reproduzir o **teste de fumaça**
   `Resultado@900 = R$ 3.731.700` (`TP=201`, `FP=699`, `FN=80`). Se não reproduzir, **parar**.
2. **Registrar a resposta do gerente** em célula, com data e a literalidade do que foi dito, e
   **ramificar**:
   - **Ramo A — premissa temporal OK** (os campos são preenchidos na abertura, com a informação
     definitiva): o campeão **segue como está**. Não há re-treino. A Parte 0 apenas documenta a
     resposta e passa adiante.
   - **Ramo B — premissa quebrada** (algum campo é preenchido *depois* da decisão de perícia):
     **medir antes de mexer.** Ajustar o campeão **sem** a(s) preditora(s) suspeita(s), com o
     protocolo inteiro (Pipeline no fold, `SEED=42`), e reportar lado a lado:
     `Resultado@900`, PR AUC e ROC AUC **com** e **sem**. Só então decidir, com o número na mão:
     - se a perda for **material** (fora da banda do bootstrap), **remover** e re-treinar;
     - se for dentro do ruído, **manter** e registrar a decisão com a justificativa.
     **A medição do custo de remover `dias_de_apolice` (e/ou `boletim_ocorrencia`) é obrigatória
     neste ramo — é o número que a auditoria deixou em aberto.**
   - **Ramo C — resposta ambígua ou ausente:** **não construir o app** sobre a premissa
     duvidosa. Registrar o risco, seguir com o restante da fase que não depende dela (Parte 1 e
     a estrutura do notebook/docx) e deixar a Parte 2 bloqueada até a resposta.

3. **Travar a configuração final** em uma frase: *"O modelo implantado é o XGBoost do hash
   `<hash>`, com as features `<lista>`, porque `<evidência>`."* Essa frase é a primeira linha do
   documento da fase.

**Critério de saída:** hash conferido, teste de fumaça reproduzido e a configuração final
travada — com o ramo (A, B ou C) explicitamente declarado.

---

## Parte 1 — Blindagem da fronteira (as camadas 2 e 3 da vírgula)

A "falha silenciosa" é o risco que mais mata projeto em produção: `valor_reclamado` chega como
texto pt-BR (`"8837,07"`) e uma conversão descuidada devolve `NaN` **sem exceção**; o pipeline
imputa a mediana e devolve uma predição **normal com o valor errado**. A camada 1 (porta única)
já existe — tudo passa por `pipeline.predict`. Faltam as camadas 2 e 3.

**Bateria adversarial de referência (já medida, reproduzir para provar a correção):**

| Entrada em `_numero` | Resultado esperado |
|---|---|
| `"8837,07"` | `8837.07` (correto) |
| `"8837.07"` | `8837.07` (correto) |
| `"21.700,00"` | `<NA>` (milhar pt-BR **não** é tratado — decisão do projeto) |
| `"R$ 100,00"` | `<NA>` |
| `"1.234"` | `1.234` (lido como 1,234 — risco de 1.000×) |
| `""` / `None` | `<NA>` |

**O que construir:**

1. **Camada 2 — contrato na fronteira (`src/quilha/contrato.py`, módulo novo).**
   - `CAMPOS_ABERTURA` = as **14 colunas de abertura** que o app recebe:
     `data_ocorrencia`, `hora_ocorrencia`, `data_abertura`, `canal_abertura`, `tipo_sinistro`,
     `uf`, `idade_segurado`, `dias_de_apolice`, `sinistros_12m`, `idade_veiculo`, `valor_fipe`,
     `valor_reclamado`, `boletim_ocorrencia`, `terceiro_identificado`.
   - `validar_payload(df) -> list[ErroContrato]`: verifica presença das 14 colunas, domínios
     categóricos válidos e **coerção numérica explícita** (usa o mesmo critério de
     `preparacao._numero`). Erro = recusa **com mensagem dizendo o campo, o valor e o motivo** —
     nunca uma predição sobre dado inválido.
   - `ErroContrato` com `campo`, `valor_recebido`, `motivo`.
   - **Regra de ouro da camada 2:** se o número não converte, o payload é **recusado**; o app
     mostra o erro e pede correção. Nada de imputar na fronteira.
2. **Camada 3 — alarme no silêncio (teste que trava + monitor).**
   - **Teste de contrato** em `tests/test_contrato.py`: uma linha por caso da bateria adversarial
     acima, provando o **comportamento atual** e o **esperado**; o teste **trava o contrato**
     — se alguém afrouxar o parser, o CI fica vermelho.
   - **Monitor de imputação** em `src/quilha/monitoramento.py` (ou função no contrato):
     `resumo_imputacoes(pipeline, df)` devolve, por feature, a fração de valores imputados
     (nulos + indicadores `missingindicator_` quando `add_indicator=True`). O app expõe isso na
     tela de governança (Parte 2). **Gatilho de alerta:** se a fração imputada de
     `valor_reclamado` passar de um limiar (ex.: 5%), a fila do mês é sinalizada como suspeita
     de problema de dado — não de fraude.
3. **Prova executada.** Rodar a bateria adversarial e **colar a saída real** mostrando, linha a
   linha, o que entra, o que sai e o veredito (aceito/recusado com motivo).

**Critério de saída:** um payload com `"21.700,00"` ou `"R$ 100,00"` em `valor_reclamado` é
**recusado com erro explícito**; um payload com `"8837,07"` é aceito e prevê corretamente; o
teste de contrato está verde; o monitor de imputação imprime a fração por feature.

---

## Parte 2 — O app de triagem (Streamlit)

O app é a interface entre a política e as pessoas da triagem. Ele **carrega o artefato auditado**
e **não decide fraude** — prioriza, explica e mostra o valor.

**Stack e entrada:**

- **Streamlit** (novo; entra no grupo de dependências de aplicação do `pyproject.toml`).
- Carrega `models/pipeline_fraude.joblib` via `carregar_pipeline(...)`. **A vírgula é resolvida
  dentro do `preparo`** que já mora no Pipeline — o app **não** converte número por fora.
- Valida tudo pela **camada 2** (Parte 1) antes de prever.
- SHAP local via a função existente `explicacao.explicar_sinistro(pipeline, linha, fundo)`.

**Três telas:**

1. **Fila do mês.** Seleciona um lote/arquivo de sinistros abertos, aplica o contrato, pontua
   com `predict_proba`, **ordena por escore** e **corta em k=900** (k ajustável e explicado:
   `k = 900 × n_mês / 12.024`). Tabela com posição, escore, valor reclamado e faixa. Botão
   **Exportar fila (CSV)** para o gestor da perícia. No topo, um resumo: *"Nesta fila, as 201
   primeiras posições concentraram as fraudes do mês de teste"* — em linguagem de fila, não de
   acusação.
2. **Sinistro individual.** Formulário com os **14 campos de abertura** (ou seleção de um
   sinistro da fila). Ao confirmar, exibe:
   - **Escore** e **posição na fila** (ex.: *"posição #37 de 12.024"*);
   - **Faixa de prioridade** (ver semáforo abaixo);
   - **"O que fez o modelo priorizar este sinistro"** — gráfico de barras SHAP das features
     originais (usa `explicar_sinistro`). **Rótulo obrigatório**: nunca "evidências de fraude".
   - **Governança:** idade e UF permanecem no escore, mas **não** são destacadas como "razões";
     a tela diz que o modelo usa atributos demográficos e que a seleção é auditada.
3. **Valor e governança.** Painel para a diretoria:
   - **Valor:** R$ 3.731.700/mês (política) × **−R$ 326.200/mês** (regra atual) → ganho
     **+R$ 4.057.900/mês**, com o **IC 95% [R$ 3.537.100; R$ 4.665.500]** ao lado.
   - **Por sinistro:** valor esperado = `chance × 21.700 − 700` (calibração bruta mantida),
     exibido como **faixa** + taxa observada quando a calibração não for confiável.
   - **Saúde do modelo:** data/hora, **hash do artefato**, versão, contagem de nulos e
     **fração de imputações** (camada 3), drift das features principais.
   - **Circuit breaker:** condição de retorno à regra anterior, à vista (ex.: Precisão@900 abaixo
     de 3,23% por dois meses, ou taxa de fraude do fluxo abaixo de 0,348%).

**Regras de cor e linguagem (não negociáveis):**

- **Semáforo reetiquetado por prioridade**, nunca por culpa:
  - 🔴 **"Periciar — alta prioridade"**
  - 🟡 **"Periciar se houver folga"**
  - 🟢 **"Sem prioridade de perícia"**
- **Mensagem permanente** na tela: *"O modelo ordena, a política decide, o perito confirma."*
- Nenhum texto, etiqueta ou cor diz "fraude" para um sinistro individual.

**Testes do app:** `tests/test_app.py` (nível de função, sem subir UI): ordenação da fila
respeita o escore; corte em k respeita a capacidade; a rota de pontuação passa pelo contrato;
um payload inválido é recusado antes de chegar ao modelo.

**Critério de saída:** o app sobe local, prioriza uma fila de exemplo, mostra escore + SHAP +
faixa para um sinistro individual e exibe o painel de valor com os números reais — **sem**
afirmar fraude em nenhum ponto.

---

## Parte 3 — O notebook consolidado único (entregável do case)

Este é o **entregável de maior peso** (10 pontos possíveis no case). É um único `.ipynb`
**executado**, com saídas visíveis, que conta o projeto inteiro do começo ao fim e prova cada
resposta.

**Requisitos de robustez (é um arquivo que outra pessoa vai rodar):**

1. **Célula 1 tolerante ao caminho.** Não assumir `cwd`. Tentar `encontrar_raiz()`; se falhar
   (arquivo solto), exibir instrução clara *"coloque o notebook na raiz do projeto junto de
   `pyproject.toml`"* e **seguir sem quebrar** com uma base de demonstração/paths resolvidos.
2. **MLOps incluído, mas jamais obrigatório.** Envolver o tracking em `contexto_run`; sem
   credencial, degrada para SQLite (`sqlite:///mlflow.db`); `registrar_modelo` captura a falha.
   **O notebook tem de rodar inteiro sem token nenhum.**
3. **Reexecução determinística.** Fixar `random_state=42` em toda parte; rodar de ponta a ponta
   antes de copiar qualquer número.
4. **Todos os 24 números do Quadro impressos.** Cada campo do Quadro de Respostas tem de
   aparecer em um `print` do notebook (o case confere o resumo nas saídas).
5. **Rastreio das 5 perguntas.** Ao lado de cada resultado, dizer a qual das 5 perguntas ele
   responde.

**Estrutura do notebook (Fases 1 a 6):**

| Seção | Conteúdo |
|---|---|
| 0. Setup | semente, versões, caminho tolerante, import das funções do projeto |
| 1. Negócio | parâmetros do Anexo 1, piso 3,2258%, critério de sucesso |
| 2. Dados e colunas | leitura crua, tipos, decisão de colunas (**P1**) — e o porquê de `valor_indenizado` sair |
| 3. Análise descritiva | taxa por tipo, canal, boletim, faixa de apólice, faixa de valor (**P2**) |
| 4. Modelagem | comparação de modelos, escolha do campeão, ROC/PR (**P3**) |
| 5. Avaliação | política, economia, dimensionamento, sensibilidade, IC (**P4/P5**) |
| 6. Implantação | contrato, app (código + capturas), monitoramento |
| 7. Quadro | os 24 campos, impressos e preenchidos |

**Notebook-fonte:** `notebooks/clean/06_consolidado.py` (via `jupytext`, sem outputs) e
**executado** `notebooks/executed/06_consolidado.ipynb`.

**Critério de saída:** o `.ipynb` roda de ponta a ponta **sem credencial**, imprime os 24 números
do Quadro e o **tempo real de execução** é **medido e registrado** (não estimado).

---

## Parte 4 — O resumo executivo (`.docx`)

O segundo entregável do case: **1–2 páginas**, dividido em **cinco partes** (Pergunta 1 a
Pergunta 5), cada parte com a resposta **e os números que a sustentam**, seguidas do **Quadro de
Respostas preenchido** (os 24 campos, fora do limite de 2 páginas). Formatos aceitos: PDF ou
DOCX — vamos de **DOCX**.

1. **Adicionar `python-docx`** ao grupo `dev` do `pyproject.toml` (não está declarado).
2. **Gerar o `.docx` a partir dos números medidos** — script em `src/quilha/relatorio.py` ou
   célula dedicada, lendo o **mesmo** `reports/tabelas/avaliacao_quadro_respostas.csv` da Fase 5
   para o Quadro (uma fonte só; CSV e documento não podem divergir). Saída na **raiz**:
   `Relatorio_Quilha_Fraude.docx`.
3. **Conteúdo de cada parte (com o número):** P1 (18 features, colunas fora, idade/UF com
   ROC/PR com e sem); P2 (taxas por grupo, com destaque para a faixa 40–50 mil a 25,15%); P3
   (modelos comparados, XGBoost, ROC/PR de treino e teste); P4 (política, 201 fraudes, R$
   3.731.700 × −R$ 326.200, ganho R$ 4.057.900, ação sobre a regra atual); P5 (acerto mínimo
   3,23%, k=1.430, 235 fraudes a 1.800, ação sobre a equipe).
4. **Regra de ouro:** **todo número do `.docx` tem de existir impresso no `.ipynb`**. Conferir
   campo a campo antes de fechar.

**Critério de saída:** `.docx` gerado, com as 5 partes + Quadro, números conferidos contra as
saídas do `.ipynb`, e `python-docx` no `pyproject.toml`.

---

## Parte 5 — Verificação final e encerramento do projeto

1. **Verificação de integridade:** `ruff check src tests` e `pytest -q` **verdes**, com a saída
   real colada. Conferir que os testes novos (contrato, app, monitor) entram na contagem.
2. **Verificação dos entregáveis:** abrir o `.ipynb` executado e conferir — 24 linhas do Quadro
   presentes, nenhuma célula com erro, MLOps degradado sem quebrar; abrir o `.docx` e conferir as
   5 partes + o Quadro.
3. **Documento `docs/06_implantacao.md`:** decisão da fase (com a configuração final travada na
   Parte 0), as camadas 2 e 3, o app (telas e regras de cor), os entregáveis e a tabela de
   critérios de aceite com o estado real de cada um.
4. **Atualizar o README** com como subir o app e onde estão os dois entregáveis do case.
5. **Encerramento:** atualizar a tabela **Progresso da execução** deste prompt; escrever a
   **frase de encerramento do projeto** (o que foi entregue, o valor em reais, a condição que
   invalidaria a recomendação); e **sinalizar ao responsável** o ponto de **commit/push** e o
   **merge do PR da Fase 6** (e do PR #2, se ainda pendente) na `main`.

---

## Critérios de aceite

| # | Critério | Evidência exigida |
|---|---|---|
| 1 | Artefato reauditado por hash e reproduz R$ 3.731.700 | Saída com hash, TP/FP/FN |
| 2 | A decisão pendente foi resolvida (ramo A/B/C) e a configuração final está travada por escrito; no ramo B, o custo de remover a preditora suspeita foi **medido** | Bloco com a resposta e o número |
| 3 | Contrato na fronteira: payload inválido é **recusado com erro explícito** | Bateria adversarial com saída real |
| 4 | Teste de contrato (camada 3) verde, travando o parser | Saída do `pytest` |
| 5 | Monitor de imputação imprime a fração por feature | Saída real |
| 6 | App sobe, ordena a fila, corta em k e mostra escore + SHAP + faixa | Execução local + capturas |
| 7 | App **não** emite veredito de fraude; semáforo reetiquetado por prioridade; mensagem permanente | Revisão de telas |
| 8 | Painel de valor com R$ 3.731.700 × −R$ 326.200 e ganho R$ 4.057.900 (IC ao lado) | Tela + saída |
| 9 | `.ipynb` único roda de ponta a ponta **sem credencial**; tempo real medido | Execução limpa + tempo |
| 10 | Os **24 números** do Quadro aparecem impressos no notebook | Conferência campo a campo |
| 11 | `.docx` com as 5 partes + Quadro, números conferidos contra o notebook | Documento gerado |
| 12 | `ruff` e `pytest` verdes com os testes novos | Saída real dos comandos |
| 13 | `docs/06_implantacao.md` e README atualizados; Progresso da execução completo | Documentos |

---

## Entregáveis

- `docs/prompts/06_implantacao.md` (este documento) e `docs/06_implantacao.md` (parecer da fase)
- `src/quilha/contrato.py`, `src/quilha/monitoramento.py` e `src/quilha/relatorio.py`
- App Streamlit (arquivo do app + `tests/test_app.py`) e `streamlit` no `pyproject.toml`
- `tests/test_contrato.py`
- `notebooks/clean/06_consolidado.py` e `notebooks/executed/06_consolidado.ipynb`
- `Relatorio_Quilha_Fraude.docx` (na raiz) e `python-docx` no `pyproject.toml`
- README atualizado e tabela **Progresso da execução** deste prompt completa

---

## Pergunta obrigatória a responder com número (não antes de medir)

> **Após a resposta do gerente e após construir o app: a política continua sendo a recomendada
> (configuração travada na Parte 0), o app a entrega sem nunca afirmar fraude, e os dois
> entregáveis do case provam, em arquivo executado, os mesmos números que sustentam as cinco
> respostas da Diretoria?**

Responda com: **(a)** a configuração final e o ramo que a decidiu, com o número que o sustenta;
**(b)** a prova de que o contrato recusa payload inválido (saída real); **(c)** a tela de valor
com o ganho em reais e o IC; **(d)** a conferência dos 24 campos do Quadro contra o `.ipynb`
executado — **inclusive se algo não fechar**. Nada de afirmar antes de medir.

---

## Modo de trabalho

Uma etapa por vez, com confirmação antes de avançar. Executar de verdade e colar a **saída
real**; se algo não rodou, dizer que não rodou. Nunca inventar número. Se existir caminho mais
sênior do que o pedido, avisar e explicar **antes** de executar. Se o pedido piorar a qualidade
técnica, avisar em duas frases e fazer mesmo assim. Encerrar cada resposta com **"Aguardando o
próximo passo."**

---

## Progresso da execução

| Parte | Descrição | Status |
|---|---|---|
| 0 | Reconciliação e decisão pendente (ramo do gerente) | **concluída** — Ramo A (sem re-treino) |
| 1 | Blindagem da fronteira (camadas 2 e 3 da vírgula) | **concluída** — contrato + monitor, 49 testes |
| 2 | O app de triagem (Streamlit) | **concluída** — 3 telas, 20 testes, 126 na suíte |
| 3 | O notebook consolidado único | **concluída** — `06_consolidado.ipynb`, 30 células, 0 erros, 24/24 campos do Quadro, 42,27 s |
| 4 | O resumo executivo `.docx` | pendente |
| 5 | Verificação final e encerramento | pendente |

---

## Apêndice — como reaproveitar este prompt em outro projeto

- **Genérico (copiar como está):** o mandato de "entregar o resultado em uso, não só medido"; o
  princípio de fronteira validada que **recusa dado inválido em vez de prever errado**; o teste
  que **trava o contrato**; o monitor de imputações; o app que **prioriza e explica** em vez de
  **julgar**; a exigência de entregável único que **roda sem infraestrutura nem credencial**; e
  a regra de que **todo número do resumo existe impresso no arquivo de análise**.
- **Específico (trocar):** os parâmetros do Anexo 1, a política e o `k`, os nomes de módulos,
  arquivos e colunas, e o rastreio das perguntas do comitê.

Regra de ouro reaproveitável: **fronteira que recusa, app que prioriza, entregável que roda
sozinho e número que aparece na saída.**

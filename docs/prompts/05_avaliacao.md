# FASE 5 — AVALIAÇÃO (CRISP-DM) · Prompt de execução

**Projeto:** Detecção de fraude em sinistros de automóvel — Quilha Seguros
**Fase:** 5 de 6 (CRISP-DM — Avaliação; a "validação" do resultado de negócio)
**Status:** concluída (Partes 0 a 8)
**Pré-requisito:** Fase 4 concluída e mergeada na `main` (PR #1, commit de merge `2ecccf4`),
com evidências em `docs/04_modelagem.md`, campeão em `models/pipeline_fraude.joblib` e a
curva `Resultado(k)` em `reports/tabelas/parte5_curva_resultado_k.csv`.
**Fonte de verdade do negócio:** `docs/case/Auto Insurance Fraud.md` (Anexos 1 a 3, Glossário
e FAQ). Em caso de divergência com qualquer outro documento, **o case manda**.

---

## Como executar este documento

Para dar início a esta fase, basta pedir:

> **"Fase 5 — execute o prompt em `docs/prompts/05_avaliacao.md`, parte por parte, começando pela Parte 0."**

O documento é autocontido de propósito: mesmo que o contexto da conversa se perca, ele traz os
números econômicos do negócio, as decisões já fechadas nas fases anteriores e as evidências
reais medidas na Fase 4.

**Partes:** 0 (reconciliação com o case) · 1 (auditoria do que se avalia) · 2 (estimativa
honesta fora da amostra) · 3 (política de perícia e economia — Pergunta 4) · 4
(dimensionamento da equipe e valor marginal — Pergunta 5) · 5 (sensibilidade, break-even e
incerteza) · 6 (aceitabilidade, risco e explicabilidade) · 7 (revisão do processo e próximos
passos) · 8 (consolidação, testes e entregáveis).

---

## Mandato desta fase (ler antes de qualquer célula)

A Fase 4 terminou uma escolha de **modelo**. A Fase 5 produz uma **decisão de negócio**.

Nesta fase o projeto deixa de ser um exercício de modelagem e passa a ser uma
**recomendação defensável ao comitê da Quilha Seguros**: qual política de perícia adotar, com
qual capacidade, a que custo, sob quais condições e com qual plano de monitoramento.

Três princípios governam tudo o que for feito aqui:

1. **O modelo não decide; ele ordena.** A decisão é a **política**: `ordenar por escore →
   cortar em k → periciar`. Os únicos parâmetros livres da política são **o escore** e **o k**
   (a capacidade). Todo o resto é restrição de negócio, não escolha do analista.
2. **A capacidade é a restrição, não o limiar.** Periciar "tudo acima de 3,2258%" e periciar
   "os 900 de maior escore" são coisas diferentes e devem ser medidas separadamente
   (a curva `Resultado(k)` da Fase 4 já separa as duas).
3. **A recomendação é o resultado líquido, não a fraude pega.** Maximizar
   `Resultado Líquido = 21.700 × TP − 700 × k`. Fraude pega é meio, nunca fim. E o critério
   de sucesso herdado da Fase 1 é **superar a regra atual**, não apenas ficar positivo.

**Critério de sucesso (herdado da Fase 1, não reabrir):**

- **Negócio:** `Resultado(900)` no teste **>** `Resultado` da regra atual com as mesmas 900
  perícias **e** `> 0`.
- **Complementar:** ganho mensal positivo ao elevar de 900 para 1.800 perícias — valida ou
  refuta a tese do Otávio ("o dobro de peritos pega o dobro de fraude").
- **Técnico:** PR AUC no teste acima da taxa base da fraude, com ganho material e sem
  evidência de sobreajuste.

---

## Contexto herdado (não reabrir)

**Insumos econômicos e operacionais (Anexo 1 — valores fixos do case):**

| Parâmetro | Valor |
|---|---|
| Sinistros abertos por mês | ~12.000 |
| Capacidade de perícia | **900/mês** (30 peritos × 30 perícias) |
| Custo de uma perícia | **R$ 700** (perito + deslocamento + atraso no pagamento) |
| Custo de uma fraude paga sem perícia | **R$ 21.700** (média) |
| Regra atual | valor reclamado **> R$ 50 mil** → ~950 pedidos/mês, cortados em 900 **por ordem de valor** |
| Premissa | havendo fraude, a perícia **sempre** a encontra e o pagamento é negado |

`21.700 / 700 = 31` → um falso negativo custa 31 falsos positivos.
**Piso de viabilidade:** uma perícia só se paga se `p ≥ 700/21.700 = 3,2258%`.

**Base e recortes (medidos nas Fases 2 e 4):**

| Recorte | Sinistros | Fraudes | Taxa |
|---|---:|---:|---:|
| Base completa (01/04 a 31/08/2026) | 60.000 | 1.480 | **2,47%** |
| Treino (abr–jul) | 48.096 | 1.199 | 2,49% |
| Teste (ago) | 11.904 | 281 | 2,36% |

**O que a Fase 4 deixou pronto (números reais — não recalcular por padrão, auditar):**

- **Campeão:** XGBoost sem reamostragem, `scale_pos_weight=1`; `max_depth=4`,
  `n_estimators=350`, `learning_rate=0,03`, `subsample=0,85`, `colsample_bytree=0,7`,
  `gamma=3`, `min_child_weight=5`, `reg_lambda=1`.
- **Holdout de agosto, corte de 900:** Resultado Líquido **R$ 3.731.700**; `TP=201`, `FP=699`,
  `FN=80`; PR AUC **0,5386**; ROC AUC **0,9366**; Precisão@900 ≈ 0,2233; Cobertura@900 ≈ 0,7153.
- **Robustez de 5 sementes:** R$ 3.731.700 ± R$ 15.344 (faixa R$ 3.710.000–R$ 3.753.400).
- **Curva `Resultado(k)`:** ótimo empírico `k=1.430` → R$ 3.968.300; corte teórico
  `k=1.660` → R$ 3.894.100; `k=900` → R$ 3.731.700; `k` máximo ainda lucrativo = **8.710**.
- **Custo sem modelo nenhum:** `21.700 × 281 = R$ 6.097.700` no mês de teste.
- **Calibração (Brier):** bruto **0,014702**; isotonic 0,014688; sigmoid 0,015210 → bruto mantido.
- **Ablação de `uf` + `idade_segurado`:** ROC AUC 0,936597 → 0,937166 (**+0,000569** ao
  remover); PR AUC 0,538556 → 0,533792 (**−0,004764** ao remover).
- **SHAP (300 sinistros):** `boletim_ocorrencia` 0,014021 · `tipo_sinistro` 0,012628 ·
  `razao_reclamado_fipe` 0,011226 · `dias_de_apolice` 0,005010 · `tem_terceiro_identificado`
  0,004004 · ... · `fim_de_semana` 0,000058 (**18º de 18**).
- **Baselines na CV (`k=720`):** Dummy prior PR AUC 0,0249 / ROC AUC 0,5000 / Resultado
  −R$ 96.040; Regra Atual PR AUC 0,0245 / ROC AUC 0,4877 / Resultado −R$ 234.920.

**Limite de validação herdado (esta fase existe para resolver isto):** agosto foi o único
holdout disponível **e** participou da escolha do campeão. Portanto o R$ 3.731.700 é
**evidência de seleção, não estimativa externa intocada**. A Fase 5 tem de produzir uma
estimativa honesta fora da amostra (Parte 2) antes de recomendar qualquer coisa.

**Decisões já fechadas que este prompt não reabre:**

- `SEED = 42` em tudo que tem aleatoriedade; sementes de robustez `(42, 7, 2026, 123, 99)`.
- **Acurácia vetada** como métrica de decisão. **Sem SMOTE.**
- Corte **cronológico** do holdout, sem `shuffle` e sem `stratify`. O Pipeline inteiro entra em
  cada fold da CV.
- `valor_indenizado`, `id_sinistro`, `data_ocorrencia`, `data_abertura` e `fraude` **nunca**
  entram como atributo. Toda feature existe **no momento da abertura**.
- O `R$ 700` **já inclui** os sete dias de atraso do cliente honesto — **não contar duas vezes**.
- Usar os valores do Anexo 1 mesmo que outro valor pareça mais realista.
- `models/pipeline_fraude.versoes.json` não é versionado; o `.joblib` é.
- Os entregáveis do case (notebook consolidado + resumo executivo) ficam para o **fim do
  projeto**, na raiz. Esta fase consolida **os números e o Quadro de Respostas**, não o PDF/DOCX.

---

## Perguntas do comitê que esta fase responde

Rastreio definido na Fase 1 (`docs/01_entendimento_negocio.md`, seção 10):

| Pergunta | O que é exigido | Fase que entrega |
|---|---|---|
| **P4** | Política recomendada, perícias/mês, fraudes encontradas, economia no mês (política **e** regra atual com 900), ganho da política sobre a regra atual, **ação sobre a regra atual** | **Fase 5** |
| **P5** | Acerto mínimo para uma perícia se pagar, perícias/mês de menor custo, ganho mensal além das 900, fraudes com 1.800 perícias/mês, **ação sobre a equipe** | **Fase 5** |
| P1, P2, P3 | Já respondidas — esta fase apenas **consolida os números** no Quadro | Fases 2 a 4 |

Duas perguntas do comitê pedem **ação**, não medida: "o que fazer com a regra atual" e "o que
fazer com a equipe". Esta fase só termina quando essas duas ações estiverem escritas, com o
número que as sustenta ao lado.

---

## Regras invioláveis (guardrails)

1. **Nada digitado à mão.** Todo número citado vem de célula executada, com saída colada.
2. **Nenhuma reabertura de tuning.** O campeão está decidido: esta fase **carrega
   `models/pipeline_fraude.joblib`** e o usa. Ajustar qualquer coisa e depois avaliar é
   vazamento. Se um novo ajuste for indispensável, ele roda **só no treino** e o resultado
   honesto vem do backtest (Parte 2), nunca do holdout reaproveitado.
3. **Holdout lido uma vez por recorte.** O backtest usa janelas cronológicas; cada janela
   avaliada é posterior ao seu treino. Sem embaralhar o tempo.
4. **Acurácia não aparece** como critério em nenhuma decisão.
5. **Toda diferença só é afirmada com incerteza.** Se a diferença política × regra atual for
   menor que o erro de amostragem (intervalo de confiança), isso é **ruído** e deve ser dito.
6. **Nenhuma recomendação fura a capacidade** sem que a decisão de ampliar a equipe seja
   explícita e quantificada (Parte 4). A capacidade é restrição de negócio.
7. **Nenhum número bonito é aceito sem contraprova.** Se um resultado parecer bom demais,
   perguntar de onde vem a informação — exatamente como o case exige em "Considerações".
8. **Comparar sempre com três referências, nunca com uma:** (a) não periciar ninguém;
   (b) a **regra atual** com 900; (c) a **referência simples** (DummyClassifier, mesma nota
   para todos → equivale a escolher 900 ao acaso).
9. **Explicitar a unidade de tempo.** A capacidade é mensal; se a janela avaliada não for
   exatamente um mês, escalar proporcionalmente e dizer isso no texto.
10. **Uma etapa por vez.** Nunca avançar sozinho. Nenhuma decisão sem evidência.

---

## Parte 0 — Reconciliação com o case e travamento do critério de decisão

Antes de qualquer análise, travar o dicionário do negócio em célula executada:

1. Reimprimir os **parâmetros do Anexo 1** (capacidade, custo de perícia, custo da fraude paga,
   regra atual, premissa da perícia) a partir de constantes nomeadas — não de literais soltos.
2. Reimprimir o **critério de sucesso da Fase 1** e o **piso de viabilidade** `3,2258%`,
   derivando o número na célula: `700 / 21.700`.
3. Reimprimir a **matemática da política**: `Resultado(k) = 21.700 × TP(k) − 700 × k`, com
   `k ≤ capacidade`, e a condição de parada do k ótimo (precisão marginal `< 3,2258%`).
4. Declarar em texto, na própria célula, **qual é a decisão desta fase** ("adotar ou não a
   política baseada no escore do campeão, com qual capacidade, e o que fazer com a regra de
   R$ 50 mil") e **qual critério a encerra** (superar a regra atual com IC 95% da diferença
   excluindo zero).

Saída esperada: um bloco impresso com todos os parâmetros, o piso, o critério e a pergunta de
decisão. Esse bloco é a primeira página do documento da fase.

---

## Parte 1 — Auditoria do que está sendo avaliado

Objetivo: provar que a política avaliada é **exatamente** o artefato aprovado na Fase 4 — nem
um modelo por engano, nem um vetor de features trocado.

1. Carregar o Pipeline por `carregar_pipeline(CAMINHO_PIPELINE)` e imprimir:
   - tipo do estimador final (deve ser `XGBClassifier`);
   - os **hiperparâmetros** e conferi-los contra a tabela da Fase 4 (`docs/04_modelagem.md`, §3);
   - a lista das **18 features** e a ordem esperada.
2. Registrar **hash SHA-256** do `.joblib` e o `gerado_em`/versão do `versoes.json` ao lado.
   Este hash entra no documento: é o que liga a avaliação ao artefato auditável.
3. **Teste de fumaça do artefato:** com `SEED=42`, recarregar o holdout com
   `dividir_temporal`, prever `predict_proba` e **reproduzir** o `Resultado@900 = R$ 3.731.700`
   com `TP=201`, `FP=699`, `FN=80`. Se não reproduzir exatamente, **parar a fase** e reportar a
   divergência: um número que não reproduz invalida tudo o que vem depois.
4. Fixar as **três referências** em objeto/constante reutilizável:
   - `PoliticaNaoPericiar` (economia = 0; custo = `21.700 × fraudes` da janela);
   - `PoliticaRegraAtual` (> R$ 50 mil, ordem decrescente de `valor_reclamado`, corta em k) —
     **reimplementar como estimador**, no mesmo padrão da Fase 4, para rodar sob as mesmas
     métricas;
   - `PoliticaAleatoria` / `DummyClassifier(strategy="prior")`.
5. Declarar explicitamente no texto que **a regra atual é uma política legítima e que pode
   já ser lucrativa**: o critério é superá-la, não apenas ficar positivo (alerta (b) da Fase 1).

---

## Parte 2 — Estimativa honesta fora da amostra (resolve o limite herdado)

O holdout de agosto foi usado na seleção. Uma decisão de comitê não pode se apoiar só nele.

1. **Backtest com origem móvel (walk-forward), sem embaralhar o tempo.** Construir janelas
   cronológicas sobre abr–ago: treina em `[m1..mn]`, avalia na janela imediatamente posterior,
   avança. Mínimo de **três janelas** de avaliação. Documentar, em tabela, para cada janela:
   período de treino, período de avaliação, nº de sinistros, nº de fraudes, taxa de fraude.
2. Para **cada janela**, reportar as quatro métricas da fase — `Resultado@k`, Precisão@k,
   PR AUC, ROC AUC — para o **campeão**, para a **regra atual** e para a **referência simples**,
   com `k` proporcional à capacidade (`k = round(900 × n_janela / 12.024)`).
3. Apresentar a **média ± desvio** entre janelas e a **diferença pareada** campeão − regra atual
   por janela. Se o desvio da diferença for maior que a diferença média, declarar ruído.
4. **Teste de degradação temporal:** ordenar as janelas no tempo e mostrar se o desempenho cai.
   Se cair, quantificar; isso vira insumo do monitoramento (Parte 7) e do alerta de circuito.
5. Fechar a parte com uma frase de decisão: *"fora da amostra, em N janelas, o campeão supera a
   regra atual em X ± Y reais por mês"* — ou o contrário, se for o caso.

---

## Parte 3 — Política de perícia e economia (Pergunta 4)

1. Definir a política como objeto de primeira classe — `PoliticaPericia(estimador, k, desempate)`
   — e uma função `aplicar_politica(politica, X)` que devolve os índices periciados. A política
   é o entregável; o modelo é insumo.
2. Medir, **no mês de teste**, para **cada** política (campeão, regra atual, aleatória,
   não periciar):
   - perícias por mês executadas (k);
   - **fraudes encontradas** (TP);
   - **economia no mês em relação a não periciar nenhum sinistro** —
     `21.700 × TP − 700 × k`;
   - custo total decomposto em `21.700 × FN + 700 × FP`;
   - Precisão@k e Cobertura@k (apoio).
3. **Cálculo da regra atual como o case a descreve:** filtrar `valor_reclamado > 50.000`,
   ordenar por valor decrescente, cortar em 900 — reportar quantos pedidos o filtro gerou no
   mês (o case fala em ~950) e quantos foram de fato periciados.
4. **Ganho da política sobre a regra atual, por mês** = `Resultado(política) −
   Resultado(regra atual)`, com o intervalo de confiança da Parte 5 ao lado.
5. Responder por escrito, com o número ao lado, **o que fazer com a regra atual** — uma ação,
   não um comentário. Exemplos de forma aceitável: substituir pela política de escore a partir
   de `<data>`, com `<k>` perícias; manter como contingência se `<condição>`; extinguir o
   critério de R$ 50 mil porque `<número>`.
6. Marcar no gráfico da curva `Resultado(k)` (artefato da Fase 4) os pontos de decisão:
   `k=900` (capacidade hoje), o corte teórico `k=1.660`, o ótimo empírico `k=1.430` e o último
   `k` ainda lucrativo `k=8.710` — cada um anotado com o valor em reais.

---

## Parte 4 — Dimensionamento da equipe e valor marginal (Pergunta 5)

O comitê vai perguntar "vale a pena mais peritos?". Aqui se responde com dinheiro.

1. **Acerto mínimo de uma perícia (a mais) para se pagar:** derivar na célula
   `700 / 21.700 = 3,2258%` e explicar em uma frase por que é o mesmo número do piso da Fase 1
   (é a mesma conta vista de dois lados: o valor marginal da k-ésima perícia zera em `p = 1/31`).
2. **Curva de valor marginal por perícia:** da curva `Resultado(k)`, extrair a precisão
   marginal (TP acumulado na k-ésima posição) e marcar onde ela cruza 3,2258%. Reportar:
   - **até quantas perícias por mês compensa periciar** (`k` máximo ainda lucrativo);
   - **o `k` de menor custo total / maior resultado líquido** — explicitando a interpretação
     usada para o campo "perícias por mês de menor custo" do Quadro (ponto em que a precisão
     marginal cruza o piso), porque o enunciado admite as duas leituras;
   - **o ganho mensal além das 900 atuais** = `Resultado(k_ótimo) − Resultado(900)`.
3. **Curva de ganho por perito.** Como a contratação é discreta (1 perito = 30 perícias/mês),
   converter a curva em **reais por perito adicional** e apontar o ponto em que o ganho
   marginal por perito deixa de ser material. Entregar uma **recomendação de dimensionamento
   com número de peritos** (ex.: "manter 30", "subir para N peritos"), sempre com o número.
4. **Testar a afirmação do Otávio** ("com o dobro de peritos, o dobro de fraude"): medir
   quantas fraudes 1.800 perícias/mês encontrariam e comparar com `2 × TP(900)`. Reportar o
   **múltiplo real** (ex.: 1,42× em vez de 2,00×) e explicar a razão econômica — o topo da fila
   já concentra as fraudes mais óbvias, então a precisão marginal **decresce** com k.
5. Fechar com a **ação sobre a equipe**: manter, ampliar para N peritos ou reduzir — com o
   número que a sustenta e a condição que a invalidaria.

---

## Parte 5 — Sensibilidade, break-even e incerteza

Esta é a parte que transforma uma medição em uma **estratégia robusta**. Os parâmetros do
Anexo 1 são estimativas; a recomendação não pode quebrar se elas estiverem erradas.

1. **Incerteza estatística (obrigatória).** Bootstrap do mês de teste com `SEED=42`
   (**2.000 reamostragens**, com reposição), recalculando `Resultado@900` para a política e
   para a regra atual. Reportar **IC 95%** para cada uma e para a **diferença pareada**.
   **Critério de decisão:** a recomendação só é afirmada como ganho se o IC 95% da diferença
   **excluir o zero**; caso contrário, dizer que é indistinguível do ruído.
2. **Grade de sensibilidade econômica.** Recalcular a política recomendada (e o ganho sobre a
   regra atual) variando, um fator por vez e depois combinados:
   - `CUSTO_FRAUDE_PAGA ∈ {10.000, 15.000, 21.700, 30.000, 40.000}`
   - `CUSTO_PERICIA ∈ {500, 700, 1.000, 1.500}`
   - `CAPACIDADE ∈ {600, 900, 1.200, 1.800, 2.400}`
   Entregar a grade como tabela/heatmap, com o `k` ótimo e o `Resultado` em cada célula.
3. **Break-even econômico.** Achar e reportar o valor crítico do parâmetro em que a política
   deixa de vencer a regra atual: **a que custo da fraude paga** (ou a que razão
   `custo_fraude / custo_pericia`) a vantagem desaparece. Esse é o número de risco do projeto.
4. **Sensibilidade à taxa de fraude (drift).** Simular a mudança da prevalência na janela de
   avaliação e reportar em que taxa de fraude a política deixa de se pagar. **Atenção ao fato
   central da Fase 2:** a taxa base (2,47%) está **abaixo** do piso de viabilidade (3,2258%) —
   ou seja, periciar ao acaso dá prejuízo; a política só funciona por **concentração** no topo
   da fila. Quantificar essa concentração e o ponto em que ela se perde.
5. **Ponto de quebra operacional.** Testar o cenário realista de volume: se um mês trouxer
   mais sinistros (ex.: pico de alagamento) a taxa de fraude do fluxo muda? A política continua
   viável com o mesmo `k`? Entregar a regra de **escala do k** com o volume
   (`k = 900 × n_mês / 12.024`) e o gatilho que exige reavaliação.
6. Cada célula das grades sai de célula executada. **Nenhum número interpolado à mão.**

---

## Parte 6 — Aceitabilidade, risco e explicabilidade

Aqui entra o que o case cobra na Pergunta 1 e o que um comitê sério cobra além dele: **o que
se pode e o que não se deve usar** para decidir quem é periciado.

1. **Reprocessar a decisão sobre `uf` e `idade_segurado` sob a ótica do negócio.** Os números
   já existem (ROC AUC 0,936597 → 0,937166; PR AUC 0,538556 → 0,533792). Ir além da Fase 4:
   - **precificar a escolha.** Reajustar o campeão **sem** as duas colunas (mesmo protocolo,
     `SEED=42`) e medir o `Resultado@900` resultante. O comitê tem de ver o **preço em reais**
     de não usar idade e UF.
   - **triagem de disparidade.** Reportar Precisão@900 e taxa de seleção por faixa de idade e
     por UF — como **triagem**, não como veredito legal. Se a seleção se concentrar de forma
     forte em grupos específicos, registrar o achado e propor mitigação.
   - **fechar a posição** com justificativa de negócio em uma frase: usar / não usar, com o
     número. Este é o campo "Idade do segurado e UF: usar ou não usar" do Quadro.
2. **Auditar o `boletim_ocorrencia` pela lente do risco.** É a feature SHAP nº 1 **e** a única
   coluna com branco (4,83%). Reafirmar a hipótese de *timing* já registrada na Fase 3: se a
   política depende dela, qual é o plano se o campo deixar de ser preenchido na abertura?
   Reportar o `Resultado@900` do modelo **sem** `boletim_ocorrencia` (o número já existe como
   apoio na Fase 3) e escrever a **regra de contingência**.
3. **Confirmar o atributo morto.** Reafirmar, com o número SHAP, que `fim_de_semana` é a
   feature de menor importância e declarar a decisão (manter ou remover) com a justificativa.
4. **Explicabilidade que o analista usa.** Verificar, em célula executada, que as explicações
   SHAP locais são legíveis e estáveis para o topo da fila — o analista precisa entender
   **por que** o sinistro está ali antes de mandar o perito. Especificar o que o app deve
   mostrar por sinistro (escore, posição na fila, 3 sinais principais, texto curto) como
   **contrato de interface** para a Fase 6. Não construir o app aqui.
5. **Registro de riscos**, com severidade e mitigação: erro de calibração em produção,
   degradação temporal, mudança de prevalência abaixo do piso, dependência do
   `boletim_ocorrencia`, concentração por grupo, custo real da perícia acima do orçado, e
   atraso do cliente honesto (que **já está dentro** do R$ 700 — não contar de novo).

---

## Parte 7 — Revisão do processo e próximos passos (CRISP-DM 5.2 e 5.3)

A fase de Avaliação do CRISP-DM não é só medir: é **revisar como se chegou aqui** e **decidir o
que fazer depois**.

1. **Revisão do processo (5.2).** Lista curta e honesta, cada item com evidência:
   - o que funcionou (ex.: Pipeline dentro do fold evitou vazamento; `scale_pos_weight=1`
     preservou a calibração e o corte de 3,2258%);
   - o que custou caro (ex.: o espelhamento run a run no DagsHub degradou de ~21s para ~206s
     por configuração e obrigou a publicar só as 4 famílias vencedoras; a run histórica da
     Regra Atual não persistiu os scores e teve de ser revalidada na Fase 4);
   - o que a incidência de agosto usado na seleção **nos ensinou** sobre o método (nunca usar
     a única janela disponível tanto para escolher quanto para reportar).
2. **Lições reaproveitáveis** — a parte que serve para o próximo projeto, escrita de forma
   genérica (o usuário vai salvar este documento como base).
3. **Determinar os próximos passos (5.3).** Tomar e justificar **uma** decisão:
   - **implantar** a política, com condições;
   - **iterar** (o que exatamente faltaria e por quê);
   - **não implantar** (se o ganho não sobreviver ao IC 95% nem à sensibilidade).
4. **Plano de monitoramento a implantar na Fase 6** — especificação, não implementação:
   - métricas mensais a acompanhar: Precisão@900, `Resultado@900` realizado, taxa de fraude do
     fluxo, distribuição do escore, PSI/deriva das features principais;
   - **gatilhos de alerta** com número (ex.: taxa de fraude do mês abaixo de X%; Precisão@900
     abaixo de Y% por dois meses; `boletim_ocorrencia` em branco acima de Z%);
   - **circuit breaker:** condição em que a fila volta à regra anterior automaticamente;
   - **cadência de recalibração/re-treino** e quem aprova.
5. Encerrar a parte com a **frase de decisão da fase** (uma frase, com o número do ganho e a
   condição que a invalidaria).

---

## Parte 8 — Consolidação, testes e entregáveis

1. **Módulo novo `src/quilha/avaliacao.py`** (não inflar `modelagem.py`), com docstring em
   português e nomes em português. Conteúdo mínimo:
   - `PoliticaPericia` e a `aplicar_politica(...)`;
   - reimplementação da `PoliticaRegraAtual` como estimador compatível com scikit-learn;
   - `resultado_economico(y_true, indices_periciados)` com a decomposição
     `21.700 × FN + 700 × FP` e a economia vs. não periciar;
   - `bootstrap_resultado(...)`, `grade_sensibilidade(...)`, `break_even(...)`,
     `curva_ganho_por_perito(...)`;
   - `triagem_disparidade(...)`;
   - `montar_quadro_respostas(...)` que devolve o Quadro em `DataFrame` e em Markdown.
   Nada de `fit` fora do treino; nada de caminho absoluto; raiz por `encontrar_raiz`.
2. **Notebook da fase**, na mecânica da Fase 3/4: `notebooks/clean/05_avaliacao.py` (fonte, sem
   outputs) e `notebooks/executed/05_avaliacao.ipynb` (executado, com saídas visíveis) via
   `jupytext`. Rodar de ponta a ponta **sem credenciais de MLOps**.
3. **Testes `tests/test_avaliacao.py`** (mínimo):
   - `resultado_economico` confere a conta manual `21.700 × TP − 700 × k` em caso sintético;
   - piso de viabilidade = `700 / 21.700` = 3,2258%;
   - a `PoliticaRegraAtual` dá nota monotônica em `valor_reclamado` e zero abaixo de R$ 50 mil
     (mesma garantia da Fase 4);
   - `aplicar_politica` respeita `k` e a ordem decrescente de escore, com desempate estável;
   - `dividir_temporal` não vaza (nenhuma data de teste anterior à de treino);
   - o `Resultado@900` do campeão carregado **reproduz** o número de referência da Fase 4.
4. **Artefatos em `reports/`.** Para não colidir com as tabelas `parte5_*` da **Fase 4**,
   usar prefixo próprio: tabelas `reports/tabelas/avaliacao_*.csv` e figuras
   `reports/figures/05_*.png` (ex.: `avaliacao_backtest.csv`, `avaliacao_sensibilidade.csv`,
   `avaliacao_bootstrap.csv`, `avaliacao_quadro_respostas.csv`, `05_01_curva_resultado_k.png`,
   `05_02_sensibilidade.png`, `05_03_ganho_por_perito.png`).
5. **Documento `docs/05_avaliacao.md`** com: decisão da fase, backtest, economia (P4),
   dimensionamento (P5), sensibilidade/incerteza, aceitabilidade/risco, revisão do processo,
   próximos passos, o Quadro de Respostas preenchido e a tabela de critérios de aceite com o
   estado real de cada um.
6. **MLOps:** registrar na fase os runs de backtest e sensibilidade (mesmo experimento
   `quilha-fraude-fase4` ou um novo `quilha-fraude-fase5`), com o hash do artefato auditado
   nas tags. Sem credenciais, degradar para SQLite sem quebrar.
7. Atualizar a tabela **Progresso da execução** deste prompt ao fim de cada parte.
8. `ruff check src tests` e `pytest -q` **verdes** ao final, com a saída real colada.

---

## Critérios de aceite

| # | Critério | Evidência exigida |
|---|---|---|
| 1 | Parâmetros do Anexo 1 e piso de 3,2258% derivados em célula | Bloco impresso |
| 2 | Artefato auditado por hash e **reproduz** R$ 3.731.700 no corte de 900 | Saída com hash, TP/FP/FN e resultado |
| 3 | Backtest com ≥3 janelas cronológicas, média ± desvio e diferença pareada | Tabela por janela |
| 4 | P4 completa: perícias/mês, fraudes, economia vs. não periciar, regra atual com 900, ganho e **ação sobre a regra atual** | Tabela + frase de ação |
| 5 | P5 completa: acerto mínimo, `k` ótimo, ganho além das 900, 1.800 perícias e **ação sobre a equipe** | Tabela + frase de ação |
| 6 | IC 95% por bootstrap do `Resultado@900` e da **diferença** política − regra atual | Tabela com IC |
| 7 | Grade de sensibilidade (custo da fraude × custo da perícia × capacidade) e break-even | Tabela/heatmap + valor crítico |
| 8 | Sensibilidade à taxa de fraude, com o ponto em que a política deixa de se pagar | Gráfico/tabela |
| 9 | Preço em reais de **não usar** `uf` e `idade_segurado` + triagem de disparidade | Números + posição escrita |
| 10 | Contingência para `boletim_ocorrencia` e confirmação do atributo morto | Números + regra |
| 11 | Revisão do processo (5.2) e decisão de próximos passos (5.3), com plano de monitoramento e gatilhos numéricos | Seção do documento |
| 12 | Quadro de Respostas preenchido, conferido contra as saídas do notebook | Tabela no documento + CSV |
| 13 | Notebook roda de ponta a ponta **sem credenciais**; `ruff` e `pytest` verdes | Execução limpa + saída real dos comandos |

---

## Entregáveis

- `notebooks/clean/05_avaliacao.py` e `notebooks/executed/05_avaliacao.ipynb`
- `docs/05_avaliacao.md`
- `src/quilha/avaliacao.py` e `tests/test_avaliacao.py`
- `reports/tabelas/avaliacao_*.csv` e `reports/figures/05_*.png`
- Quadro de Respostas preenchido (no documento e em `avaliacao_quadro_respostas.csv`)
- Runs de backtest e sensibilidade registrados no MLflow (DagsHub quando houver credencial)
- Tabela **Progresso da execução** deste prompt atualizada

---

## Pergunta obrigatória a responder com número (não antes de medir)

> **Após medir: a política baseada no escore do campeão supera a regra atual de R$ 50 mil com
> as mesmas 900 perícias — e esse ganho sobrevive (a) ao intervalo de confiança de 95% da
> diferença e (b) à variação dos parâmetros econômicos do Anexo 1?**

Responda com: **(a)** o `Resultado@900` da política e da regra atual lado a lado, com a
diferença; **(b)** o IC 95% por bootstrap dessa diferença e o veredito (exclui ou não o zero);
**(c)** a grade de sensibilidade mostrando em que ponto a vantagem desaparece (break-even);
**(d)** a conclusão — **inclusive se a resposta for "não"**. Se o ganho não sobreviver, diga
exatamente sob quais parâmetros a política ainda se justifica e sob quais não. Nada de afirmar
antes de medir, e nada de suavizar o resultado.

---

## Quadro de Respostas (consolidar; preencher com os números das células)

| Pergunta | Campo | Resposta |
|---|---|---|
| 1 | Colunas usadas no modelo final | |
| 1 | Colunas deixadas de fora | |
| 1 | Idade do segurado e UF: usar ou não usar | |
| 1 | ROC AUC · PR AUC no teste: com idade e UF · sem idade e UF | |
| 2 | Taxa de fraude da base | |
| 2 | Taxa e nº por tipo: Colisão · Vidros · Roubo/Furto · Terceiros · Alagamento · Incêndio | |
| 2 | Taxa e nº por canal: App · Corretor · Central 0800 | |
| 2 | Taxa e nº por boletim: Sim · Não · em branco | |
| 2 | Tratamento dado ao boletim em branco no modelo | |
| 2 | Taxa e nº por tempo de apólice: até 30 · 31 a 90 · 91 a 180 · 181 a 365 · +365 | |
| 2 | Taxa e nº por valor reclamado: até 10 · 10–20 · 20–30 · 30–40 · 40–50 · 50–60 · +60 mil | |
| 3 | Modelos comparados · modelo recomendado | |
| 3 | Período do treino · período do teste · nº de sinistros no teste | |
| 3 | ROC AUC · PR AUC da referência simples no teste | |
| 3 | ROC AUC · PR AUC no treino · no teste (modelo recomendado) | |
| 4 | Política recomendada · perícias por mês | |
| 4 | Fraudes encontradas · economia no mês: política recomendada | |
| 4 | Fraudes encontradas · economia no mês: regra atual com 900 perícias | |
| 4 | Ganho da política sobre a regra atual, por mês | |
| 4 | Ação sobre a regra atual | |
| 5 | Acerto mínimo para uma perícia se pagar | |
| 5 | Perícias por mês de menor custo · ganho mensal além das 900 | |
| 5 | Fraudes encontradas com 1.800 perícias por mês | |
| 5 | Ação sobre a equipe | |

Campos das Perguntas 1 a 3 vêm das Fases 2 a 4 — **conferir contra as saídas**, não recriar.
Campos das Perguntas 4 e 5 são produzidos nesta fase.

---

## Modo de trabalho

Uma etapa por vez, com confirmação antes de avançar. Executar de verdade e colar a **saída
real**; se algo não rodou, dizer que não rodou. Nunca inventar número. Se existir caminho mais
sênior do que o pedido, avisar e explicar **antes** de executar. Se o pedido piorar a qualidade
técnica, avisar em duas frases e fazer mesmo assim. Encerrar cada resposta com
**"Aguardando o próximo passo."**

---

## Progresso da execução

| Parte | Descrição | Status |
|---|---|---|
| 0 | Reconciliação com o case e travamento do critério | concluída |
| 1 | Auditoria do artefato e das três referências | concluída |
| 2 | Backtest com origem móvel (estimativa fora da amostra) | concluída |
| 3 | Política e economia — Pergunta 4 | concluída |
| 4 | Dimensionamento e valor marginal — Pergunta 5 | concluída |
| 5 | Sensibilidade, break-even e incerteza | concluída |
| 6 | Aceitabilidade, risco e explicabilidade | concluída |
| 7 | Revisão do processo e próximos passos | concluída |
| 8 | Consolidação, testes e entregáveis | concluída |

---

## Apêndice — como reaproveitar este prompt em outro projeto

Este documento foi escrito para ser a **base de outros cases**. O que é genérico e o que é
específico da Quilha:

- **Genérico (copiar como está):** a estrutura CRISP-DM 5.1–5.3 (avaliar resultado, revisar
  processo, determinar próximos passos); o mandato de "produzir decisão, não modelo"; as três
  referências obrigatórias (não fazer nada / política vigente / aleatório); a exigência de
  **IC 95% e sensibilidade** antes de recomendar; a precificação da escolha ética; o plano de
  monitoramento com gatilhos numéricos.
- **Específico (trocar):** os parâmetros do Anexo 1 (capacidade, custos, regra atual), a função
  de custo `21.700 × TP − 700 × k`, o piso `3,2258%`, os nomes de arquivos, módulos e o
  rastreio das perguntas do comitê.

Regra de ouro reaproveitável: **nenhuma recomendação sem incerteza, nenhum ganho sem
break-even, nenhuma ação sem número.**

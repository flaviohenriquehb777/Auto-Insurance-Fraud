# Fase 3 — Preparação dos Dados

> **Projeto:** Detecção de fraude em sinistros de automóvel — Quilha Seguros
> **Fase CRISP-DM:** 3 de 6 — Preparação dos Dados
> **Base de entrada:** `data/raw/quilha_sinistros.csv` (60.000 sinistros, 17 colunas)
> **Código:** `src/quilha/preparacao.py`, `src/quilha/pipeline.py`, `src/quilha/explicacao.py`
> **Notebook:** `notebooks/executed/03_preparacao_dados.ipynb` (executado, com saídas)
> **Testes:** `tests/test_preparacao.py` (19 testes, todos passando)
> **Artefato:** `models/pipeline_fraude.joblib` + `models/pipeline_fraude.versoes.json`
> **Data:** 2026-11-04 — **Versão:** 1.0.0

---

## 1. Produto da fase

| Entregável | Onde | Estado |
|---|---|---|
| Função de preparo `preparar(df)` | [preparacao.py](file:///c:/Users/flavi/Documents/GitHub/Auto-Insurance-Fraud/src/quilha/preparacao.py) | Pronto |
| Pipeline serializável (preparo → imputação → modelo) | [pipeline.py](file:///c:/Users/flavi/Documents/GitHub/Auto-Insurance-Fraud/src/quilha/pipeline.py) | Pronto |
| Explicação SHAP agregada por feature original | [explicacao.py](file:///c:/Users/flavi/Documents/GitHub/Auto-Insurance-Fraud/src/quilha/explicacao.py) | Pronto |
| Notebook limpo + executado | `notebooks/clean/03_preparacao_dados.*`, `notebooks/executed/03_preparacao_dados.ipynb` | Pronto |
| Testes automatizados | [test_preparacao.py](file:///c:/Users/flavi/Documents/GitHub/Auto-Insurance-Fraud/tests/test_preparacao.py) | 19/19 passando |
| Artefato salvo em disco | `models/pipeline_fraude.joblib` (10,8 KB) | Pronto |

O CSV original **não foi editado**: todo tratamento mora em código.

---

## 2. Entrada e saída

```
Entrada (schema cru do CSV)                    Saída de `preparar`
────────────────────────────                   ────────────────────────────────
17 colunas, tudo texto                          21 colunas tipadas

  id_sinistro            ──────────────────►   id_sinistro        (controle)
  data_ocorrencia        ──────────────────►   data_ocorrencia    (controle)
  data_abertura          ──────────────────►   data_abertura      (controle)
                                               ── 18 FEATURES ──
  idade_segurado         ──────────────────►   idade_segurado
  dias_de_apolice        ──────────────────►   dias_de_apolice
                                               faixa_apolice        (derivada)
  sinistros_12m          ──────────────────►   sinistros_12m
  idade_veiculo          ──────────────────►   idade_veiculo
  valor_fipe             ──────────────────►   valor_fipe
  valor_reclamado        ──────────────────►   valor_reclamado
                                               razao_reclamado_fipe (derivada)
                                               faixa_valor_reclamado(derivada)
  hora_ocorrencia        ──────────────────►   hora                 (derivada)
                                               periodo_dia          (derivada)
  data_ocorrencia        ──────────────────►   fim_de_semana        (derivada)
  data_ocorrencia+abert. ──────────────────►   atraso_abertura_dias (derivada)
  canal_abertura         ──────────────────►   canal_abertura
  tipo_sinistro          ──────────────────►   tipo_sinistro
  uf                     ──────────────────►   uf
  boletim_ocorrencia     ──────────────────►   boletim_ocorrencia   (branco → categoria)
  terceiro_identificado  ──────────────────►   tem_terceiro_identificado (derivada)

  valor_indenizado       ──────► DESCARTADA (vazamento)
  fraude                 ──────► DESCARTADA (gabarito)
```

**Nada é descartado em linha:** 60.000 entram, 60.000 saem. Colunas de controle
(3) + features (18) = 21.

---

## 3. Decisões de tratamento, com o número que sustenta cada uma

### a) `id_sinistro` — chave de controle, nunca feature

Identificador único; 0 duplicatas na base (Fase 2, seção 8.1). Não é atributo. É
preservado na saída para rastreabilidade: quando o app mostrar "sinistro
SIN-26-120990, probabilidade 24%", é esse campo que amarra a explicação ao caso.

### b) `valor_indenizado` — excluída por vazamento

| Grupo | Sinistros | Taxa de fraude |
|---|---|---|
| `valor_indenizado` = 0 (não pago) | 8.064 (13,44%) | **17,37%** |
| `valor_indenizado` > 0 (pago) | 51.936 (86,56%) | 0,15% |
| | | **razão de 115x** |

O campo é preenchido **no encerramento** do sinistro — depois da decisão de perícia.
Um valor que só existe depois da resposta não pode entrar na pergunta. É o caso
clássico de *data leakage*: o modelo aprenderia "não pagou → é fraude", uma
tautologia, e na abertura o campo simplesmente não existe.

### c) `data_ocorrencia` e `data_abertura` — não são atributos

Data crua não é feature: um modelo treinado em abr–jul/2026 não sabe o que fazer
com ago/2026, e um número de dia não tem monotonicidade com fraude. As duas colunas
**permanecem no DataFrame** apenas para (i) a divisão temporal, (ii) a auditoria e
(iii) alimentar atributos derivados interpretáveis (`fim_de_semana`,
`atraso_abertura_dias`). Nenhuma informação temporal se perde — ela entra na forma
que o negócio consegue ler.

### d) `boletim_ocorrencia` em branco — categoria explícita, não imputação

| Categoria | Sinistros | Fraudes | Taxa de fraude |
|---|---|---|---|
| Não | 27.970 | 1.044 | 3,73% |
| **Não informado** (era branco) | **2.899** | **94** | **3,24%** |
| Sim | 29.131 | 342 | 1,17% |
| *Base* | *60.000* | *1.480* | *2,47%* |

2899 campos em branco = 4,83% da base. O branco tem taxa de fraude de 3,24%,
**acima da média de 2,47%** — ou seja, o campo em branco carrega sinal. Imputar
pela moda (que seria "Sim") jogaria esses 2.899 sinistros de 3,24% para 1,17%:
destruiria informação. Vira categoria própria.

**Hipótese declarada e risco de timing.** Este tratamento assume que o
`boletim_ocorrencia` **já é conhecido na abertura** do sinistro. Essa premissa não
está escrita no case — o dicionário diz apenas "se o segurado apresentou boletim de
ocorrência (Sim ou Não)". Se o campo for preenchido **depois** (quando o documento é
anexado), usá-lo na nota de triagem é *data leakage* e o modelo não funcionaria na
produção. O que os dados permitem dizer (medido):

| Evidência | Número | Leitura |
|---|---|---|
| Campo preenchido na base | 95,17% | o branco é exceção, não regra |
| Branco por tipo de sinistro | 1,59% (Roubo/Furto) a 7,59% (Vidros) | sem padrão estrutural que prove preenchimento tardio |
| Custo de remover a coluna | **−R$ 447.020/mês** e **−0,0507 de PR AUC** (5 sementes) | é a feature mais valiosa da base |
| Modelo **sem** a coluna | PR AUC 0,4617 vs. 0,0247 da referência simples | perde qualidade, mas **não** fica inválido |

**Decisão: manter a coluna**, com a premissa registrada. Removê-la "por segurança"
custaria R$ 447 mil/mês de economia esperada — caro demais para um risco não
confirmado. Dois fatos tornam o risco administrável: (1) a única hipótese é o
*timing* da captura, e ela se resolve com uma pergunta à área (não com dado);
(2) se a resposta for ruim, o dano é uma perda de qualidade medida — o Pipeline
continua rodando e continua batendo a referência simples, porque um chamado sem o
campo cai na categoria "Não informado" em vez de quebrar. **Ação para o deploy:**
confirmar com a Quilha se o campo existe na abertura; se existir apenas depois,
religar o modelo sem ele (`FEATURES` sem `boletim_ocorrencia`, re-treino, mesmas
métricas).

### e) `terceiro_identificado` — derivada booleana, original descartada

| `tem_terceiro_identificado` | Sinistros | Taxa de fraude |
|---|---|---|
| True | 25.617 | 3,39% |
| False | 34.383 | 1,78% |

A coluna original é estruturalmente ditada por `tipo_sinistro`: só varia dentro de
"Colisão" (nos demais tipos é constante). Como `tipo_sinistro` já entra no modelo,
manter as duas é duplicar a mesma informação com nomes diferentes — e o
`OneHotEncoder` criaria colunas perfeitamente colineares, o que estraga a leitura
do SHAP (a contribuição se dividiria entre duas features correlacionadas).
A informação útil — "existe um terceiro identificado?" — é recuperada em uma
bandeira binária, que corta a taxa de fraude ao meio (1,78% → 3,39%).

**Decisão: mantém-se apenas a derivada.**

### f) `hora_ocorrencia` — três atributos derivados

| `periodo_dia` | Sinistros | Taxa de fraude |
|---|---|---|
| madrugada (00–05h) | 4.539 | 3,30% |
| manhã (06–11h) | 18.622 | 2,16% |
| tarde (12–17h) | 20.224 | 2,55% |
| noite (18–23h) | 16.615 | 2,47% |

| `fim_de_semana` | Sinistros | Taxa de fraude |
|---|---|---|
| Sim | 14.961 | 2,49% |
| Não | 45.039 | 2,46% |

- `hora` (0–23) entra como contínua: a hora crua permite cortes que o balde de
  período não permite. Valor fora de 00–23 vira nulo.
- `periodo_dia` entrega a leitura de negócio. A madrugada tem 3,30% contra 2,47% da
  base — 34% acima.
- `fim_de_semana` **não separa nada** (2,49% vs 2,46%, diferença de 0,03 ponto). É
  mantido porque é derivado de um campo que já está na base e não custa nada; o
  modelo da Fase 4 decide com evidência (importância SHAP) se descarta. Registrado
  aqui como atributo de sinal fraco, para não vender o que ele não entrega.

### g) `atraso_abertura_dias` — existe no momento da decisão

`data_abertura − data_ocorrencia`. Ambas as datas são conhecidas na abertura, então
o atributo é legítimo. A Fase 2 mostrou que 688 sinistros têm ocorrência fora da
janela auditada (evento em março, abertura em abril) — o que **confirma** que o
atraso é real e não erro de dado. Um atraso grande entre o evento e a comunicação é
um dos sinais clássicos de fraude (o segurado pensa antes de acionar).

### h) `faixa_apolice` — nas faixas do case

| Faixa | Sinistros | Taxa de fraude |
|---|---|---|
| até 30 dias | 1.181 | **13,46%** |
| 31 a 90 | 2.319 | 10,56% |
| 91 a 180 | 3.450 | 8,96% |
| 181 a 365 | 8.119 | 1,96% |
| mais de 365 | 44.931 | 1,35% |

É o atributo mais forte da base: apólice com menos de 30 dias tem taxa de fraude
**10x a média**. `dias_de_apolice` cru também entra: a faixa dá a leitura de
negócio, a contagem preserva a granularidade fina.

### i) `valor_reclamado` — faixa, razão e a decisão sobre log1p

| Faixa | Sinistros | Taxa de fraude |
|---|---|---|
| até 10 mil | 38.954 | 1,10% |
| 10 a 20 | 8.674 | 2,78% |
| 20 a 30 | 3.354 | 3,16% |
| 30 a 40 | 2.035 | 2,95% |
| **40 a 50** | **2.231** | **25,15%** |
| 50 a 60 | 1.177 | 3,48% |
| acima de 60 | 3.575 | 1,17% |

O pico em **40 a 50 mil** é o achado mais forte da fase: 25,15% de fraude, 10x a
média, e é justamente a faixa **logo abaixo** do corte da regra atual (R$ 50 mil).
Quem frauda sabe da regra e pede abaixo dela. A faixa é o que captura esse degrau;
uma árvore conseguiria aprender sozinha, mas a feature explícita torna o padrão
legível para o comitê.

`razao_reclamado_fipe` = `valor_reclamado / valor_fipe`. Um pedido acima do valor de
referência do veículo é candidato natural a orçamento inflado ou "dano total
forçado". Divisão por FIPE nula ou não positiva devolve nulo (nunca infinito).

**Log1p nas colunas monetárias: NÃO aplicado.** A justificativa é numérica: a
correlação de Spearman entre `valor_reclamado` e `log1p(valor_reclamado)` é
**1,000000**. Um log1p é uma transformação estritamente monótona; qualquer modelo
baseado em árvores (o plano da Fase 4) toma decisões por comparações de ordem
(`valor > limite`), e ordenar é tudo o que ele faz. O log não muda nenhuma ordem,
logo não muda nenhuma divisão, logo não muda o modelo — **mas destrói a unidade**:
a contribuição do SHAP passaria a ser lida em log-reais, e o comitê leria
"contribuição 0,37" em vez de "R$ 8.400". Para modelos lineares ou redes o log
seria necessário; para árvores, é custo sem benefício. A assimetria de 3,51 é real,
mas quem a absorve é o `pd.cut` (que discretiza a cauda) e o próprio modelo.

### j) `data_ocorrencia` fora da janela — não descartar

688 registros têm ocorrência anterior a 01/04/2026 com abertura dentro da janela
auditada. Não são erro: são sinistros legítimos comunicados com atraso (o evento de
março, o aviso de abril). Descartá-los removeria casos de atraso longo justamente
na ponta do sinal descrito em (g). A coluna de controle temporal é `data_abertura`.

### k) Nenhuma linha descartada

| Verificação | Resultado |
|---|---|
| Linhas descartadas por nulo | **0** |
| Linhas descartadas por outlier | **0** |
| Nulos novos em campos numéricos | **0** |

Nenhum outlier foi removido: a Fase 2 não encontrou valor impossível (idades entre
18–100, sem negativos, sem FIPE zero). O que existe é cauda longa — valor alto de
sinistro e de veículo é plausível e é exatamente onde mora a fraude. Descartar
outlier aqui seria descartar o alvo.

---

## 4. A função `preparar`: contrato, pureza e robustez

**Assinatura:** `preparar(df) -> pd.DataFrame`

**Contrato de entrada.** Aceita o schema cru **sem** `fraude` e **sem**
`valor_indenizado` (o app recebe um chamado novo, sem gabarito). Funciona com a base
inteira ou com **uma linha**. Aceita as colunas em **qualquer ordem**. **Ignora**
colunas extras — se o app mandar `id_sinistro` ou um campo de controle a mais, nada
quebra e a saída não muda.

**Contrato de saída.** Colunas e ordem **estáveis e previsíveis**, sempre as mesmas.
Datas viram `datetime`, contagens viram inteiro tratando nulo (`Int64`), categóricas
viram texto.

**Pureza.** Nenhuma estatística aprendida: zero média, mediana, moda ou quantil.
Isso é papel do `SimpleImputer`, dentro do Pipeline. A função não tem estado global,
não tem efeito colateral e é idempotente — rodar duas vezes na mesma entrada crua
devolve resultado idêntico.

**Robustez para dado novo.** *Nenhuma entrada provoca exceção.* Campo vazio, texto
inválido, `valor_reclamado` ausente e data malformada viram nulo (`NaN`/`NaT`/
`<NA>`), e a imputação do Pipeline resolve depois. Justificativa: o app é a
fronteira com o mundo real — o corretor pode digitar a data como `31/02/2026` ou
deixar o valor em branco, e o serviço não pode cair por isso.

| Entrada suja | Saída |
|---|---|
| `""` (campo em branco) | nulo |
| `"31/02/2026"` (data inexistente) | nulo |
| `"abc"`, `"??"` em campo numérico | nulo |
| `"99:99"` em hora | nulo |
| coluna desconhecida | ignorada |

**Formatos aceitos em `valor_reclamado`** — o CSV usa vírgula decimal, mas um app
pode enviar qualquer um dos três: `"8837,07"`, `"8837.07"` e `8837.07` produzem
todos `8837.07`.

**Detalhe de implementação que não é óbvio.** O texto é normalizado para `object`
com `numpy.nan` na fronteira (nunca `pd.NA`). Motivo empírico, descoberto por sonda
nesta fase: o `SimpleImputer` do scikit-learn 1.9.1 quebra com `pd.NA` em coluna
`object` (`TypeError: boolean value of NA is ambiguous`). Pela mesma razão, a
bandeira `tem_terceiro_identificado` (dtype `boolean` nullable) entra na rama
**numérica** do `ColumnTransformer`, e não na categórica.

---

## 5. Divisão treino/teste por tempo

**Evidência para escolher o corte** — taxa de fraude e volume por mês de abertura:

| Mês de abertura | Sinistros | Fraudes | Taxa |
|---|---|---|---|
| 2026-04 | 11.692 | 283 | 2,42% |
| 2026-05 | 12.146 | 313 | 2,58% |
| 2026-06 | 11.947 | 314 | 2,63% |
| 2026-07 | 12.311 | 289 | 2,35% |
| 2026-08 | 11.904 | 281 | 2,36% |

A taxa é estável: amplitude de **0,28 ponto percentual** em cinco meses, sem mês
atípico. Logo, cortar no último mês não cria nem teste com fraude de menos nem teste
com período anômalo.

**Corte: 2026-08-01.**

| Conjunto | Período | Sinistros | Fraudes | Taxa |
|---|---|---|---|---|
| Treino | 01/04/2026 → 31/07/2026 (4 meses) | 48.096 | 1.199 | 2,49% |
| Teste | 01/08/2026 → 31/08/2026 (1 mês) | 11.904 | 281 | 2,36% |

**Sem `shuffle`. Sem `stratify` no holdout.** O corte é cronológico; estratificar é
impossível em corte por data e, além disso, desnecessário: as duas taxas já são
quase idênticas (2,49% vs 2,36%), então o teste tem fraude de sobra para medir
(281 casos). O holdout mede o mundo real — sinistros mais recentes que os do treino,
exatamente como o case exige.

**O teste tem exatamente 1 mês de propósito.** A capacidade de perícia do case é
mensal (900/mês). Com o teste cobrindo um mês fechado, a avaliação da política na
Fase 5 é 1:1 com a capacidade: "das 900 vagas de agosto, quantas acertaram?" — sem
nenhuma conversão de escala.

**Validação cruzada (fica registrada para a Fase 4).** Quando houver CV — busca de
hiperparâmetros ou estimativa de estabilidade — o `StratifiedKFold` com seed fixa é
obrigatório. Com 2,47% de positivos, um fold sem estratificação pode cair com 0
fraudes e a PR AUC oscila por azar amostral, não por mérito do modelo. A regra é:
**holdout mede o mundo real; CV estratificada mede estabilidade.** Não se misturam.

---

## 6. O Pipeline

```
Pipeline([
    ("preparo",   FunctionTransformer(preparar)),                        # 1. Preparo
    ("imputacao", ColumnTransformer([                                    # 2. Imputação
        ("numericas",   SimpleImputer(strategy="median", add_indicator=True),
                        [9 numéricas + 1 booleana]),
        ("categoricas", Pipeline([
            ("imputacao",   SimpleImputer(strategy="most_frequent")),
            ("codificacao", OneHotEncoder(handle_unknown="ignore",
                                          sparse_output=False)),
        ]),             [8 categóricas]),
    ])),
    ("modelo",    DummyClassifier(strategy="prior")),                    # 3. Fase 4
]).set_output(transform="pandas")
```

**Estratégia de imputação, diferenciada por tipo e justificada por número:**

- **Numéricas → `median`.** `valor_reclamado` tem assimetria **3,51** e
  `valor_indenizado` 3,71 (Fase 2, seção 9): a média é arrastada pela cauda. Em uma
  coluna com cauda longa, a média não é um valor típico — a mediana é.
- **`add_indicator=True`** — reserva para nulo real em campo numérico no `fit`.
  Conceitualmente, um sinistro sem valor informado não é o mesmo que um sinistro
  com valor mediano: a ausência pode ela própria ser sinal (a Fase 2 mostrou isso
  no `boletim_ocorrencia`). Com a base atual, porém, não há nulo numérico nenhum —
  então a bandeira não existe (ver nota abaixo).
- **Categóricas → `most_frequent`.** Mediana de texto não existe. A moda é a única
  estratégia com sentido para uma variável nominal.
- **`OneHotEncoder(handle_unknown="ignore", sparse_output=False)`** — categoria nova
  no app devolve zeros em vez de levantar exceção, e a matriz densa é o que o SHAP
  precisa para nomear colunas.

**Nota honesta sobre o `add_indicator`.** Depois de `preparar`, a base **não tem
nulo em nenhuma coluna** — os 2.899 nulos do CSV viraram a categoria "Não
informado". O `SimpleImputer` só cria coluna de indicador para colunas que tinham
nulo **no `fit`**; medido: `indicator_.features_.shape == (0,)` e a matriz densa
sai com **0 colunas de indicador**. E não adianta o app mandar campo vazio: um
chamado com dois campos numéricos em branco sai com **o mesmo número de colunas**
da matriz de treino (medido com `SimpleImputer(strategy="median",
add_indicator=True)` ajustado na base). Ou seja, a bandeira **não** protege o
chamado novo.

O que acontece de fato com um campo numérico em branco no app: ele é imputado
pela **mediana do treino** (`idade_segurado` → 44, `valor_fipe` → 5.824,76) e
entra no modelo como um valor típico. O modelo não distingue "vazio" de "valor
mediano", porque nunca viu uma linha vazia no treino — não há como aprender isso.
A bandeira fica como **reserva futura** (se alguma coluna numérica passar a ter
nulo real no `fit`, ela nasce sozinha e a agregação do SHAP já sabe devolvê-la à
feature original). O único campo com branco real hoje é o `boletim_ocorrencia`,
que é categórico e tem tratamento próprio (seção 3-d).

**Nenhum transformador com `fit` é ajustado fora do treino.** O `fit` do Pipeline
recebe só o treino; o `predict_proba` recebe o teste. Imputadores e `OneHotEncoder`
aprendem apenas com o passado — é o próprio `Pipeline` que garante isso, e é
justamente por isso que tudo mora dentro dele.

**Nada de balanceamento artificial.** Sem SMOTE nesta fase. Com 2,47% de positivos,
um sobreamostrador sintético cria fraudes que não existem e distorce a
probabilidade — e a probabilidade é o que o SHAP explica ao comitê. Se o
desbalanceamento atrapalhar, a discussão de `class_weight` acontece na Fase 4, com
evidência de PR AUC.

**A matriz densa tem 67 colunas** (10 numéricas + 57 de one-hot):

| Coluna categórica | Categorias |
|---|---|
| `canal_abertura` | 3 — App, Central 0800, Corretor |
| `tipo_sinistro` | 6 — Alagamento, Colisão, Incêndio, Roubo/Furto, Terceiros, Vidros |
| `uf` | 27 |
| `boletim_ocorrencia` | 3 — Não, **Não informado**, Sim |
| `periodo_dia` | 4 |
| `fim_de_semana` | 2 |
| `faixa_apolice` | 5 |
| `faixa_valor_reclamado` | 7 |

**O modelo é um placeholder.** `DummyClassifier(strategy="prior")` existe para provar
que o Pipeline roda de ponta a ponta; ele não aprende nada. O modelo real entra na
Fase 4 sem tocar em mais nada: é só passar `construir_pipeline(modelo=...)`.

---

## 7. Persistência e o artefato no Streamlit Cloud

**O problema levantado na fase:** o `.gitignore` ignorava `models/*`, e o Streamlit
Cloud clona o repositório do GitHub para servir o app — ele **não tem** nenhum
armazenamento de artefatos separado (não é um registry de modelos). Se o `.joblib`
não estiver no repositório, o app sobe e quebra no primeiro `predict_proba`.

**Solução aplicada:** exceção explícita no `.gitignore`.

```
models/*
!models/.gitkeep
!models/*.joblib
```

O artefato passa a ser versionado; o resto da pasta continua fora.

**Avaliação das alternativas, para registro:**

| Alternativa | Por que não |
|---|---|
| Treinar no `startup` do app | Sobe o tempo de boot, gasta CPU do free tier, e o modelo servido passa a depender do ambiente de execução — reprodutibilidade perdida |
| Git LFS | Custo operacional (cota de banda/storage) para um arquivo de KB; resolve um problema que não existe neste tamanho |
| Baixar de um bucket (S3/GCS) | Exige credencial no Streamlit Secrets e um serviço externo só para servir 11 KB |

**Estado atual:** o artefato tem **10,8 KB** (o `DummyClassifier` não guarda dados;
o Pipeline real da Fase 4 deve ficar na casa dos MB, ainda muito abaixo do limite de
100 MB por arquivo do GitHub).

**Round-trip verificado.** Salvar → carregar do disco → prever devolve **o mesmo
valor**, bit a bit (`0.0249293081` antes e depois).

**Registro de versões gravado ao lado** (`models/pipeline_fraude.versoes.json`), para
que a nuvem que serve o app não rode com versões diferentes das que geraram o
artefato:

| Componente | Versão local |
|---|---|
| Python | 3.12.12 |
| scikit-learn | 1.9.1 |
| pandas | 3.1.0 |
| numpy | 2.5.4 |

O arquivo guarda também a **lista das 18 features**: se o Pipeline for recarregado
com uma ordem de colunas diferente, a divergência aparece na hora.

---

## 8. Explicabilidade: agregação SHAP por feature original

**O problema.** O `OneHotEncoder` dilui uma variável em várias colunas:
`tipo_sinistro` vira `tipo_sinistro_Colisão`, `tipo_sinistro_Vidros`, ... O SHAP
explica coluna a coluna, então a explicação crua sairia como
"`categoricas__tipo_sinistro_Colisão` contribuiu +0,03" — ilegível para a diretoria.

**A solução.** Somar as contribuições de volta à feature original. A regra:

1. `missingindicator_<feature>` → volta para `<feature>` (as colunas criadas pelo
   `add_indicator` também entram na agregação — sinal de dado ausente pertence à
   feature da qual o dado faltou);
2. nome idêntico a uma feature original (numérica ou booleana) → ela mesma;
3. qualquer outro caso é one-hot `<feature>_<categoria>` → casa com a **feature mais
   longa** que seja prefixo do nome. O maior prefixo resolve sozinho ambiguidades
   entre nomes que são prefixo um do outro (ex.: `faixa_apolice` vs.
   `faixa_valor_reclamado`).

**Validação nesta fase (lógica pura, matriz sintética, sem modelo real):**

| Coluna transformada | Valor SHAP | Feature original | Soma |
|---|---|---|---|
| `numericas__idade_segurado` | 1,0 | `idade_segurado` | 1,0 |
| `numericas__valor_fipe` | 2,0 | `valor_fipe` | 2,5 |
| `numericas__missingindicator_valor_fipe` | 0,5 | `valor_fipe` | ↑ |
| `categoricas__tipo_sinistro_Colisão` | −3,0 | `tipo_sinistro` | 1,0 |
| `categoricas__tipo_sinistro_Roubo` | 4,0 | `tipo_sinistro` | ↑ |
| `categoricas__uf_SP` | 1,5 | `uf` | 1,5 |
| `categoricas__faixa_valor_reclamado_40 a 50` | −2,0 | `faixa_valor_reclamado` | −2,0 |

Soma das colunas transformadas: **4,0**. Soma depois da agregação: **4,0**.
Agregar é reagrupar colunas, nunca descartar valor — e o teste automatizado trava
essa propriedade. Nome que não casa com nenhuma feature **levanta erro** em vez de
chutar: melhor falhar alto do que explicar a feature errada ao comitê.

**A explicação ponta a ponta com o modelo real fica para a Fase 4**, quando existir
um modelo que aprenda algo. A função `explicar_sinistro` já está escrita e é
modelo-agnóstica (usa `shap.Explainer` sobre `predict_proba`), então serve tanto
para o `DummyClassifier` de agora quanto para a árvore da Fase 4.

---

## 9. Testes

`uv run pytest tests/test_preparacao.py -v` → **19 passaram em 9,35s**.

| # | Teste | O que trava |
|---|---|---|
| 1 | `preparar(uma_linha) == linha da base` | Critério 1 (colunas, ordem e dtypes) |
| 2 | Ordem das colunas não importa | Contrato de entrada |
| 3 | Idempotência na mesma entrada crua | Pureza da função |
| 4 | Sem nulos novos em campos numéricos | Critério 2 |
| 5 | Três formatos de `valor_reclamado` | Robustez de entrada |
| 6 | Sem `fraude` e sem `valor_indenizado` | Critério 3 (vazamento) |
| 7 | Coluna extra é ignorada | Contrato de entrada |
| 8 | Campo vazio, data malformada, texto inválido | Robustez: nenhuma exceção |
| 9 | Linha totalmente vazia | Robustez extrema |
| 10-11 | Mapeamento one-hot/indicador e erro em nome desconhecido | Agregação |
| 12-14 | Agregação: 1 coluna por feature, soma preservada, várias linhas, matriz inconsistente | Lógica SHAP |
| 15 | `fit` no treino + `predict_proba` em linha crua | Critério 4/5 ponta a ponta |
| 16 | Salvar → carregar → mesmo valor | Critério 5 |
| 17 | Categoria nunca vista | `handle_unknown` |
| 18 | Campo vazio no Pipeline | Imputação do app |
| 19 | Caminho do artefato | Deploy |

---

## 10. Lista final de features e quando cada uma é conhecida

**18 features.** Nenhuma delas existe depois da decisão de perícia — é isso que
torna o conjunto utilizável no momento da abertura.

| # | Feature | Tipo | Derivada de | Quando é conhecida |
|---|---|---|---|---|
| 1 | `idade_segurado` | numérica (inteiro) | — | Antes do sinistro (cadastro) |
| 2 | `dias_de_apolice` | numérica (inteiro) | — | Antes do sinistro (cadastro) |
| 3 | `sinistros_12m` | numérica (inteiro) | — | Antes do sinistro (histórico) |
| 4 | `idade_veiculo` | numérica (inteiro) | — | Antes do sinistro (cadastro) |
| 5 | `valor_fipe` | numérica | — | Antes do sinistro (tabela FIPE) |
| 6 | `valor_reclamado` | numérica | — | **Na abertura** — é o próprio pedido |
| 7 | `hora` | numérica (inteiro) | `hora_ocorrencia` | **Na abertura** (o evento já ocorreu) |
| 8 | `atraso_abertura_dias` | numérica (inteiro) | `data_abertura` − `data_ocorrencia` | **Na abertura** |
| 9 | `razao_reclamado_fipe` | numérica | `valor_reclamado` / `valor_fipe` | **Na abertura** |
| 10 | `tem_terceiro_identificado` | booleana | `terceiro_identificado` | **Na abertura** |
| 11 | `canal_abertura` | categórica | — | **Na abertura** |
| 12 | `tipo_sinistro` | categórica | — | **Na abertura** |
| 13 | `uf` | categórica | — | **Na abertura** |
| 14 | `boletim_ocorrencia` | categórica | — (branco → "Não informado") | **Na abertura** |
| 15 | `periodo_dia` | categórica | `hora_ocorrencia` | **Na abertura** |
| 16 | `fim_de_semana` | categórica | `data_ocorrencia` | **Na abertura** |
| 17 | `faixa_apolice` | categórica | `dias_de_apolice` | Antes do sinistro (cadastro) |
| 18 | `faixa_valor_reclamado` | categórica | `valor_reclamado` | **Na abertura** |

### Colunas do CSV que **não** são features

| Coluna | O que acontece | Motivo |
|---|---|---|
| `fraude` | Descartada | Gabarito — é a resposta a prever |
| `valor_indenizado` | Descartada | **Vazamento**: só existe no encerramento (115x de diferença de taxa) |
| `id_sinistro` | Controle | Chave; amarra a explicação ao caso, não é atributo |
| `data_ocorrencia` | Controle | Alimenta `fim_de_semana` e `atraso_abertura_dias`; a data crua não é feature |
| `data_abertura` | Controle | Coluna de corte temporal e auditoria |
| `hora_ocorrencia` | Consumida | Vira `hora` e `periodo_dia` |
| `terceiro_identificado` | Substituída | Redundante com `tem_terceiro_identificado` |

**Resumo do caminho:** 17 colunas no CSV → 7 fora do conjunto de atributos
(2 descartadas por vazamento/gabarito, 3 de controle, 2 consumidas por derivação) →
**18 features** que o modelo lê.

---

## 11. Critérios de aceite

| # | Critério | Estado | Evidência |
|---|---|---|---|
| 1 | `preparar(uma_linha)` == linha correspondente da base | Atendido | Teste 1; `assert_frame_equal` com dtypes |
| 2 | Zero NaN novo em campo numérico; zero linha descartada | Atendido | 60.000 → 60.000; 0 nulos novos |
| 3 | Nenhuma coluna vazada no conjunto de atributos | Atendido | Lista da seção 10 + teste 6 |
| 4 | Nenhum transformador com `fit` fora do treino | Atendido | `fit` no treino, `predict_proba` no teste |
| 5 | Pipeline salvo, recarregado e prevendo | Atendido | `0.0249293081` antes e depois |
| 6 | Notebook executado + documento da fase | Atendido | `notebooks/executed/03_preparacao_dados.ipynb`, este documento |

---

## 12. Pontos que passam para a Fase 4

### Decisões já tomadas no fechamento da fase

Medições feitas com um modelo de árvore real (`HistGradientBoostingClassifier`,
seed 42), treino abr–jul (48.096 sinistros) e teste ago (11.904, 281 fraudes),
capacidade de **900 perícias**, **5 sementes** por cenário. Diferenças abaixo de
R$ 100 mil/mês ficaram dentro do desvio das diferenças pareadas — nesse patamar
não há sinal, há ruído.

| Cenário | ROC AUC | PR AUC | Precisão@900 | Fraudes pegas | Resultado/mês |
|---|---|---|---|---|---|
| **Com idade e UF (18 features)** | 0,9280 | **0,5124** | 21,93% | 197,4 | R$ 3.653.580 |
| Sem `uf` | 0,9286 | 0,5042 | 22,31% | 200,8 | R$ 3.727.360 |
| Sem `idade_segurado` | 0,9288 | 0,5064 | 22,09% | 198,8 | R$ 3.683.960 |
| Sem as duas (16 features) | 0,9292 | 0,5092 | 21,96% | 197,6 | R$ 3.657.920 |
| Sem `fim_de_semana` | 0,9286 | 0,5119 | 21,82% | 196,4 | R$ 3.631.880 |
| Sem `boletim_ocorrencia` | 0,8930 | 0,4617 | 19,64% | 176,8 | R$ 3.206.560 |

1. **`uf` e `idade_segurado`: usar as duas.** Decidido. O PR AUC é o **melhor com
   as duas** (0,5124 vs. 0,5042 / 0,5064 / 0,5092) — é a métrica que vale sob
   desbalanceamento. As diferenças de ROC AUC ficam na 4ª casa e o resultado
   financeiro de quem remove (–R$ 30 mil a +R$ 74 mil/mês) está **dentro do desvio
   das diferenças pareadas** (R$ 55,9 mil a R$ 98,4 mil): não é ganho, é sorte de
   semente. Nenhuma evidência justifica abrir mão de duas features de cadastro que
   já existem no chamado. *Números desta medição são preliminares* — a Fase 4
   reporta os mesmos quatro números com o modelo campeão, para o quadro de respostas.
2. **`fim_de_semana`: manter.** Decidido. Amplitude própria de **0,04 p.p.**
   (2,49% vs. 2,46%) e nenhum ganho em interação: na grade `fim_de_semana` ×
   `periodo_dia`, a maior diferença (madrugada: 3,21% vs. 3,60%) é sobre 1.111
   sinistros e não sobrevive ao erro-padrão. Sem removê-la o resultado é
   −R$ 21.700/mês (dentro do ruído) e o PR AUC cai 0,0005. Como a coluna é
   **derivada da data** (não é campo que o operador digita) e a árvore simplesmente
   não a usa quando ela não ajuda, o custo de mantê-la é zero: manter é a decisão
   de menor risco e menor retrabalho. Fica registrada como **atributo neutro**, e a
   Fase 4 confirma o descarte pela importância SHAP se ela continuar sem uso.
3. **`add_indicator=True`: manter o código, corrigir a documentação.** Decidido.
   Medido: a bandeira é *no-op* no treino **e** no app (0 colunas criadas em ambos
   — seção 6). Ou seja, ela não protege nada hoje; o que protegia era a crença de
   que protegia. A bandeira fica como reserva (se uma coluna numérica passar a ter
   nulo real no `fit`, ela nasce e o SHAP já sabe devolvê-la à feature original) e o
   caminho passou a ser **testado de propósito**
   (`test_indicador_de_ausente_aparece_quando_ha_nulo_no_treino`), em vez de
   descrito só no comentário. Um campo numérico em branco no app é imputado pela
   mediana do treino — comportamento documentado, não acidente.
4. **`boletim_ocorrencia`: manter, com a hipótese de *timing* registrada.**
   Decidido (seção 3-d). Removê-la custa R$ 447 mil/mês — é a feature mais valiosa
   da base. A ação de verificação fica agendada para antes do go-live.
5. **`models/pipeline_fraude.versoes.json`: não versionar.** Decidido. O `uv.lock`
   já congela o ambiente inteiro de forma reprodutível, e o registro carrega um
   `gerado_em` que mudaria a cada re-execução do notebook, gerando diff inútil no
   repositório. Só o `.joblib` é versionado — é o que o deploy realmente carrega.

### O que segue para a Fase 4

6. **`class_weight` vs. sobreamostragem.** Decidir com PR AUC e Precisão@900, nunca
   com acurácia.
7. **Modelo real no lugar do `DummyClassifier`**: basta
   `construir_pipeline(modelo=...)` — nada mais muda.
8. **Explicação SHAP ponta a ponta** com o modelo real, agora que a agregação por
   feature original está validada (o mapeamento do indicador de ausente já tem teste).

---

## Apêndice — como reproduzir

```bash
uv run pytest tests/test_preparacao.py -v                       # testes
uv run jupytext --to notebook notebooks/clean/03_preparacao_dados.py \
    -o notebooks/clean/03_preparacao_dados.ipynb                # notebook limpo
uv run jupytext --to notebook --execute notebooks/clean/03_preparacao_dados.py \
    -o notebooks/executed/03_preparacao_dados.ipynb             # notebook executado
```

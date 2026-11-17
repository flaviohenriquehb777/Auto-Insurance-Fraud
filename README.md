<h1 align="center">Priorização de Perícias em Sinistros de Automóvel</h1>

<p align="center">
  <strong>Quilha Seguros</strong><br>
  Política de triagem orientada por dados, com capacidade mensal limitada
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-1.67-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit 1.67"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Licen%C3%A7a-MIT-green.svg" alt="Licença MIT"></a>
  <a href="https://github.com/flaviohenriquehb777/Auto-Insurance-Fraud/actions/workflows/ci.yml"><img src="https://github.com/flaviohenriquehb777/Auto-Insurance-Fraud/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
</p>

<p align="center">
  <a href="https://quilha-triagem.streamlit.app/">
    <img src="assets/previews/painel-quilha-neon.png" alt="Painel Quilha Seguros — clique para abrir a demonstração interativa." width="900">
  </a>
</p>

<p align="center"><a href="https://quilha-triagem.streamlit.app/"><strong>▶ Abrir demonstração interativa</strong></a></p>

> **Escopo:** projeto de estudo de caso, não é um sistema de decisão autônoma nem um serviço de produção. O modelo prioriza a fila; a decisão de periciar continua humana.

## Sumário

- [Visão geral do modelo](#visao-geral-do-modelo)
- [Objetivos da análise](#objetivos-da-analise)
- [Estrutura do modelo e do repositório](#estrutura-do-modelo-e-do-repositorio)
- [Base de dados](#base-de-dados)
- [Metodologia de análise: CRISP-DM](#metodologia-de-analise-crisp-dm)
- [Resultados-chave e apresentação](#resultados-chave-e-apresentacao)
- [Tecnologias utilizadas](#tecnologias-utilizadas)
- [Instalação e uso](#instalacao-e-uso)
- [Pipeline reproduzível com DVC](#pipeline-reproduzivel-com-dvc)
- [Uso do app: fila, escolha do sinistro e normalização](#uso-do-app-fila-escolha-do-sinistro-e-normalizacao)
- [Publicação no GitHub Pages](#publicacao-no-github-pages)
- [Licença](#licenca)
- [Contato](#contato)

## Visão geral do modelo

O projeto aborda a alocação de **900 perícias por mês** entre sinistros de automóvel. Em vez de
classificar casos por um limiar genérico, o Pipeline calcula um escore de priorização e organiza a
fila para que a equipe avalie primeiro os casos de maior prioridade estimada.

O artefato final é um Pipeline serializado (preparação, imputação e XGBoost), com **18 features**
conhecidas no momento da abertura do sinistro. Ele não declara que uma pessoa cometeu fraude: a
interface comunica prioridade operacional, apresenta sinais explicativos e mantém a decisão sob
responsabilidade do perito.

## Objetivos da análise

- Comparar uma política de escore com a regra atual baseada em valor reclamado acima de R$ 50 mil.
- Maximizar o resultado líquido dentro da capacidade fixa de perícia.
- Validar o modelo com separação temporal, usando sinistros futuros como holdout.
- Tornar explícitos os contratos de entrada, a explicabilidade e os gatilhos de monitoramento.
- Entregar uma interface que organiza a fila sem emitir veredito sobre o segurado.

## Estrutura do modelo e do repositório

O fluxo de pontuação é:

```text
Payload CSV → normalização da representação → contrato de dados → preparo → imputação → XGBoost
            → ordenação por escore → corte proporcional em k → fila para revisão humana
```

| Caminho | Responsabilidade |
|---|---|
| `src/quilha/normalizacao.py` | Reescreve a **representação** (milhar pt-BR, caixa/espaço, data `dd/mm/aaaa`) e **relata** cada mudança. Só representação; significado e domínio seguem para o contrato. |
| `src/quilha/contrato.py` | Valida campos e recusa o lote inteiro quando há entrada inválida. |
| `src/quilha/preparacao.py` | Prepara as 18 features canônicas do Pipeline. |
| `src/quilha/pipeline.py` | Carrega e valida o artefato serializado. |
| `src/quilha/app.py` | Fila, análise individual, valor e governança no Streamlit. |
| `src/quilha/avaliacao.py` | Políticas, métricas econômicas, backtest e Quadro de Respostas. |
| `src/quilha/relatorio.py` | Gera o resumo executivo DOCX usando o CSV auditado como fonte única. |
| `src/quilha/estagios.py` | Estágios do pipeline DVC (`preparo`, `avaliacao`) executados por `dvc repro`. |
| `notebooks/clean/06_consolidado.py` | Fonte Jupytext do notebook consolidado das Fases 1 a 6. |
| `tests/` | Testes de contrato, aplicação, avaliação e geração de relatório. |
| `docs/` | Pareceres, método, decisões e documentação das fases. |

## Base de dados

A base do estudo de caso contém **60.000 sinistros rotulados** e **17 colunas**, cobrindo o período
de abril a agosto de 2026. Há **1.480 casos rotulados como fraude (2,47%)**. O holdout temporal é
agosto de 2026, com 11.904 sinistros; abril a julho compõem o período de treino.

`valor_indenizado` (resultado posterior ao fechamento), identificadores e campos de data/hora
originais não entram como preditores. A decisão de implantação registrada em
[docs/06_implantacao.md](docs/06_implantacao.md) confirma que as features mantidas estão disponíveis
no momento da abertura.

O app aceita arquivos **CSV** de sinistros abertos e valida o esquema antes de pontuar. Use apenas
dados autorizados; não envie dados pessoais ou operacionais a uma demonstração pública.

## Metodologia de análise: CRISP-DM

| Fase | Trabalho realizado | Documentação |
|---|---|---|
| 1. Entendimento do negócio | Objetivo econômico, capacidade de 900 perícias e critérios de sucesso. | [Fase 1](docs/01_entendimento_negocio.md) |
| 2. Entendimento dos dados | Perfil da base, qualidade, alvo e sinais exploratórios. | [Fase 2](docs/02_entendimento_dados.md) |
| 3. Preparação dos dados | Contrato analítico, transformação e features sem vazamento. | [Fase 3](docs/03_preparacao_dados.md) |
| 4. Modelagem | Baselines, validação temporal, seleção e explicabilidade. | [Fase 4](docs/04_modelagem.md) |
| 5. Avaliação | Política econômica, robustez, incerteza e governança. | [Fase 5](docs/05_avaliacao.md) |
| 6. Implantação | Artefato, contrato de entrada, app, monitoramento e entregáveis. | [Fase 6](docs/06_implantacao.md) |

## Resultados-chave e apresentação

Os valores abaixo são resultados medidos no holdout de agosto de 2026, sob as premissas do estudo
de caso. Não são uma promessa de resultado futuro.

| Indicador | Resultado |
|---|---:|
| Referência simples (`DummyClassifier`, PR AUC) | 0,024929 |
| XGBoost, validação cruzada no treino (ROC AUC / PR AUC) | 0,940794 / 0,519659 |
| XGBoost, holdout temporal (ROC AUC / PR AUC) | 0,936597 / 0,538556 |
| Fraudes encontradas no corte auditado | 201 em 900 perícias (precisão de 22,33%) |
| Resultado líquido da política de escore | R$ 3.731.700,00 no mês de teste |
| Resultado líquido da regra atual, com 900 perícias | −R$ 326.200,00 no mês de teste |
| Ganho da política sobre a regra atual | R$ 4.057.900,00; IC 95% bootstrap: R$ 3.537.100,00 a R$ 4.665.500,00 |

O modelo foi selecionado por resultado líquido sob capacidade, não por acurácia. A conclusão depende
das premissas do case (inclusive que a perícia encontra uma fraude quando ela existe) e deve ser
reavaliada com dados operacionais antes de qualquer uso real. Consulte [Avaliação](docs/05_avaliacao.md)
e o [parecer da Fase 6](docs/06_implantacao.md) para método, ressalvas e decisões.

### Entregáveis reproduzíveis

- **Notebook consolidado:** a fonte versionada é `notebooks/clean/06_consolidado.py`; o `.ipynb`
  executado é gerado localmente pelo comando da seção [Instalação e uso](#instalacao-e-uso).
- **Resumo executivo:** gerado por `src/quilha/relatorio.py` a partir de
  `reports/tabelas/avaliacao_quadro_respostas.csv`.

Por configuração do repositório, `06_consolidado.ipynb` e `Relatorio_Quilha_Fraude.docx` são
artefatos locais ignorados pelo Git; eles não aparecem como arquivos baixáveis neste repositório.

## Tecnologias utilizadas

| Área | Ferramentas |
|---|---|
| Linguagem e ambiente | Python 3.12, `uv` |
| Dados e modelagem | pandas, NumPy, scikit-learn, XGBoost, LightGBM |
| Explicabilidade e interface | SHAP, Streamlit |
| Experimentação e análise | MLflow, Jupyter, Jupytext |
| Versionamento de dados e pipeline | DVC |
| Relatórios e qualidade | python-docx, pytest, Ruff, GitHub Actions |

## Instalação e uso

Pré-requisitos: **Python 3.12** e [`uv`](https://docs.astral.sh/uv/).

```powershell
git clone https://github.com/flaviohenriquehb777/Auto-Insurance-Fraud.git
Set-Location Auto-Insurance-Fraud
uv sync --all-groups
uv run streamlit run app.py
```

Sem upload de arquivo, o app abre com o lote de demonstração do mês de teste. Para usar outro lote,
envie um CSV com os campos aceitos pelo contrato; uma linha inválida faz o app recusar o lote
completo, em vez de produzir uma fila parcial.

Para reproduzir os dois entregáveis locais:

```powershell
# Notebook executado (a fonte Jupytext permanece versionada)
uv run jupytext --to notebook --execute notebooks/clean/06_consolidado.py -o 06_consolidado.ipynb

# Resumo executivo DOCX (o pacote do projeto está em src/)
$env:PYTHONPATH = "src"
uv run python -m quilha.relatorio
```

Para validar código e testes:

```powershell
uv run ruff check src tests
uv run pytest -q
```

## Pipeline reproduzível com DVC

O repositório versiona o **pipeline de dados** com [DVC](https://dvc.org/). O `dvc repro`
recalcula, a partir da base crua, a divisão temporal e a medição da política que a Fase 5
afirmou — de modo que os números centrais do estudo deixam de ser "digitados" e passam a ser
reproduzíveis por qualquer pessoa que clone o repositório.

O grafo tem dois estágios:

```text
preparo:    data/raw/quilha_sinistros.csv
            ─► data/processed/treino.csv  +  data/processed/teste.csv

avaliacao:  data/processed/treino.csv  +  data/processed/teste.csv
            + models/pipeline_fraude.joblib
            ─► reports/metricas/avaliacao.json
```

| Estágio | O que faz | Saídas |
|---|---|---|
| `preparo` | Divide a base crua pelo corte temporal (abr–jul treino / ago teste), sem reescrever a regra de corte. | `data/processed/treino.csv`, `data/processed/teste.csv` |
| `avaliacao` | Mede a política do campeão congelado contra a regra atual de R$ 50 mil, sob o critério `21.700 × TP − 700 × k`. | `reports/metricas/avaliacao.json` (métrica) |

Duas decisões de desenho:

- **O modelo auditado é entrada, não saída.** `models/pipeline_fraude.joblib` entra como `dep` e
  nunca é regravado — não existe estágio de treino. Isso protege o artefato aprovado e mantém a
  cadeia focada em *reproduzir a medição*, não em re-treinar.
- **A base crua permanece no Git.** `data/raw/quilha_sinistros.csv` é dependência (o app e o
  `pytest` a leem) e não um `out`. Só os derivados entram no DVC; por isso o app publicado e o CI
  funcionam sem depender do remote.

### Como reproduzir

O `PYTHONPATH` de que o pacote `quilha` precisa é declarado em `dvc.env` (versionado de propósito)
e carregado por `uv run --env-file dvc.env` — o mesmo contrato do `app.py` da raiz.

```powershell
uv sync --all-groups        # instala o grupo mlops (dvc, mlflow)

# O remote local já vem declarado em .dvc/config (versionado) e aponta para a pasta
# irmã ../quilha-dvc-store, que é criada no primeiro `dvc push` — sem custo de nuvem.
dvc repro                   # reproduz o pipeline; no-op se nada mudou
dvc metrics show            # exibe reports/metricas/avaliacao.json
dvc push                    # envia as saídas ao remote local
```

O `dvc.lock` guarda o hash de cada entrada e saída da última execução: se um `dep` muda, o DVC
refaz só o que for afetado; se nada mudou, cada estágio é ignorado. Em ambiente novo, `dvc pull`
restaura as saídas a partir do remote.

**Medição reproduzida** (`reports/metricas/avaliacao.json`) — são os mesmos valores da Fase 5:

| Métrica | Valor |
|---|---:|
| Sinistros no mês de teste | 11.904 |
| Fraudes no mês de teste | 281 |
| Verdadeiros positivos em 900 perícias | 201 |
| Resultado da política do campeão | R$ 3.731.700,00 |
| Resultado da regra atual | −R$ 326.200,00 |
| Ganho do campeão sobre a regra atual | R$ 4.057.900,00 |

## Uso do app: fila, escolha do sinistro e normalização

O app tem três telas, alimentadas por **um único lote** enviado na barra lateral — a mesma população
para a fila, para o sinistro individual, para a saúde do modelo e para o drift.

- **Fila do mês.** Pontua o lote, ordena pelo escore e corta em `k` proporcional
  (`900 × n_sinistros / 12.024`, decisão D4). O corte é capacidade de negócio, não escolha do
  analista. A fila é exportável em CSV.
- **Sinistro individual.** O perito **escolhe** qual sinistro detalhar; o app nunca escolhe por ele.
- **Valor e governança.** Ganho mensal com intervalo de confiança, valor por sinistro, saúde do
  modelo e gatilhos de circuit breaker.

**Como escolher o sinistro (tela 2), origem "Da fila do mês":**

- **Clique na linha da fila** (tela 1): a seleção viaja para a tela 2 e o seletor já abre naquele
  sinistro.
- **Seletor da fila**: lista `#posição · id_sinistro · escore`, com busca por digitação. A posição e
  o escore conferem com a fila da tela 1 — no lote de demonstração (mês de teste), o topo é
  `#1 · SIN-26-100637 · escore 0,9561`.
- **Digitar um chamado**: para o sinistro que ainda não está no lote; o perito preenche os campos do
  contrato à mão.

**Normalização da representação.** Antes do contrato, toda entrada — o lote e o chamado digitado —
passa por `quilha.normalizacao.normalizar`, que reescreve apenas a **forma** do valor e **relata**
cada mudança na tela (expander "Normalização de representação"). No lote de UAT versionado
`data/raw/lote_teste_uat.csv` (46 linhas):

| Regra | Exemplo | Resultado |
|---|---|---|
| B1 · milhar pt-BR | `valor_reclamado = "1.234,56"` | `1234.56` |
| B2 · caixa/espaço | `uf = " sp "` | `"SP"` |
| B3 · data `DD/MM/AAAA` | `data_ocorrencia = "15/08/2026"` | `"2026-08-15"` |

O que **não** é representação continua recusado pelo contrato, com linha, campo, valor e motivo (no
mesmo lote: `idade_segurado = "abc"`, `hora_ocorrencia = "99:99"`,
`canal_abertura = "Telefone"`). Uma linha inválida recusa o lote inteiro (`fail closed`), em vez de
produzir uma fila parcial: a linha omitida pode ser exatamente a fraude.

Princípio que separa os dois casos: **representação → normaliza e reporta; significado, ambiguidade
ou domínio → recusa.** Consulte o [parecer das melhorias](docs/07_melhorias.md) para as evidências.

## Publicação no GitHub Pages

**GitHub Pages hospeda conteúdo estático; não executa este app Python/Streamlit.** Se usado, Pages
pode publicar uma página de apresentação e documentação do projeto. A demonstração interativa deve
ser implantada separadamente, por exemplo no
[Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud).

A demonstração interativa está publicada em:

> **[quilha-triagem.streamlit.app](https://quilha-triagem.streamlit.app/)**

A captura no topo deste README é clicável e leva direto ao app. O deploy é feito a partir da
branch `main`, com Python 3.12 e dependências resolvidas pelo `uv.lock`.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

## Contato

<div align="center">

| Canal | Link / contato |
|---|---|
| Nome | Flávio Henrique Barbosa |
| LinkedIn | [linkedin.com/in/flavio-henrique-barbosa-38465938](https://www.linkedin.com/in/flavio-henrique-barbosa-38465938) |
| E-mail | [flaviohenriquehb777@outlook.com](mailto:flaviohenriquehb777@outlook.com) |

</div>

<p align="center">
  <a href="https://www.linkedin.com/in/flavio-henrique-barbosa-38465938">
    <img src="assets/brand/favicon.png" alt="LinkedIn de Flávio Henrique Barbosa" width="48">
  </a>
</p>

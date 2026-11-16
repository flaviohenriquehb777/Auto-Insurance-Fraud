# Melhorias da POC — Partes A e B · Parecer

**Projeto:** Detecção de fraude em sinistros de automóvel — Quilha Seguros
**Escopo:** elevar a POC entregue na Fase 6 **sem reabrir o modelo** e **sem reintroduzir a falha
silenciosa**. Plano e execução em [docs/prompts/07_melhorias.md](prompts/07_melhorias.md).
**Data:** 16/11/2026.

***

## Fronteira travada — o que NÃO mudou

Esta é uma melhoria de **produto e de fronteira**, não de modelo. Os invariantes abaixo foram
reconferidos **depois** das duas partes e permanecem idênticos:

| Invariante | Valor (conferido nesta entrega) |
|---|---|
| Artefato implantado | SHA-256 `65c7eb5ea1325f4911f9223fa3bf1d2f7aec909e77a30865a47f25a837bb10a3` |
| Estimador | XGBoost (`max_depth=4`, `n_estimators=350`, `learning_rate=0.03`, `random_state=42`) |
| Features | **18** canônicas |
| Teste de fumaça — Campeão @900 (ago/2026) | `TP=201` · **R$ 3.731.700,00** · `confere=True` |
| Ganho da política | **R$ 4.057.900,00** · IC 95% [R$ 3.537.100; R$ 4.665.500] |
| Suíte de testes | **166 verdes** (baseline 133 → 137 na Parte A → 166 na Parte B) |

Nenhum arquivo de modelo foi tocado: `models/pipeline_fraude.joblib` não faz parte deste diff.

***

## Parte A — O perito escolhe o sinistro (tela 2) · CONCLUÍDA

**Problema real:** a tela "Sinistro individual" abria `head(1)` — **a primeira linha do arquivo**,
um sinistro arbitrário, sem relação com a fila vista na tela 1; e a tela 2 tinha **upload próprio**,
separado da fila, então o rótulo "Da fila do mês" não correspondia ao comportamento.

**O que mudou:**

| Item | Antes | Depois |
|---|---|---|
| Fonte do lote | quatro uploads independentes | **um único lote** na barra lateral, guardado em `st.session_state`; a fila é calculada uma vez e reaproveitada |
| Escolha do sinistro | `head(1)` (arbitrário) | o perito escolhe por **id** (busca), por **posição na fila** ou **clicando na linha da fila** |
| Ponte entre as telas | — | a fila carrega `linha_no_lote`, e é por ela que a tela 2 acha o sinistro — a **mesma** linha do lote |
| "Digitar um chamado" | permanecia | **permanece** (o caso legítimo do sinistro que ainda não está no lote) |

A escolha mora em **funções puras** (`rotulos_da_fila`, `escolher_da_fila(lote, fila, indice)`), com
testes; a tela ficou fina.

**Evidência real (16/11/2026):**

```text
A · fila do mês de teste: 900 entradas | k proporcional = 891
A · topo:   #1 · SIN-26-100637 · escore 0.9561
A · segundo:#2 · SIN-26-159848 · escore 0.9287
A · linha_no_lote do topo: 11273
A · id no lote (linha 11273): SIN-26-100637
A · id devolvido pela escolha: SIN-26-100637
A · a escolha devolve exatamente a linha do lote: True
```

Ou seja: escolher o índice 0 na tela 2 abre **exatamente** o sinistro do topo da fila da tela 1.

***

## Parte B — Normalização determinística, com relatório · CONCLUÍDA

**Problema real:** a fronteira recusava textos que são apenas **outra representação** do mesmo
valor — recusar representação inequívoca é atrito operacional sem ganho de segurança.

### Princípio que rege a parte

> **Representação → normaliza e reporta. Significado, ambiguidade ou domínio → recusa.**

### Regras adotadas (determinísticas e reportadas)

| # | Regra | Exemplo | Resultado |
|---|---|---|---|
| B1 | Número pt-BR com **os dois** separadores | `"1.234,56"` | `1234.56` — **aceito** |
| B2 | Categórica com **caixa/espaço** | `" sp "` | `"SP"` — **aceito** |
| B3 | Data `DD/MM/AAAA` (convenção dia-primeiro) | `"15/08/2026"` | `2026-08-15` — **aceito** |

### O que CONTINUA recusado (é isto que prova que o limite está no lugar certo)

| Caso | Por quê |
|---|---|
| `idade_segurado = "abc"` | não é representação; não existe número ali |
| `hora_ocorrencia = "99:99"` | não existe 99 horas — coagir = inventar |
| `canal_abertura = "Telefone"` | exigiria **tabela de sinônimos do negócio** (não adotada) |
| número com **um só** ponto (`"1.234"`) | ambíguo (1,234 ou 1.234?); segue lido como `1.234` — ver nota abaixo |

### Garantia "sem falha silenciosa"

- O normalizador (`quilha.normalizacao.normalizar`) **devolve um relatório** do que mudou (linha,
  campo, de → para, regra), **exibido na tela** (expander). Nada é corrigido às escondidas.
- O que **não** for normalizável segue para o **contrato**, que recusa com linha, campo, valor e
  motivo — exatamente como antes (`fail closed`).
- **Um só parser de número:** `preparacao._numero` é importado pela preparação, pelo contrato e pelo
  normalizador. Não há duas verdades sobre a vírgula.

### Um buraco real fechado (decisão D5)

Na Parte B o contrato passou a **tolerar representação** (aceita `" sp "` porque casa com `"SP"`).
O caminho **"Digitar um chamado"** validava mas **não** normalizava: um `" sp "` digitado à mão
passaria pela validação e chegaria **cru** ao modelo, que não conhece a categoria — **escore trocado
sem aviso**, exatamente a falha silenciosa que o projeto existe para matar. A correção foi estender
a mesma fronteira (normalizar → relatar → contrato → modelo) ao chamado digitado.

**Evidência real — a falha fechada:**

```text
B · relatório individual: [{'linha': 0, 'campo': 'uf', 'de': ' sp ', 'para': 'SP', 'regra': 'B2 · caixa/espaço'}]
B · escore cru = 0.020030141 | canônico = 0.021106075 | normalizado = 0.021106075
B · normalizado == canônico: True | cru != canônico: True
```

O escore **cru** difere do canônico — era esse número trocado que passaria calado; depois de
normalizar, ele coincide com o canônico.

### Nota de reversão consciente (declarada)

A decisão documentada na Fase 6 ([docs/06_implantacao.md](06_implantacao.md), Parte 1) fixava
`"21.700,00"` e `"1.234"` como **não tratados**. A regra **B1** reverte isso **apenas para o caso
inequívoco** (dois separadores: ponto de milhar **e** vírgula decimal). O caso de **um só ponto**
continua como está. A reversão é **parcial e declarada** — e o CI a trava: o teste
`tests/test_contrato.py` fica vermelho se alguém voltar a **recusar** `"21.700,00"`.

***

## B5 — Lote de UAT estendido

O `data/raw/lote_teste_uat.csv` passou de **44** para **46** linhas, com os casos de representação
**acrescentados** (`"1.234,56"`, `" sp "`) e os irrecuperáveis **mantidos**.

**Saída real (o que o app faz com esse arquivo):**

```text
linhas de dados no lote: 46
relatório de normalização (o que a melhoria B reescreveu):
   linha            campo          de        para                   regra
      41 data_ocorrencia   15/08/2026  2026-08-15   B3 · data DD/MM/AAAA
      45             uf          sp          SP   B2 · caixa/espaço
      44 valor_reclamado     1.234,56     1234.56   B1 · milhar pt-BR
erros que SEGUEM recusados (3):
   [linha 40] idade_segurado = 'abc': não converte para número (formatos aceitos: '8837,07', '8837.07' ou '1.234,56')
   [linha 42] hora_ocorrencia = '99:99': não está no formato 'HH:MM' entre 00:00 e 23:59
   [linha 43] canal_abertura = 'Telefone': valor fora do domínio: esperado um de ['App', 'Corretor', 'Central 0800']
```

*(as linhas acima são 0-based, na mesma numeração que o contrato usa.)*

**Correção honesta de uma ambiguidade do plano:** o prompt listava `"15/08/2026"` tanto entre as
representações a resolver **quanto** entre os "quatro irrecuperáveis" — contradição, porque com B3 a
data deixa de ser erro. Resolvi pelo princípio: `"15/08/2026"` passou a ser **caso de representação
resolvido**; os "quatro irrecuperáveis" ficam **três** (`abc`, `99:99`, `Telefone`), que é o que de
fato permanece irrecuperável. O lote continua carregando os três, e o app segue recusando o lote
inteiro por causa deles (`fail closed`).

> **Nota (16/11/2026):** os arquivos de UAT (`lote_teste_uat.csv` e `lote_teste_uat_valido.csv`)
> **passaram a ser versionados** no repositório, para que a demonstração seja reprodutível por
> qualquer pessoa. Isso corrige a nota anterior da Fase 6, que os descrevia como não versionados.

***

## Verificação final (saída real, nada digitado à mão)

| Verificação | Resultado |
|---|---|
| `ruff check src tests` | **All checks passed!** |
| `pytest -q` | **166 passed** (baseline 133; +4 na Parte A, +29 na Parte B) |
| Hash do `.joblib` | `65c7eb5ea1325f4911f9223fa3bf1d2f7aec909e77a30865a47f25a837bb10a3` |
| Teste de fumaça | `TP: 201 · periciados: 900 · resultado: 3731700.0 · confere: True` |
| Fila × escolha | topo `SIN-26-100637`; índice 0 devolve a **mesma** linha do lote (`True`) |
| Normalização individual | `" sp "` → `"SP"`, escore normalizado == canônico |

Novos testes que travam a melhoria no CI:

| Arquivo | O que protege |
|---|---|
| `tests/test_normalizacao.py` (novo, ~20 testes) | B1/B2/B3, relatório, idempotência, não alterar a entrada, irrecuperáveis intactos, e a falha silenciosa da categoria |
| `tests/test_contrato.py` (atualizado) | bateria adversarial: `"1.234,56"` aceito, `"21.700,00"` aceito, `"1.234"` continua `1.234`, `"abc"`/`"99:99"`/`"Telefone"` recusados |
| `tests/test_app.py` (atualizado) | escolha do sinistro na fila |

***

## Limite desta entrega (o que continua fora do escopo)

O projeto segue sendo **POC**, não produção. Continuam **fora** desta entrega: banco de dados ao
vivo, autenticação e papéis, persistência das decisões do perito (rótulo para re-treino), log de
auditoria e monitoramento vivo/MLOps. A normalização adicionada é **determinística e reportada**,
não aprendida — ela não tenta "adivinhar" o dado.

***

## Como demonstrar no app publicado

1. Abrir **[quilha-triagem.streamlit.app](https://quilha-triagem.streamlit.app/)** (após o push da
   `main`).
2. Enviar o `data/raw/lote_teste_uat.csv` e observar: o **relatório de normalização** (3 valores
   reescritos) e a **recusa** do lote com **3 erros** — representação resolvida, irrecuperável
   barrado.
3. Enviar o `data/raw/lote_teste_uat_valido.csv` e percorrer a **fila**; na aba **Sinistro
   individual**, escolher um sinistro por id ou clicar na linha da fila.

***

## Progresso

| Parte | Descrição | Status |
|---|---|---|
| A | O perito escolhe o sinistro (tela 2) | **concluída** — 137 testes, fumaça intacta |
| B | Normalização determinística com relatório | **concluída** — 166 testes, hash e fumaça intactos |
| C | Encerramento (parecer, README, Progresso) | **concluída** — este documento |

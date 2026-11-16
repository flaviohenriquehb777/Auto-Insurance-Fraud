# MELHORIAS DA POC · Prompt de execução

**Projeto:** Detecção de fraude em sinistros de automóvel — Quilha Seguros
**Escopo:** elevar a POC entregue na Fase 6 a um nível impecável — **sem reabrir o modelo** e
**sem reintroduzir a falha silenciosa**.
**Status:** plano aprovado pelo responsável. **Parte A pendente**, uma parte por vez, com
autorização a cada passo.
**Pré-requisito:** Fase 6 concluída — app publicado em `https://quilha-triagem.streamlit.app/`,
`ruff` e **133 testes** verdes, artefato auditado em `models/pipeline_fraude.joblib`.

---

## Como executar este documento

Para iniciar, basta pedir:

> **"Execute o prompt em `docs/prompts/07_melhorias.md`, parte por parte, começando pela Parte A."**

---

## Fronteira do que NÃO muda (travar antes de qualquer célula)

Esta é uma melhoria de **produto e de fronteira**, não de modelo. Portanto:

| Invariante | Valor que não pode mudar |
|---|---|
| Artefato implantado | SHA-256 `65c7eb5ea1325f4911f9223fa3bf1d2f7aec909e77a30865a47f25a837bb10a3` |
| Teste de fumaça — Campeão @900 | `TP=201` · **R$ 3.731.700,00** |
| Ganho da política | **R$ 4.057.900,00** · IC 95% [R$ 3.537.100; R$ 4.665.500] |
| Suíte de testes | baseline de **133 testes** verdes (só cresce) |
| Regra de interface | "O modelo ordena, a política decide, o perito confirma." |

Se qualquer um destes mudar durante a execução, **parar** e investigar antes de seguir.

---

## Decisões já tomadas (não reabrir)

1. **Ordem:** **Parte A → Parte B.** Primeiro a seleção do sinistro (impacto imediato na
   experiência, risco baixo); depois a normalização determinística.
2. **Datas `dd/mm/aaaa`:** aceitar as **inequívocas** (algum componente > 12) **e** adotar a
   **convenção pt-BR (dia primeiro)** para as ambíguas — **sempre com relatório** do que foi
   interpretado. A origem é brasileira; a convenção é a intenção de quem digita.
3. **O que é "tratar" um erro:** apenas **representação** (formato). **Significado, ambiguidade e
   domínio continuam recusados.**

---

## Princípio que rege as duas partes

> **Representação → normaliza e reporta. Significado, ambiguidade ou domínio → recusa.**

O ganho não é "fazer o modelo adivinhar". É **ampliar o que a fronteira aceita, apenas onde a
correção é inequívoca e auditável**, e **reportar o que foi normalizado** — para que nada seja
corrigido às escondidas.

---

## Parte A — O perito escolhe o sinistro (tela 2)

**Problema real (evidência no código):** hoje a tela "Sinistro individual" com a origem "Da fila
do mês" abre `lote_individual.head(1)` — **a primeira linha do arquivo**, um sinistro arbitrário,
sem relação com a fila vista na tela 1. Além disso, a tela 2 tem **upload próprio**
(`lote_individual`), separado da fila (`lote_fila`) — o rótulo "Da fila do mês" não corresponde
ao comportamento.

**O que construir:**

1. **Uma fonte só de lote e fila** compartilhada entre as telas. Hoje há quatro uploads
   independentes (`lote_fila`, `lote_individual`, `lote_saude`, `lote_drift`); consolidar em um
   lote único na barra lateral, guardado em `st.session_state`, com a fila calculada uma vez e
   reaproveitada.
2. **Seleção explícita do sinistro na tela 2:** o usuário escolhe **qual** sinistro detalhar —
   por `id_sinistro` (busca/`selectbox`) ou por **posição na fila**. O app nunca escolhe por ele.
3. **Clique na linha da fila (tela 1) → detalhe (tela 2):** selecionar uma linha da fila leva o
   `id_sinistro` para a tela 2 via `session_state`.
4. **"Digitar um chamado" permanece** — é o caso legítimo do sinistro que ainda não está no lote.
5. **Testabilidade:** a escolha fica em **funções puras** (`rotulos_da_fila` e
   `escolher_da_fila(lote, fila, indice)`), com testes; a tela permanece fina.

**Critério de saída:** na tela 2, escolher um `id_sinistro` (ou clicar na linha da fila) abre
**aquele** sinistro; o escore, a posição e o SHAP conferem com a fila da tela 1; o
"Digitar um chamado" continua funcionando; `ruff` + `pytest` verdes com os testes novos; **os
números auditados intactos**.

**Entrega (evidência real, 2026-11-16):** `ruff` → *All checks passed!*; `pytest` → **137
passaram** (eram 133: +4 testes da escolha). No lote de teste real: fila de **891** entradas
(`k=891`), topo `#1 · SIN-26-100637 · escore 0,9561`, e a escolha do índice 0 devolveu
exatamente **SIN-26-100637** — a mesma linha que o perito vê na fila. Teste de fumaça intacto:
`TP=201`, `900` perícias, **R$ 3.731.700,00**, `confere=True`.

---

## Parte B — Normalização determinística, com relatório

**Problema real:** a fronteira recusa, hoje, textos que são apenas **outra representação** do
mesmo valor — e recusar representação inequívoca é atrito operacional sem ganho de segurança.

**Regras SEGURAS (cada uma com teste; todas determinísticas e reportadas):**

| # | Regra | Exemplo | Situação atual |
|---|---|---|---|
| B1 | Número pt-BR com **os dois** separadores | `"1.234,56"` → `1234.56` | recusado (hoje só aceita `8837,07` e `8837.07`) |
| B2 | Categórica com **caixa/espaço** | `" sp "` → `"SP"` | recusado (domínio é exato) |
| B3 | Data `dd/mm/aaaa` (convenção pt-BR) | `"15/08/2026"` → `2026-08-15` | recusado (só ISO) |

**Regras que CONTINUAM recusando (por princípio — é isto que prova que o limite está no lugar
certo):**

| Caso | Por quê |
|---|---|
| `idade_segurado = "abc"` | não é representação; não existe número ali |
| `hora_ocorrencia = "99:99"` | não existe 99 horas — coagir = inventar |
| `canal_abertura = "Telefone"` | exigiria **tabela de sinônimos do negócio** (não adotada) |
| número com **um só** ponto (`"1.234"`) | ambíguo (1,234 ou 1.234?) — a Fase 6 já o lê como `1.234`; ver nota |

> **Nota de reversão consciente:** a decisão documentada na Fase 6
> (`docs/prompts/06_implantacao.md`, Parte 1) fixava `"21.700,00"` e `"1.234"` como **não
> tratados**. A regra **B1** reverte isso **apenas para o caso inequívoco** (dois separadores). O
> caso de um só ponto **continua** como está — a reversão é parcial e declarada. Confirmar na
> execução da Parte B.

**Garantia "sem falha silenciosa":**

- O normalizador **devolve um relatório** do que mudou (quantos valores, quais linhas, de → para),
  **exibido na tela**. Nada é corrigido às escondidas.
- O que **não** for normalizável segue para o **contrato**, que recusa com linha, campo, valor e
  motivo — exatamente como hoje.
- **Um só parser de número:** a conversão (estendida para o milhar pt-BR) mora em **um único
  lugar**, importado pela preparação, pelo contrato e pelo normalizador. Nada de duas verdades.

**Onde mexer:** o parser canônico de número (fonte única, já importado pelo contrato); o contrato
(categórica e data); um **novo módulo de normalização** com `normalizar` + relatório; o app
(exibir o relatório); `tests/`.

**B5 — estender o lote de UAT:** acrescentar ao `lote_teste_uat.csv` os casos de representação
(`"1.234,56"`, `" sp "`, `"15/08/2026"`) para **provar** as novas regras, **mantendo** os quatro
irrecuperáveis.

**Critério de saída:** subir o lote de UAT e ver: as três classes de representação **resolvidas
automaticamente com relatório visível**; os quatro irrecuperáveis **ainda recusados** com
mensagem clara; `ruff` + `pytest` verdes; **os números auditados intactos** (teste de fumaça
reproduz R$ 3.731.700).

---

## Parte C — Encerramento

1. **Verificação de integridade:** `ruff check src tests` e `pytest -q` verdes, saída real colada.
2. **Reexecutar o teste de fumaça** e conferir o hash do artefato — os números não mudaram.
3. **Documentos:** atualizar o **parecer** (`docs/06_implantacao.md` ou novo parecer das
   melhorias) e o **README** (como escolher o sinistro e o que a normalização faz); atualizar a
   tabela **Progresso da execução** deste prompt.
4. **Evidência:** capturas/ saídas reais das telas e do relatório de normalização.
5. **Sinalizar** o ponto de **commit/push** (não commitar sem autorização).

---

## Guardrails (invioláveis)

1. **Não tocar no modelo.** Sem re-treino, sem trocar features, sem tocar no artefato.
2. **Falha silenciosa é proibida.** Toda normalização é determinística e **reportada**; o resto
   **recusa** com erro explícito.
3. **Um número, uma fonte.** Um só parser de número para preparação, contrato e normalizador.
4. **Os números auditados não mudam** — hash, R$ 3.731.700 e ganho R$ 4.057.900 são a linha de
   base; qualquer divergência **para** a execução.
5. **Uma etapa por vez**, com autorização a cada passo. Nenhuma decisão sem evidência.
6. **Saída real, nunca inventada.** Se algo não rodou, dizer que não rodou.
7. **Nada construído sobre número não verificado.**

---

## Critérios de aceite

| # | Critério | Evidência exigida |
|---|---|---|
| 1 | Tela 2 abre o sinistro **escolhido** (por id, clique ou posição) | Captura + teste |
| 2 | Lote e fila vêm de **uma fonte** única compartilhada | Código + teste |
| 3 | "Digitar um chamado" continua funcionando | Captura + teste |
| 4 | B1/B2/B3 normalizam com **relatório visível** | Saída real do app |
| 5 | Irrecuperáveis (`abc`, `99:99`, `Telefone`) **ainda recusados** | Saída real do app |
| 6 | Teste por regra + teste de que o lixo continua barrado | Saída do `pytest` |
| 7 | Números auditados **intactos** (fumaça R$ 3.731.700; hash igual) | Saída real |
| 8 | `ruff` e `pytest` verdes com os testes novos | Saída real dos comandos |
| 9 | Parecer, README e Progresso da execução atualizados | Documentos |

---

## Entregáveis

- `docs/prompts/07_melhorias.md` (este documento) e o parecer das melhorias
- `src/quilha/app.py` (seleção do sinistro) e `tests/test_app.py`
- Módulo de normalização + parser estendido + contrato + `tests/`
- `data/raw/lote_teste_uat.csv` estendido (casos de representação)
- README atualizado e tabela **Progresso da execução** completa

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
| A | O perito escolhe o sinistro (tela 2) | **concluída** — 137 testes, fumaça intacta |
| B | Normalização determinística com relatório | pendente |
| C | Encerramento (docs, README, evidência) | pendente |

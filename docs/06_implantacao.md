# Fase 6 — Implantação (CRISP-DM) · Parecer

**Configuração final travada (Parte 0):** o modelo implantado é o `Pipeline` serializado em
[models/pipeline_fraude.joblib](../models/pipeline_fraude.joblib) — **SHA-256
`65c7eb5ea1325f4911f9223fa3bf1d2f7aec909e77a30865a47f25a837bb10a3`** — cujo estimador final é
**XGBoost** (`max_depth=4`, `n_estimators=350`, `learning_rate=0.03`, `subsample=0.85`,
`colsample_bytree=0.7`, `gamma=3`, `min_child_weight=5`, `reg_lambda=1`, `scale_pos_weight=1`,
`random_state=42`), alimentado pelas **18 features canônicas** de `quilha.preparacao.FEATURES`.
Essa configuração **não muda** nesta fase, porque o negócio confirmou que **todas as preditoras
são conhecidas e definitivas no momento da abertura** — não há vazamento temporal (Ramo A).

***

## Parte 0 — Reconciliação e decisão pendente · CONCLUÍDA

### A resposta do gerente de sinistros (registro literal)

> A base `quilha_sinistros.csv` é consolidada e representa a **fotografia exata do sinistro no
> momento da sua abertura** (exceto os campos que claramente envolvem o resultado no fechamento
> do sinistro). Para a construção do modelo preditivo, importa estritamente o que a seguradora
> sabe nesse instante inicial. Alterações que ocorram mais tarde na vida real (como a entrega
> posterior de um documento) não devem ser consideradas na modelagem para evitar o vazamento de
> dados.
>
> - **Dias de apólice:** dado definitivo e matemático — tempo decorrido do início da apólice até
>   a data de abertura.
> - **Terceiro identificado:** registra a declaração feita **no ato do aviso** — Sim, Não ou Não
>   se aplica.
> - **Boletim de ocorrência:** indica a situação do documento **na abertura** — Sim, Não ou em
>   branco (o branco cabe à equipe decidir; ver decisão já tomada abaixo).

### Veredito: **Ramo A — premissa temporal confirmada**

As três preditoras em dúvida são **conhecidas e definitivas no momento da triagem**. Logo:

| Preditora | Situação na abertura | Vazamento | Decisão |
|---|---|---|---|
| `dias_de_apolice` | definitivo e matemático | **Não** | mantida |
| `terceiro_identificado` | declaração no ato do aviso | **Não** | mantida |
| `boletim_ocorrencia` | situação do documento na abertura | **Não** | mantida |

**Ressalva R1 da auditoria da Fase 5: encerrada.** No Ramo A **não há re-treino** e **não é
preciso medir** o custo de remover qualquer dessas preditoras (a medição só existiria no
Ramo B).

**Sobre o `boletim_ocorrencia` em branco** (ponto que o negócio devolveu para nós): a decisão já
está tomada desde a Fase 2 e permanece — o branco vira a **categoria própria "Não informado"** e
**nunca é imputado**. O branco carrega sinal; imputá-lo o apagaria.

### Reauditoria do artefato (saída real, nada digitado à mão)

| Verificação | Resultado |
|---|---|
| SHA-256 do `.joblib` | `65c7eb5ea1325f4911f9223fa3bf1d2f7aec909e77a30865a47f25a837bb10a3` — confere com o auditado |
| Tamanho do arquivo | 449.791 bytes |
| Etapas do Pipeline | `preparo` → `imputacao` → `modelo` |
| Estimador final | `xgboost.sklearn.XGBClassifier` |
| Hiperparâmetros | `max_depth=4`, `n_estimators=350`, `learning_rate=0.03`, `subsample=0.85`, `colsample_bytree=0.7`, `gamma=3`, `min_child_weight=5`, `reg_lambda=1`, `scale_pos_weight=1`, `random_state=42` |
| Features no artefato | **18**, na ordem canônica (idêntica a `FEATURES`) |
| Teste de fumaça — Campeão @900 (ago/2026) | `TP=201` · `FP=699` · `FN=80` · precisão 22,33% · cobertura 71,53% · **R$ 3.731.700,00** |
| Cross-check — Regra atual (> R$ 50 mil) @900 | 927 acima do limiar, 900 periciados · `TP=14` · **−R$ 326.200,00** |
| Ganho da política sobre a regra atual | **R$ 4.057.900,00** |

O teste de fumaça **reproduz exatamente** o número auditado na Fase 5. A configuração está
travada.

### Consequências para o restante da fase

1. **Sem re-treino** — o artefato auditado é o artefato implantado.
2. **A Parte 1 (blindagem da fronteira) está liberada** para começar.
3. O app (Parte 2) e o notebook consolidado (Parte 3) partem deste mesmo hash.

***

## Parte 1 — Blindagem da fronteira (camadas 2 e 3 da vírgula) · CONCLUÍDA

A porta única de entrada sempre foi `pipeline.predict`; o que faltava era **fechar a porta a dado
inválido em vez de prever errado**. Duas camadas novas, ambas comentadas em português:

| Camada | Onde vive | O que faz |
|---|---|---|
| 2 — contrato na fronteira | [src/quilha/contrato.py](../src/quilha/contrato.py) | Valida os **14 campos de abertura**; se um número não converte, **recusa com erro explícito** e diz qual linha do lote falhou. **Nunca imputa na fronteira.** |
| 3 — alarme no silêncio | [src/quilha/monitoramento.py](../src/quilha/monitoramento.py) | Mede a **fração imputada por feature** a cada lote; acima de **5%** dispara alerta de *"problema de dado, não de fraude"*. |

### Bateria adversarial (saída real, nada digitado à mão)

| Entra em `valor_reclamado` | `_numero` devolve | Veredito do contrato |
|---|---|---|
| `'8837,07'` | 8837.07 | **ACEITO** |
| `'8837.07'` | 8837.07 | **ACEITO** |
| `'21.700,00'` | `<NA>` | **RECUSADO** — não converte (formatos aceitos: `'8837,07'` ou `'8837.07'`) |
| `'R$ 100,00'` | `<NA>` | **RECUSADO** — não converte |
| `'1.234'` | 1.234 | **ACEITO** (risco de 1.000× documentado) |
| `''` / `None` | `<NA>` | **RECUSADO** — vazio: o contrato não imputa |

Outros campos: `data_abertura='13:53'` e `'18/07/2026'` → **recusados** (só ISO `AAAA-MM-DD`);
`hora_ocorrencia='99:99'` → **recusado**; `uf='XX'`, `tipo_sinistro='Batida'`,
`canal_abertura=''` → **recusados**; `boletim_ocorrencia=''` → **ACEITO** (branco é caso de
negócio e vira "Não informado"); coluna ausente → erro `'(ausente)'` com o nome do campo.

### Monitor de imputação (camada 3)

| Cenário | Resultado real |
|---|---|
| Mês de teste real (11.904 sinistros) | **0 imputações** em todas as features — alertas `[]` |
| Lote com `valor_reclamado` em branco (100 sinistros) | **3 features** sinalizadas a **100%** (`valor_reclamado`, `razao_reclamado_fipe`, `faixa_valor_reclamado`) |

Um único branco em `valor_reclamado` contamina as **três features** que dependem dele — por isso o
monitor denuncia o campo de origem, não só a coluna derivada.

### Achado real que mudou o desenho

`pd.to_datetime("13:53", format="mixed")` **não** devolve `NaT`: resolve para **hoje às 13:53**; e
`"03/04/2026"` seria lido como **4 de março**. Era uma **falha silenciosa de verdade** (data virava
o dia da execução). Correção: o contrato aceita **apenas** data ISO explícita `AAAA-MM-DD` ou um
objeto `datetime.date`/`Timestamp`.

### Critério de saída — atendido

| Critério do prompt | Resultado |
|---|---|
| `'21.700,00'` ou `'R$ 100,00'` recusado com erro explícito | **atendido** (linha a linha, acima) |
| `'8837,07'` aceito e prevê corretamente | **atendido** (payload de 14 colunas prevê igual à linha crua completa) |
| Teste que trava o contrato no CI | **atendido** — [tests/test_contrato.py](../tests/test_contrato.py), **49 testes** |
| Monitor imprime a fração por feature | **atendido** (tabela acima) |

Verificação de qualidade: `ruff check src tests` → **All checks passed!**; `pytest -q` →
**106 passed in 25.60s**.

***

## Progresso

| Parte | Descrição | Status |
|---|---|---|
| 0 | Reconciliação e decisão pendente | **concluída** — Ramo A |
| 1 | Blindagem da fronteira (camadas 2 e 3 da vírgula) | **concluída** — contrato + monitor, 49 testes |
| 2 | O app de triagem (Streamlit) | pendente |
| 3 | O notebook consolidado único | pendente |
| 4 | O resumo executivo `.docx` | pendente |
| 5 | Verificação final e encerramento | pendente |

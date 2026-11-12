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

## Parte 2 — O app de triagem (Streamlit) · CONCLUÍDA

O app **não decide fraude**: ele **prioriza** a fila, **explica** por que um sinistro subiu e
**mostra o valor** da política. A decisão de periciar é humana — a mensagem permanente de todas
as telas é *"O modelo ordena, a política decide, o perito confirma."*

| Arquivo | Papel |
|---|---|
| [src/quilha/app.py](../src/quilha/app.py) | Toda a lógica como **função pura** (testável sem UI) + as três telas em `main()`. O `streamlit` só é importado dentro de `main()`. |
| [app.py](../app.py) | Lançador da raiz (`streamlit run app.py`): põe `src/` no caminho de importação, sem regra de negócio. |
| [tests/test_app.py](../tests/test_app.py) | **20 testes** de nível de função — porta, fila, linguagem e número. |

### As três telas

1. **Fila do mês** — recebe o lote, passa pelo contrato (camada 2), pontua, ordena por escore e
   corta em `k`; exporta a fila em CSV.
2. **Sinistro individual** — escore, posição na fila, faixa de prioridade e os sinais SHAP em
   linguagem de priorização.
3. **Valor e governança** — o ganho em reais com IC, o valor por sinistro, a saúde do modelo
   (hash + imputações) e os gatilhos de circuit breaker.

### Decisões de implantação travadas (levadas ao resumo da diretoria)

| # | Decisão | Razão de mundo real |
|---|---|---|
| **D1** | Lote com linha inválida **recusa o lote inteiro** (*fail closed*) | A fila é uma ordenação sobre o **universo fechado do mês**: omitir linhas em silêncio pode omitir a fraude. O app devolve a **lista de erros por linha** (campo, valor, motivo) em CSV para correção rápida. |
| **D2** | Semáforo **por posição, nunca por culpa** | 🔴 = dentro da capacidade (`posição ≤ k`); 🟡 = até `2 × k` (capacidade do mês seguinte); 🟢 = o resto. É posição em fila de trabalho, não veredito sobre a pessoa. |
| **D3** | Valor por sinistro como **faixa**, com a **taxa observada** ao lado | A probabilidade do campeão **não é calibrada** (Brier 0,014702 sem calibração vs 0,014688 com isotonic — ganho de 0,000011; sigmoid piora para 0,015210). O número aparece como estimativa, ancorado na taxa observada da faixa de escore. |
| **D4** | Capacidade **proporcional ao lote**: `k = 900 × n / 12.024` | Mantém a **taxa de perícia** (900/12.024 = 7,485%) constante. É o que preserva a economia auditada: periciar mais fundo na fila derruba a precisão abaixo dos 22,33% medidos. |

### Evidência real (saída do `pytest` e do smoke test, nada digitado à mão)

| Verificação | Resultado real |
|---|---|
| `ruff check src tests` | **All checks passed!** |
| `pytest tests/test_app.py -q` | **20 passed in 3.45s** |
| `pytest -q` (suíte completa) | **126 passed in 17.87s** |
| Fila do mês de teste (11.904 sinistros) | ordenada pelo escore; corte em `k = 891`; topo = `SIN-26-100637` (escore 0,956084); **201 fraudes** nas primeiras posições (**22,56%** de acerto) |
| Sinistro individual (topo) | escore 0,956084 · valor bruto R$ 20.047,02 · faixa 0,1802 com **taxa observada 17,88%** → R$ 3.180,86 · piso 3,2258% |
| A porta (lote com `valor_reclamado='21.700,00'` e `uf='ZZ'`) | **42 erros** de contrato listados por linha/campo/valor; **nenhuma** predição emitida |
| Conferência do artefato | `TP=201`, `periciados=900`, **R$ 3.731.700,00**, `confere=True` |
| Hash do `.joblib` implantado | `65c7eb5e…bb10a3` |
| Saúde e drift | 0 imputações e `alertas=[]` no mês de teste; PSI ≤ 0,0079 em todas as features (**estável**) |

### Um ponto de atenção honesto: 891 × 900

A tela de fila aplica **D4** e corta o lote de agosto (11.904 sinistros, ~1% abaixo do mês de
referência de 12.024) em **891 perícias**; a **conferência do artefato** usa `k = 900`, que é a
capacidade cheia com que a Fase 5 mediu o `Resultado@900`. Os dois números **não se contradizem**:
o primeiro é a fila **operacional** proporcional ao lote; o segundo é a **reprodução do número
auditado**. O topo da fila é idêntico nos dois casos (as 201 fraudes estão nas primeiras
posições), então o acerto não muda — só as 9 últimas perícias do corte. A decisão de manter a
proporcionalidade é deliberada: é ela que conserva a **taxa** de perícia e, com ela, a economia
auditada.

### Critério de saída — atendido

| Critério do prompt | Resultado |
|---|---|
| O app sobe local e prioriza uma fila de exemplo | **atendido** (fila do mês de teste acima) |
| Mostra escore + SHAP + faixa para um sinistro individual | **atendido** (escore, posição, faixa e sinais SHAP) |
| Exibe o painel de valor com os números reais | **atendido** (R$ 3.731.700 × −R$ 326.200, IC e conferência `confere=True`) |
| **Sem** afirmar fraude em nenhum ponto | **atendido** — teste `test_rotulos_e_mensagem_permanente_nao_acusam_ninguem` |

***

## Progresso

| Parte | Descrição | Status |
|---|---|---|
| 0 | Reconciliação e decisão pendente | **concluída** — Ramo A |
| 1 | Blindagem da fronteira (camadas 2 e 3 da vírgula) | **concluída** — contrato + monitor, 49 testes |
| 2 | O app de triagem (Streamlit) | **concluída** — 3 telas, 20 testes, 126 na suíte |
| 3 | O notebook consolidado único | pendente |
| 4 | O resumo executivo `.docx` | pendente |
| 5 | Verificação final e encerramento | pendente |

# Fase 2 — Entendimento dos Dados

> **Projeto:** Detecção de fraude em sinistros de automóvel — Quilha Seguros
> **Fase CRISP-DM:** 2 de 6 — Entendimento dos Dados
> **Base:** `data/raw/quilha_sinistros.csv` (60.000 sinistros, 17 colunas, 01/04/2026 a 31/08/2026)
> **Notebook:** `notebooks/executed/02_entendimento_dados.ipynb` (executado, com saídas)
> **Evidências visuais:** `reports/figures/02_*.png`
> **Data:** 2026-11-03 — **Versão:** 1.0.0

---

## 1. Carga e dimensões

| Verificação | Resultado |
|---|---|
| Linhas | 60.000 |
| Colunas | 17 |
| Conforme Anexo 2 do case | Sim (`shape == (60000, 17)`) |
| Duplicatas de `id_sinistro` | 0 |
| Linhas inteiramente duplicadas | 0 |

A base foi lida com `dtype=str` na primeira passada, de propósito: assim o formato original de
cada coluna é auditado antes de qualquer conversão. Se deixássemos o pandas inferir,
`valor_reclamado` (única coluna com vírgula decimal) viraria texto silenciosamente.

## 2. Tipagem e dicionário de dados

Conversões aplicadas (todas em código, o CSV não foi editado):

| Coluna | Formato original | Tratamento | Tipo final |
|---|---|---|---|
| `valor_reclamado` | `20387,95` (vírgula decimal) | vírgula → ponto, depois decimal | `float64` |
| `valor_fipe`, `valor_indenizado` | ponto decimal | decimal | `float64` |
| `data_ocorrencia`, `data_abertura` | `AAAA-MM-DD` | texto → data | `datetime64` |
| `idade_segurado`, `dias_de_apolice`, `sinistros_12m`, `idade_veiculo`, `fraude` | inteiros | texto → inteiro | `Int64` |
| `hora_ocorrencia` | `HH:MM` | mantido texto; **100% no padrão válido (60.000/60.000)** | texto |
| Categóricas (`canal_abertura`, `tipo_sinistro`, `uf`, `boletim_ocorrencia`, `terceiro_identificado`) | texto | **sem espaços nas pontas, sem variação de caixa** | texto |

**Controle de qualidade da conversão:** foram comparados os nulos de antes e depois da tipagem.
Nenhum valor virou nulo por erro de formato.

## 3. Variável-alvo

| Classe | Contagem | % |
|---|---|---|
| 0 — Legítimo | 58.520 | 97,53% |
| 1 — Fraude | 1.480 | 2,47% |

- **Taxa de fraude da base:** 2,47%
- **Proporção:** 1 fraude confirmada para **39,54** sinistros legítimos (≈ 1 : 40)
- **Sinistros por mês:** 12.000 · **Fraudes por mês:** **296,0**

## 4. A acurácia do "modelo de uma linha"

Um classificador que responde "ninguém frauda" para todo sinistro acerta toda a classe majoritária:

| | Valor |
|---|---|
| Acertos | 58.520 |
| Fraudes ignoradas | 1.480 |
| **Acurácia** | **97,53%** |

Sem aprender absolutamente nada, esse "modelo" já entrega 97,53% de acurácia. Acurácia está
descartada como métrica de decisão neste projeto.

## 5. Métricas do projeto

| Métrica | O que mede | Cuidado principal |
|---|---|---|
| Acurácia | Acerto global | **Inutilizável:** o modelo ingênuo já marca 97,53% |
| ROC AUC | Probabilidade de dar nota maior a um fraudador que a um legítimo; independe do corte | Otimista em evento raro: a taxa de falsos positivos é diluída por 58.520 legítimos |
| PR AUC (`average_precision_score`) | Equilíbrio precisão × cobertura em todos os cortes | A referência **não é 0,5**: é a taxa base (2,47%) |
| Precisão@900 | Dos 900 periciados, quantos eram fraude | É a métrica que vira dinheiro sob capacidade fixa |
| Cobertura@900 | Das fraudes existentes, quantas entraram na fila | Limitada pela capacidade mensal |
| `Resultado(k)` | 21.700 × fraudes periciadas − 700 × k | Depende do custo unitário e da premissa de perícia infalível |

## 6. Valores nulos

| Coluna | Qtd. | % da base |
|---|---|---|
| `boletim_ocorrencia` | 2.899 | 4,83% |

**Única coluna com nulos na base.** Não houve necessidade de correção de tipo — o campo em
branco é uma informação ausente de origem, não um erro de carga.

## 7. Impacto de descartar as linhas com nulo

| | Valor |
|---|---|
| Sinistros descartados | 2.899 (4,83% da base) |
| Fraudes perdidas | **94 (6,35% de todas as fraudes)** |
| Custo dessas fraudes, se pagas | R$ 2.039.800,00 |

**Métrica disponível:** nº de sinistros. **Não é possível contar clientes em sentido estrito** —
a base identifica o sinistro (`id_sinistro`), não o segurado (não há CPF). Perder 4,83% das linhas
significaria perder 6,35% das fraudes: descarte é desproporcionalmente caro porque a fraude é
mais frequente justamente onde há campo em branco.

## 8. Outros problemas de qualidade encontrados

### 8.1 Datas

| Verificação | Resultado |
|---|---|
| `data_abertura` fora de 01/04/2026–31/08/2026 | 0 |
| `data_ocorrencia` anterior a 01/04/2026 | 688 (mín. 2026-03-02) |
| `data_ocorrencia` posterior a `data_abertura` | 0 |
| Atraso ocorrência → abertura | mín. 0 · mediana 1 · máx. 45 dias |

As 688 ocorrências fora da janela são **legítimas**: o evento aconteceu em março e a abertura
caiu dentro da janela auditada. A coluna de controle da divisão treino/teste é `data_abertura`.

### 8.2 Valores monetários

| Verificação | Resultado |
|---|---|
| Valores negativos (qualquer coluna monetária) | 0 |
| `valor_reclamado` = 0 | 0 (mín. R$ 184,08) |
| `valor_indenizado` = 0 | **8.066 (13,44%)** |
| `valor_indenizado` > `valor_reclamado` | 0 |
| `valor_reclamado` > `valor_fipe` | 117 (0,19%) |

### 8.3 Faixas numéricas

| Coluna | Mín | Máx | Implausíveis |
|---|---|---|---|
| `idade_segurado` | 18 | 85 | 0 |
| `dias_de_apolice` | 0 | 5.000 | 0 |
| `sinistros_12m` | 0 | 4 | 0 |
| `idade_veiculo` | 0 | 25 | 0 |

Nenhum valor fora de faixa plausível. Não há outliers que exijam tratamento por regra de domínio.

### 8.4 Categorias

| Coluna | Categorias | Padronização |
|---|---|---|
| `canal_abertura` | 3 (App 44,60% · Corretor 34,83% · Central 0800 20,57%) | limpa |
| `tipo_sinistro` | 6 (Colisão 52,40% · Vidros 19,95% · Roubo/Furto 9,85% · Terceiros 9,74% · Alagamento 5,00% · Incêndio 3,06%) | limpa |
| `uf` | 27 UFs (SP 28,40%) | limpa |
| `boletim_ocorrencia` | Sim 48,55% · Não 46,62% · **em branco 4,83%** | limpa |
| `terceiro_identificado` | Sim 42,70% · Não se aplica 37,86% · Não 19,45% | limpa |

### 8.5 `terceiro_identificado` é estruturalmente ditada por `tipo_sinistro`

| `tipo_sinistro` | Não | Não se aplica | Sim |
|---|---|---|---|
| Alagamento | 0 | 3.002 | 0 |
| Colisão | 11.668 | 0 | 19.775 |
| Incêndio | 0 | 1.835 | 0 |
| Roubo/Furto | 0 | 5.908 | 0 |
| Terceiros | 0 | 0 | 5.842 |
| Vidros | 0 | 11.970 | 0 |

A coluna é **redundante por construção**: só carrega informação dentro de "Colisão" (Sim vs. Não);
nos demais tipos é constante ("Não se aplica" ou "Sim"). Tratar na Fase 3.

### 8.6 `valor_indenizado` entrega o alvo (vazamento)

| Grupo | Sinistros | Taxa de fraude |
|---|---|---|
| `valor_indenizado` = 0 | 8.066 (13,44%) | **17,37%** |
| `valor_indenizado` > 0 | 51.934 (86,56%) | **0,15%** |

Fraude confirmada implica pagamento negado. A coluna é preenchida **no encerramento** —
não existe no momento da decisão — e separa as classes com 115× de diferença de taxa.
**Vazamento confirmado. Fora do modelo.**

## 9. Estatística descritiva

| Variável | Mín | 25% | Mediana | 75% | Máx | Média | Desvio | Assimetria |
|---|---|---|---|---|---|---|---|---|
| `idade_segurado` | 18,00 | 35,00 | 44,00 | 53,00 | 85,00 | 44,19 | 12,75 | 0,11 |
| `dias_de_apolice` | 0,00 | 365,00 | 882,00 | 1.456,00 | 5.000,00 | 1.059,65 | 869,00 | 1,50 |
| `sinistros_12m` | 0,00 | 0,00 | 0,00 | 0,00 | 4,00 | 0,22 | 0,47 | 2,17 |
| `idade_veiculo` | 0,00 | 3,00 | 5,00 | 8,00 | 25,00 | 6,15 | 4,11 | 1,24 |
| `valor_fipe` | 15.000,00 | 39.800,00 | 58.500,00 | 83.800,00 | 420.000,00 | 66.404,75 | 37.879,94 | 1,66 |
| `valor_reclamado` | 184,08 | 2.220,61 | 5.824,76 | 15.527,51 | 416.670,02 | 15.374,31 | 24.897,06 | 3,51 |
| `valor_indenizado` | 0,00 | 721,98 | 2.741,18 | 10.961,65 | 416.670,02 | 12.455,42 | 24.568,12 | 3,71 |

Assimetria alta em `valor_reclamado` (3,51) e `valor_indenizado` (3,71): distribuição de cauda
longa à direita. Relevante na seleção de modelos e na transformação de atributos.

## 10. Evidências visuais

| Arquivo | Conteúdo |
|---|---|
| `02_01_distribuicao_alvo.png` | Contagem e percentual de cada classe |
| `02_02_distribuicao_valor_reclamado.png` | Distribuição do valor reclamado (linear e log10) |
| `02_03_campos_nulos.png` | Percentual de campos em branco por coluna |
| `02_04_valor_reclamado_por_classe.png` | Valor reclamado por classe (log, sem outliers) |

## 11. Achados que governam as próximas fases

1. **A capacidade de perícia não é o gargalo.** São **296 fraudes por mês** para **900 vagas**
   mensais — as fraudes usam apenas **32,9% da capacidade**. O gargalo é **saber onde elas estão**,
   não ter peritos.
2. **O piso de viabilidade (3,23%) está acima da taxa base (2,47%).** Periciar ao acaso dá
   prejuízo; todo o valor da política vem de **concentrar** os 900 slots no topo da fila.
3. **Acurácia está vetada.** O modelo ingênuo marca 97,53%.
4. **`valor_indenizado` está fora**, por vazamento comprovado (17,37% vs. 0,15%).
5. **Descartar linhas com nulo é caro:** 4,83% dos sinistros carregam 6,35% das fraudes.

## 12. Decisões pendentes para a Fase 3 (nada decidido nesta fase)

| Tema | Decisão necessária |
|---|---|
| `valor_indenizado` | Excluir formalmente do conjunto de atributos (evidência na seção 8.6) |
| `boletim_ocorrencia` em branco | Categoria própria ("Não informado") vs. imputação — a Pergunta 2 do case cobra a decisão |
| `terceiro_identificado` | Representação alternativa após a constatação da seção 8.5 |
| `hora_ocorrencia` | Extrair atributos (madrugada, fim de semana) ou descartar |
| `uf` e `idade_segurado` | Teste com e sem as duas colunas, com ROC AUC e PR AUC (exigência da Pergunta 1) |
| `id_sinistro` | Chave de controle; nunca entra no modelo |

---

## Histórico de revisões

| Versão | Data | Descrição |
|---|---|---|
| 1.0.0 | 2026-11-03 | Versão inicial da Fase 2 — Entendimento dos Dados. |

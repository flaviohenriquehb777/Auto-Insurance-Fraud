# Fase 1 — Entendimento do Negócio

> **Projeto:** Priorização da Fila de Perícia com Modelos de Classificação em Sinistros de Automóvel
> **Cliente (caso):** Quilha Seguros
> **Fase CRISP-DM:** 1 de 6 — Entendimento do Negócio
> **Data:** 2026-11-03
> **Versão:** 1.0.0
> **Fonte de verdade:** `docs/case/Auto Insurance Fraud.md` (Anexos 1 a 3)

---

## 1. Contexto de negócio

A Quilha Seguros é uma seguradora brasileira de médio porte, especializada em seguro de automóvel para pessoas físicas. Protege cerca de **410 mil veículos** e recebe em média **12 mil avisos de sinistro por mês**.

Nos últimos dois anos a sinistralidade subiu acima do crescimento da frota, sem explicação por clima, trânsito ou preço de peças. A suspeita é fraude. Para medir o problema, a diretoria contratou uma auditoria externa que revisou todos os sinistros abertos entre **01/04/2026 e 31/08/2026** e atribuiu veredito a cada um. Pela primeira vez a empresa possui uma base rotulada sinistro a sinistro.

A oportunidade é usar essa base para **decidir quem vai para a perícia**.

## 2. Problema de negócio

A Quilha combate a fraude com perícia: um perito visita o veículo, confere o local, cruza documentos e libera ou nega o pagamento. A perícia é eficaz, mas **escassa**.

A empresa tem **900 perícias por mês** (30 peritos × 30 perícias) para um fluxo de 12.000 sinistros/mês — capacidade de apenas **7,5%** do fluxo.

A regra vigente é **cega a risco**: pericia todo sinistro com valor reclamado acima de R$ 50 mil (~950 pedidos/mês), atendidos por ordem de valor até o limite de 900.

**O problema de negócio:** ordenar os sinistros pela chance de fraude **no momento da abertura** e converter essa ordem em uma política de perícia que gere saldo líquido positivo dentro do limite rígido de 900 perícias/mês.

## 3. Tradução analítica

O problema **não** é "classificar fraude". É um problema de **ranking sob restrição rígida de capacidade** — a decisão não é *se* periciar, é *quais* 900 periciar.

| Implicação | Justificativa |
|---|---|
| O modelo precisa **ordenar**, não decidir | Um limiar fixo de probabilidade (ex.: 0,5) ignora que existem exatamente 900 vagas. A decisão operacional é "top-900", não "probabilidade acima do corte". |
| **PR AUC** serve para escolher o modelo | É livre de limiar e compara modelos de forma justa. Em evento raro, a referência é a **taxa base da fraude**, não 0,5. |
| **Precision@900** decide o dinheiro | Com k fixo, maximizar verdadeiros positivos equivale a maximizar a precisão no topo da fila. |

## 4. Modelo de decisão

```
Resultado(k) = 21.700 · (fraudes periciadas) − 700 · k        s.a. k ≤ 900/mês
```

Valor marginal da k-ésima perícia:

```
21.700 · p − 700 ≥ 0   ⟺   p ≥ 1/31 = 3,226%
```

**Piso de viabilidade: uma perícia só se paga se a chance de fraude daquele sinistro superar 3,23%.**

## 5. Parâmetros econômicos (Anexo 1 — valores do enunciado)

| Parâmetro | Valor |
|---|---|
| Sinistros abertos por mês | ~12.000 |
| Capacidade de perícia | 900/mês |
| Custo de uma perícia | R$ 700 (com ou sem fraude) |
| Custo de uma fraude paga sem perícia | R$ 21.700 (média) |
| Regra atual | valor > R$ 50 mil → ~950 pedidos/mês, cortado em 900 |
| Premissa | havendo fraude, a perícia **sempre** a encontra e o pagamento é negado |

Proporção de perda: **21.700 / 700 = 31,0** — um falso negativo custa 31 vezes um falso positivo.

## 6. Alertas e riscos identificados

**(a) Não contar os 7 dias de atraso duas vezes.** O Anexo 1 define o R$ 700 como composto por "perito, deslocamento, atraso no pagamento". O atraso **já está dentro** do custo unitário. Somá-lo novamente infla artificialmente a economia.

**(b) "Tornar o saldo positivo" pode ser premissa falsa.** A regra atual pode **já ser lucrativa**. O critério verdadeiro é **superar a regra atual**, não apenas ficar positivo — ficar positivo e perder para a regra vigente é fracasso do projeto.

**(c) Se a taxa de fraude for menor que 7,5%, o gargalo não é o modelo.** 900 vagas para ~720 fraudes/mês (hipótese de taxa 6%) significaria que a equipe atual já basta, e a resposta da Pergunta 5 mudaria por completo. **Hipótese a testar na Fase 2.**

**(d) Unidade de tempo na avaliação.** A capacidade é mensal (900/mês), mas o conjunto de teste será um **período**. Se o teste não corresponder a exatamente um mês, a comparação de política deve ser escalada proporcionalmente, sob pena de produzir números sem validade.

**(e) `valor_indenizado` é candidato a vazamento.** A coluna é preenchida no **encerramento** do sinistro e não existe no momento da decisão. Não pode entrar no modelo (decisão formal na Fase 2).

## 7. Critérios de sucesso

| Tipo | Critério |
|---|---|
| **Técnico** | PR AUC no teste superior à taxa base da fraude (baseline do `DummyClassifier`), com ganho material; sem evidência de sobreajuste (comparação treino × teste). |
| **Negócio** | `Resultado(900)` no teste **> `Resultado` da regra atual com 900 perícias** e **> 0**. |
| **Complementar** | Ganho mensal positivo ao elevar de 900 para 1.800 perícias — valida ou refuta a tese do Otávio. |

O limiar numérico de "ganho material" será fixado na Fase 2, após a medição da taxa base da fraude. Fixá-lo antes seria arbitrário.

## 8. Premissas herdadas do enunciado

1. Perícia sempre encontra a fraude quando ela existe → **a perícia não possui falso negativo**.
2. Custo da fraude paga: R$ 21.700 (usar o valor do enunciado, mesmo que pareça irreal).
3. Toda informação usada pelo modelo deve existir **no instante da abertura** do sinistro.
4. A base entregue é a base válida, sem alterações: todo tratamento é feito em código.

## 9. Escopo

| Dentro | Fora |
|---|---|
| Priorização da fila de perícia | Precificação do seguro |
| Política de alocação das 900 perícias | Retenção e renovação de clientes |
| Dimensionamento da equipe de perícia | Precificação de oficinas e negociação com corretores |

## 10. Rastreio: fase CRISP-DM → pergunta da diretoria

| Fase | Pergunta respondida |
|---|---|
| 2 — Entendimento dos Dados | P1 (seleção de colunas e vazamento) e P2 (distribuições) |
| 3 — Preparação dos Dados | insumo para P3 |
| 4 — Modelagem | P3 (modelos e baselines) |
| 5 — Avaliação | P4 (política e economia) e P5 (dimensionamento da equipe) |
| 6 — Implantação | entrega operacional |

## 11. Baselines obrigatórios (definidos na Fase 1, medidos na Fase 4)

| Baseline | Tipo | Papel |
|---|---|---|
| Regra atual (top 900 por valor > R$ 50 mil) | **Negócio** | Piso a ser superado; se o modelo não vencer esta regra, não há valor em implantá-lo. |
| `DummyClassifier` (mesma nota para todos) | **Técnico** | Prova que o algoritmo aprende padrão real e não apenas a classe majoritária. Referência de PR AUC = taxa base. |

---

## Histórico de revisões

| Versão | Data | Descrição |
|---|---|---|
| 1.0.0 | 2026-11-03 | Versão inicial da Fase 1 — Entendimento do Negócio. |

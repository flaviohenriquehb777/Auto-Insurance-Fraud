"""Avaliação da política de perícia — Fase 5 do CRISP-DM (a etapa de decisão).

Esta camada responde às Perguntas 4 e 5 do comitê e decide o que fazer com a regra
atual de R$ 50 mil e com a equipe de perícia. Ela **não treina modelo**: carrega o
campeão aprovado na Fase 4 e o converte em política.

Conceitos que governam o módulo
-------------------------------
* **O modelo ordena; a política decide.** A política é o par (escore, capacidade):
  ordena os sinistros pelo escore e corta em `k`.
* **A capacidade é restrição de negócio, não escolha do analista.** Nada aqui
  recomenda periciar além da capacidade mensal sem que a decisão de ampliar a
  equipe seja explícita e precificada.
* **O critério é o resultado líquido**, `custo_fraude × TP − custo_pericia × k`.
  Fraude detectada é meio, nunca fim.

Os parâmetros econômicos vêm do Anexo 1 do case e são **importados** de
`quilha.modelagem` de propósito: uma única fonte de verdade impede que a Fase 5
divirja da Fase 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from quilha.dados import PASTA_FIGURAS
from quilha.modelagem import (
    CAPACIDADE_MENSAL,
    CUSTO_FRAUDE_PAGA,
    CUSTO_PERICIA,
    LIMIAR_CORTE_TEORICO,
    LIMIAR_REGRA_ATUAL,
    SEED,
    SINISTROS_POR_MES,
    RegraAtualQuilha,
    calcular_k_proporcional,
)
from quilha.pipeline import construir_pipeline

# ---------------------------------------------------------------------------
# Parâmetros da fase
# ---------------------------------------------------------------------------
# Reamostragens do bootstrap que produz o intervalo de confiança do resultado
# líquido no mês de teste (Parte 5).
N_REAMOSTRAGENS_BOOTSTRAP = 2_000

# Grades da análise de sensibilidade (um fator por vez e depois combinados).
GRADE_CUSTO_FRAUDE_PAGA: tuple[float, ...] = (10_000.0, 15_000.0, 21_700.0, 30_000.0, 40_000.0)
GRADE_CUSTO_PERICIA: tuple[float, ...] = (500.0, 700.0, 1_000.0, 1_500.0)
GRADE_CAPACIDADE: tuple[int, ...] = (600, 900, 1_200, 1_800, 2_400)

# Faixas etárias da triagem de disparidade (Parte 6). Intervalos fechados: (rótulo, mín, máx).
FAIXAS_IDADE_DISPARIDADE: tuple[tuple[str, int, int], ...] = (
    ("18 a 29", 18, 29),
    ("30 a 44", 30, 44),
    ("45 a 59", 45, 59),
    ("60 ou mais", 60, 200),
)

# Colunas usadas na triagem de disparidade. Os nomes são os do chamado cru.
COLUNA_IDADE_DISPARIDADE = "idade_segurado"
COLUNA_UF_DISPARIDADE = "uf"

# Pergunta que encerra a fase: a recomendação só é afirmada se sobreviver a ela.
PERGUNTA_DE_DECISAO = (
    "A política baseada no escore do campeão supera a regra atual de R$ 50 mil com as "
    "mesmas 900 perícias — e esse ganho sobrevive ao intervalo de confiança de 95% da "
    "diferença e à variação dos parâmetros econômicos do Anexo 1?"
)


# ---------------------------------------------------------------------------
# Matemática da decisão
# ---------------------------------------------------------------------------
def piso_viabilidade(
    custo_pericia: float = CUSTO_PERICIA,
    custo_fraude_paga: float = CUSTO_FRAUDE_PAGA,
) -> float:
    """Probabilidade mínima de fraude para uma perícia se pagar.

    Uma perícia custa `custo_pericia` e evita `custo_fraude_paga` quando o sinistro
    é fraude. O valor esperado da k-ésima perícia zera em:

        custo_fraude_paga × p − custo_pericia = 0  ⟺  p* = custo_pericia / custo_fraude_paga

    Com os valores do Anexo 1: `700 / 21.700 = 3,2258%`.
    """
    return custo_pericia / custo_fraude_paga


def resultado_liquido(
    tp: int,
    k: int,
    custo_pericia: float = CUSTO_PERICIA,
    custo_fraude_paga: float = CUSTO_FRAUDE_PAGA,
) -> float:
    """Resultado líquido da política no período: perda evitada menos custo das perícias.

    `tp` é o número de fraudes periciadas — por premissa do Anexo 1, a perícia sempre
    encontra a fraude quando ela existe — e `k` é o número de perícias executadas.
    """
    return custo_fraude_paga * tp - custo_pericia * k


def custo_sem_pericia(fn: int, custo_fraude_paga: float = CUSTO_FRAUDE_PAGA) -> float:
    """Custo de não periciar ninguém: cada fraude não detectada é paga integralmente."""
    return custo_fraude_paga * fn


def resumo_parametros_decisao() -> dict[str, float | int | str]:
    """Bloco travado da Parte 0: parâmetros do case, piso e pergunta de decisão.

    Devolve um dicionário (e não um texto impresso) para que o mesmo conteúdo sirva
    ao documento da fase, ao notebook e aos testes — sem número digitado à mão.
    """
    return {
        "capacidade_mensal": CAPACIDADE_MENSAL,
        "sinistros_por_mes": SINISTROS_POR_MES,
        "custo_pericia": CUSTO_PERICIA,
        "custo_fraude_paga": CUSTO_FRAUDE_PAGA,
        "razao_custo_fraude_pericia": CUSTO_FRAUDE_PAGA / CUSTO_PERICIA,
        "limiar_regra_atual": LIMIAR_REGRA_ATUAL,
        "piso_viabilidade": piso_viabilidade(),
        "limiar_corte_teorico_fase4": LIMIAR_CORTE_TEORICO,
        "seed": SEED,
        "pergunta_de_decisao": PERGUNTA_DE_DECISAO,
    }


# ---------------------------------------------------------------------------
# Política como objeto e as três referências obrigatórias (Parte 1)
# ---------------------------------------------------------------------------
# A política é o par (escore, capacidade): ordena pelo escore e corta em `k`.
# O estimador só entra como gerador do escore; quem decide é a política.
@dataclass(frozen=True)
class PoliticaPericia:
    """Política de perícia sob restrição de capacidade.

    `estimador=None` (com `k=0`) representa "não periciar ninguém". `desempate`
    nomeia a coluna usada para desempatar escores idênticos; em `None` mantém-se
    a ordem original do DataFrame (ordenação estável), que é exatamente o
    critério da Fase 4 — é o que garante reproduzir o `Resultado@900` auditado.
    """

    nome: str
    k: int
    estimador: object | None = None
    desempate: str | None = None

    def escore(self, X: pd.DataFrame) -> np.ndarray:
        """Escore por sinistro: coluna da classe positiva de `predict_proba`."""
        if self.estimador is None:
            return np.zeros(len(X), dtype=float)
        return escore_positivo(self.estimador, X)


def aplicar_politica(politica: PoliticaPericia, X: pd.DataFrame) -> np.ndarray:
    """Índices (posições em `X`) dos sinistros que a política manda periciar.

    Ordena pelo escore decrescente e corta os `k` primeiros. Com `desempate`
    nomeado, escores idênticos são resolvidos pela coluna indicada (decrescente);
    sem ele, vale a ordem original — estável, igual à da Fase 4.
    """
    n_linhas = len(X)
    if politica.estimador is None or politica.k <= 0 or n_linhas == 0:
        return np.array([], dtype=int)

    escore = politica.escore(X)
    if politica.desempate is not None:
        secundario = (
            pd.to_numeric(X[politica.desempate], errors="coerce").fillna(0.0).to_numpy()
        )
        # `lexsort`: a última chave é a primária; os sinais invertem para decrescente.
        ordem = np.lexsort((-secundario, -escore))
    else:
        ordem = np.argsort(-escore, kind="mergesort")
    return ordem[: min(politica.k, n_linhas)]


def resultado_economico(
    y_true: pd.Series | np.ndarray,
    indices_periciados: np.ndarray,
    custo_pericia: float = CUSTO_PERICIA,
    custo_fraude_paga: float = CUSTO_FRAUDE_PAGA,
) -> dict[str, float | int]:
    """Economia e decomposição de custo de uma decisão de perícia.

    `economia = 21.700 × TP − 700 × k` é o resultado líquido em relação a **não
    periciar ninguém**; `custo_total = 21.700 × FN + 700 × FP` é o custo que
    sobra. As duas contas fecham no mesmo total: `economia = custo_sem_pericia −
    custo_total`, com `custo_sem_pericia = 21.700 × (nº de fraudes)`.
    """
    y = np.asarray(y_true, dtype=int)
    selecao = np.zeros(len(y), dtype=bool)
    selecao[np.asarray(indices_periciados, dtype=int)] = True

    tp = int(y[selecao].sum())
    k = int(selecao.sum())
    fp = k - tp
    positivos = int(y.sum())
    fn = positivos - tp

    return {
        "n_periciados": k,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_fraudes": positivos,
        "precisao": float(tp / k) if k else 0.0,
        "cobertura": float(tp / positivos) if positivos else 0.0,
        "economia": float(custo_fraude_paga * tp - custo_pericia * k),
        "custo_total": float(custo_fraude_paga * fn + custo_pericia * fp),
        "custo_sem_pericia": float(custo_fraude_paga * positivos),
    }


def construir_referencias(k: int = CAPACIDADE_MENSAL) -> dict[str, PoliticaPericia]:
    """As três referências obrigatórias da fase, em objetos reutilizáveis.

    (a) **não periciar ninguém** — economia zero, custo `21.700 × fraudes`;
    (b) **regra atual de R$ 50 mil** — o escore vem de `RegraAtualQuilha`, o
    mesmo estimador auditado na Fase 4, **embrulhado no `Pipeline`** como na Fase
    4: assim ele recebe o chamado cru e o `valor_reclamado` chega convertido a
    número (na base crua ele vem como texto com vírgula decimal);
    (c) **referência simples** — `DummyClassifier(strategy="prior")` dá a mesma
    nota a todos, o que equivale a escolher `k` ao acaso.

    As três aceitam o mesmo chamado cru de `X`; os estimadores de (b) e (c)
    precisam ser ajustados **no treino** antes do uso, e (a) não tem estimador.
    """
    return {
        "nao_periciar": PoliticaPericia(nome="Não periciar ninguém", k=0),
        "regra_atual": PoliticaPericia(
            nome="Regra atual (> R$ 50 mil)",
            k=k,
            estimador=construir_pipeline(modelo=RegraAtualQuilha()),
        ),
        "aleatoria": PoliticaPericia(
            nome="Referência simples (prior)",
            k=k,
            estimador=DummyClassifier(strategy="prior"),
        ),
    }


# ---------------------------------------------------------------------------
# Backtest com origem móvel — estimativa honesta fora da amostra (Parte 2)
# ---------------------------------------------------------------------------
def escore_positivo(estimador, X: pd.DataFrame) -> np.ndarray:
    """Coluna da classe positiva de `predict_proba` (apenas aplica, nunca ajusta)."""
    proba = np.asarray(estimador.predict_proba(X), dtype=float)
    return proba[:, 1] if proba.ndim == 2 else proba


def janelas_walk_forward(datas, minimo_meses_treino: int = 1) -> list[dict]:
    """Janelas de origem móvel sobre meses cronológicos.

    Treina em `[m1..mn]` e avalia no mês **imediatamente posterior** `m(n+1)`, com
    treino expansivo — sem embaralhar o tempo. Com os cinco meses da base
    (abr–ago) e um mês mínimo de treino saem **quatro** janelas de avaliação
    (mai, jun, jul e ago).
    """
    periodo = pd.to_datetime(pd.Series(np.asarray(datas))).dt.to_period("M")
    meses = sorted(periodo.unique())
    janelas: list[dict] = []
    for posicao in range(minimo_meses_treino, len(meses)):
        treino = meses[:posicao]
        avaliacao = meses[posicao]
        janelas.append(
            {
                "treino_rotulo": f"{treino[0]} a {treino[-1]}",
                "aval_rotulo": str(avaliacao),
                "mascara_treino": periodo.isin(treino).to_numpy(),
                "mascara_avaliacao": (periodo == avaliacao).to_numpy(),
            }
        )
    return janelas


def medir_janela(y_true, escore, k: int) -> dict[str, float | int]:
    """Métricas de uma janela para um escore, cortando em `k` (proporcional)."""
    y = np.asarray(y_true, dtype=int)
    escore = np.asarray(escore, dtype=float)
    n_linhas = len(y)
    k_efetivo = min(int(k), n_linhas)
    economia = resultado_economico(y, np.argsort(-escore, kind="mergesort")[:k_efetivo])
    tem_positivo = 0 < int(y.sum()) < n_linhas

    return {
        "n_sinistros": n_linhas,
        "n_fraudes": economia["n_fraudes"],
        "taxa_fraude": float(y.mean()) if n_linhas else 0.0,
        "k": k_efetivo,
        "tp": economia["tp"],
        "fp": economia["fp"],
        "fn": economia["fn"],
        "precisao_k": economia["precisao"],
        "cobertura_k": economia["cobertura"],
        "resultado_k": economia["economia"],
        "custo_total": economia["custo_total"],
        "pr_auc": float(average_precision_score(y, escore)) if tem_positivo else float("nan"),
        "roc_auc": float(roc_auc_score(y, escore)) if tem_positivo else float("nan"),
    }


def executar_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    datas,
    modelo_campeao,
    minimo_meses_treino: int = 1,
) -> pd.DataFrame:
    """Backtest walk-forward com o campeão, a regra atual e a referência simples.

    **Sem reabertura de tuning:** o campeão entra por `clone(modelo_campeao)`, isto
    é, com os **mesmos** hiperparâmetros do artefato; só mudam os dados de treino.
    O `k` de cada janela é proporcional à capacidade mensal
    (`round(900 × n_janela / 12.024)`), como manda o guardrail da unidade de tempo.
    """
    janelas = janelas_walk_forward(datas, minimo_meses_treino)
    registros: list[dict] = []

    for janela in janelas:
        X_treino = X.loc[janela["mascara_treino"]]
        y_treino = y.loc[janela["mascara_treino"]]
        X_aval = X.loc[janela["mascara_avaliacao"]]
        y_aval = y.loc[janela["mascara_avaliacao"]]
        k = calcular_k_proporcional(len(X_aval))

        campeao = construir_pipeline(modelo=clone(modelo_campeao))
        campeao.fit(X_treino, y_treino)

        regra_atual = construir_pipeline(modelo=RegraAtualQuilha())
        regra_atual.fit(X_treino, y_treino)

        aleatoria = DummyClassifier(strategy="prior")
        aleatoria.fit(X_treino, y_treino)

        estimadores = {
            "campeao": campeao,
            "regra_atual": regra_atual,
            "aleatoria": aleatoria,
        }
        for politica, estimador in estimadores.items():
            metricas = medir_janela(y_aval, escore_positivo(estimador, X_aval), k)
            registros.append(
                {
                    "janela": janela["aval_rotulo"],
                    "treino": janela["treino_rotulo"],
                    "politica": politica,
                    **metricas,
                }
            )

    return pd.DataFrame(registros)


# ---------------------------------------------------------------------------
# Política e economia no mês de teste (Parte 3)
# ---------------------------------------------------------------------------
def construir_politica_campeao(estimador, k: int = CAPACIDADE_MENSAL) -> PoliticaPericia:
    """A política recomendada: escore do campeão da Fase 4, cortado na capacidade de hoje."""
    return PoliticaPericia(nome="Campeão (escore XGBoost)", k=k, estimador=estimador)


def medir_politicas_no_mes(
    politicas: dict[str, PoliticaPericia],
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Mede cada política no mês de teste sob o **mesmo** critério econômico.

    Devolve uma linha por política com o `k` executado, `TP`/`FP`/`FN`, Precisão@k,
    Cobertura@k, economia (`21.700 × TP − 700 × k`) e custo total
    (`21.700 × FN + 700 × FP`). O estimador de cada política entra **já ajustado**:
    medir aqui não treina nada.
    """
    linhas: list[dict] = []
    for chave, politica in politicas.items():
        economia = resultado_economico(y, aplicar_politica(politica, X))
        linhas.append(
            {"chave": chave, "politica": politica.nome, "capacidade": politica.k, **economia}
        )
    return pd.DataFrame(linhas)


def _valor_reclamado_numerico(X: pd.DataFrame) -> np.ndarray:
    """`valor_reclamado` como número, tolerando o texto com vírgula da base crua.

    É a mesma conversão que `quilha.preparacao` faz dentro do Pipeline; repetida
    aqui porque a regra atual do case é medida sobre o chamado **cru**, onde o valor
    chega como `"8837,07"` — `pd.to_numeric` sozinho devolveria nulo para todo mundo.
    """
    serie = X["valor_reclamado"]
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce").fillna(0.0).to_numpy()
    texto = serie.astype("string").str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(texto, errors="coerce").fillna(0.0).to_numpy()


def regra_atual_do_case(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    limiar: float = LIMIAR_REGRA_ATUAL,
    k: int = CAPACIDADE_MENSAL,
) -> dict[str, float | int]:
    """Executa a regra atual **exatamente como o case a descreve**.

    Filtra `valor_reclamado > limiar`, ordena por valor decrescente e corta em `k`.
    Devolve quantos pedidos o filtro gera (`n_acima_limiar`) e quantos foram de fato
    periciados (`n_periciados`), além da decomposição econômica de `resultado_economico`.
    """
    valor = _valor_reclamado_numerico(X)
    acima = np.nonzero(valor > limiar)[0]
    ordem = acima[np.argsort(-valor[acima], kind="mergesort")]
    periciados = ordem[: min(int(k), len(ordem))]

    resultado = resultado_economico(y, periciados)
    resultado["n_acima_limiar"] = int(acima.size)
    resultado["limiar"] = float(limiar)
    return resultado


def pontos_de_decisao_k(
    tabela: pd.DataFrame,
    resumo: dict[str, float | int],
    capacidade: int = CAPACIDADE_MENSAL,
) -> list[tuple[int, float, str]]:
    """Os pontos de decisão sobre a curva `Resultado(k)`, com o valor em reais.

    Ordem: **capacidade de hoje**, **corte teórico** `p*`, **ótimo empírico** e o
    **último `k` ainda lucrativo**. Os três últimos saem do `resumo` da curva —
    nada digitado à mão.
    """
    rotulados = [
        (int(capacidade), "capacidade hoje"),
        (int(resumo["k_corte_teorico"]), "corte teórico p*"),
        (int(resumo["k_otimo_empirico"]), "ótimo empírico"),
        (int(resumo["k_maximo_ainda_lucrativo"]), "último k lucrativo"),
    ]
    return [
        (k, float(tabela.loc[tabela["k"] == k, "resultado_liquido"].iloc[0]), rotulo)
        for k, rotulo in rotulados
    ]


def salvar_curva_decisao(
    tabela: pd.DataFrame,
    pontos: list[tuple[int, float, str]],
    caminho: Path | str | None = None,
) -> Path:
    """Curva `Resultado(k)` do campeão no mês de teste, com os pontos de decisão anotados."""
    if caminho is None:
        caminho = PASTA_FIGURAS / "05_01_curva_resultado_k.png"
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    figura, eixo = plt.subplots(figsize=(11, 6))
    eixo.plot(
        tabela["k"],
        tabela["resultado_liquido"] / 1000.0,
        color="#1f4e79",
        lw=2,
        label="Resultado líquido (R$ mil)",
    )
    eixo.axhline(0.0, color="0.4", lw=1)

    cores = ["#c62828", "#f9a825", "#2e7d32", "#6a1b9a"]
    for (k, valor, rotulo), cor in zip(pontos, cores, strict=True):
        eixo.axvline(k, color=cor, ls="--", lw=1.4)
        eixo.plot([k], [valor / 1000.0], marker="o", color=cor, ms=6)
        eixo.annotate(
            f"{rotulo}\nk={k} · R$ {valor:,.0f}",
            xy=(k, valor / 1000.0),
            xytext=(6, 10 if valor >= 0 else -30),
            textcoords="offset points",
            color=cor,
            fontsize=9,
            fontweight="bold",
        )

    eixo.set_xlabel("k — sinistros periciados, por ordem decrescente de risco")
    eixo.set_ylabel("Resultado líquido (R$ mil)")
    eixo.set_title("Fase 5 — curva Resultado(k) do campeão no mês de teste e os pontos de decisão")
    eixo.grid(alpha=0.3)
    eixo.legend(loc="lower left")
    figura.tight_layout()
    figura.savefig(caminho, dpi=150)
    plt.close(figura)
    return caminho


# ---------------------------------------------------------------------------
# Dimensionamento da equipe e valor marginal (Parte 4)
# ---------------------------------------------------------------------------
# O case define a capacidade como 30 peritos × 30 perícias/mês. A contratação é
# discreta: cada perito a mais vale 30 perícias a mais na curva Resultado(k).
PERICIAS_POR_PERITO = 30
PERITOS_ATUAIS = CAPACIDADE_MENSAL // PERICIAS_POR_PERITO


def k_marginal_cruza_piso(
    tabela_curva: pd.DataFrame,
    piso: float | None = None,
) -> int:
    """Primeiro `k` em que a probabilidade do k-ésimo sinistro cai abaixo do piso.

    É o ponto em que a **perícia marginal** deixa de ter valor esperado positivo —
    o `limiar_corte_teorico` da Fase 4 visto pela curva. Contado à parte, serve de
    contraprova do `k_corte_teorico` que vem no resumo da curva.
    """
    if piso is None:
        piso = piso_viabilidade()
    probabilidade = tabela_curva["probabilidade_do_k_esimo"].to_numpy()
    abaixo = np.nonzero(probabilidade[1:] < piso)[0]
    return int(abaixo[0] + 1) if abaixo.size else len(tabela_curva) - 1


def curva_ganho_por_perito(
    tabela_curva: pd.DataFrame,
    peritos_max: int,
    pericas_por_perito: int = PERICIAS_POR_PERITO,
    peritos_base: int = PERITOS_ATUAIS,
) -> pd.DataFrame:
    """Resultado e ganho marginal por número de peritos (contratação discreta).

    A capacidade é `k = peritos × pericas_por_perito`, e o resultado sai da curva
    `Resultado(k)`. `ganho_marginal_perito` é o acréscimo de resultado trazido pelo
    perito adicional e `ganho_sobre_base`, o acumulado em relação à equipe de
    `peritos_base` (hoje, 30 peritos = 900 perícias).
    """
    maximo_k = len(tabela_curva) - 1
    linhas = [
        {
            "peritos": peritos,
            "k": min(peritos * pericas_por_perito, maximo_k),
        }
        for peritos in range(1, peritos_max + 1)
    ]
    quadro = pd.DataFrame(linhas)
    quadro["resultado_liquido"] = [
        float(tabela_curva.loc[tabela_curva["k"] == k, "resultado_liquido"].iloc[0])
        for k in quadro["k"]
    ]
    resultado_base = float(
        quadro.loc[quadro["peritos"] == peritos_base, "resultado_liquido"].iloc[0]
    )
    quadro["ganho_marginal_perito"] = quadro["resultado_liquido"].diff()
    quadro["ganho_sobre_base"] = quadro["resultado_liquido"] - resultado_base
    return quadro


def salvar_ganho_por_perito(
    quadro: pd.DataFrame,
    marcos: dict[str, int],
    caminho: Path | str | None = None,
) -> Path:
    """Figura do ganho por perito: resultado acumulado e ganho marginal do perito adicional."""
    if caminho is None:
        caminho = PASTA_FIGURAS / "05_03_ganho_por_perito.png"
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    figura, eixo = plt.subplots(figsize=(11, 6))
    eixo.plot(
        quadro["peritos"],
        quadro["resultado_liquido"] / 1000.0,
        color="#1f4e79",
        lw=2,
        label="Resultado líquido (R$ mil)",
    )
    eixo.set_xlabel("peritos na equipe")
    eixo.set_ylabel("Resultado líquido (R$ mil)", color="#1f4e79")
    eixo.grid(alpha=0.3)

    eixo_marg = eixo.twinx()
    eixo_marg.bar(
        quadro["peritos"],
        quadro["ganho_marginal_perito"] / 1000.0,
        color="#c62828",
        alpha=0.30,
        label="ganho marginal do perito (R$ mil)",
    )
    eixo_marg.axhline(0.0, color="0.4", lw=1)
    eixo_marg.set_ylabel("Ganho marginal do perito (R$ mil)", color="#c62828")

    cores = ["#2e7d32", "#f9a825", "#6a1b9a"]
    for (rotulo, peritos), cor in zip(marcos.items(), cores, strict=True):
        eixo.axvline(peritos, color=cor, ls="--", lw=1.4, label=f"{rotulo} = {peritos}")

    eixo.set_title("Fase 5 — ganho por perito na curva Resultado(k) do campeão")
    linhas, rotulos_eixo = eixo.get_legend_handles_labels()
    linhas_marg, rotulos_marg = eixo_marg.get_legend_handles_labels()
    eixo.legend(linhas + linhas_marg, rotulos_eixo + rotulos_marg, loc="upper left", fontsize=9)
    figura.tight_layout()
    figura.savefig(caminho, dpi=150)
    plt.close(figura)
    return caminho


# ---------------------------------------------------------------------------
# Sensibilidade, break-even e incerteza (Parte 5)
# ---------------------------------------------------------------------------
def tp_acumulado_por_k(y_true, escore) -> np.ndarray:
    """TP acumulado ao periciar em ordem decrescente de escore, para `k = 0..n`.

    Devolve um vetor de tamanho `n + 1` em que a posição `k` é o número de fraudes
    capturadas ao cortar a fila em `k`. Como o **ranking** de uma política não
    depende dos parâmetros econômicos, este vetor permite recomputar `Resultado(k)`
    sob qualquer custo sem reordenar nada.
    """
    y = np.asarray(y_true, dtype=int)
    ordem = np.argsort(-np.asarray(escore, dtype=float), kind="mergesort")
    return np.concatenate([[0], np.cumsum(y[ordem])])


def resultado_por_k(
    tp_acumulado: np.ndarray,
    custo_pericia: float = CUSTO_PERICIA,
    custo_fraude_paga: float = CUSTO_FRAUDE_PAGA,
) -> np.ndarray:
    """Vetor `Resultado(k) = custo_fraude × TP(k) − custo_pericia × k`, para `k = 0..n`."""
    k = np.arange(len(tp_acumulado))
    return custo_fraude_paga * tp_acumulado - custo_pericia * k


def _resumo_amostra(amostra: np.ndarray) -> dict[str, float]:
    """Média, desvio e IC 95% (percentis 2,5 e 97,5) de uma amostra bootstrap."""
    return {
        "media": float(np.mean(amostra)),
        "desvio": float(np.std(amostra, ddof=1)),
        "ic_inferior": float(np.percentile(amostra, 2.5)),
        "ic_superior": float(np.percentile(amostra, 97.5)),
    }


def bootstrap_resultado(
    y_true,
    escores: dict[str, np.ndarray],
    k: int = CAPACIDADE_MENSAL,
    n_reamostragens: int = N_REAMOSTRAGENS_BOOTSTRAP,
    seed: int = SEED,
    custo_pericia: float = CUSTO_PERICIA,
    custo_fraude_paga: float = CUSTO_FRAUDE_PAGA,
) -> tuple[pd.DataFrame, dict[str, float | bool]]:
    """IC 95% do `Resultado@k` por bootstrap no mês de teste (reamostragem pareada).

    Cada reamostragem sorteia `n` linhas **com reposição** e **repete a decisão**
    inteira: reordena pelo escore e corta em `k`. Assim o intervalo captura tanto a
    variabilidade das fraudes quanto a da própria fila, não só a contagem de TP.
    A mesma reamostragem alimenta todas as políticas de `escores` (comparação
    **pareada**); a linha `diferenca` é a primeira política menos a segunda.
    """
    y = np.asarray(y_true, dtype=int)
    n = len(y)
    chaves = list(escores)
    matrizes = {chave: np.asarray(escores[chave], dtype=float) for chave in chaves}
    corte = min(k, n)
    gerador = np.random.default_rng(seed)

    amostras = {chave: np.empty(n_reamostragens, dtype=float) for chave in chaves}
    for posicao in range(n_reamostragens):
        idx = gerador.integers(0, n, size=n)
        y_amostra = y[idx]
        for chave in chaves:
            escore_amostra = matrizes[chave][idx]
            selecao = np.argsort(-escore_amostra, kind="mergesort")[:corte]
            tp = int(y_amostra[selecao].sum())
            amostras[chave][posicao] = custo_fraude_paga * tp - custo_pericia * corte

    chave_a, chave_b = chaves[0], chaves[1]
    diferenca = amostras[chave_a] - amostras[chave_b]

    linhas = [{"metrica": chave, **_resumo_amostra(amostras[chave])} for chave in chaves]
    linhas.append({"metrica": "diferenca", **_resumo_amostra(diferenca)})
    tabela = pd.DataFrame(linhas)[["metrica", "media", "desvio", "ic_inferior", "ic_superior"]]

    inferior = float(np.percentile(diferenca, 2.5))
    superior = float(np.percentile(diferenca, 97.5))
    resumo: dict[str, float | bool] = {
        "comparacao": f"{chave_a} - {chave_b}",
        "diferenca_media": float(np.mean(diferenca)),
        "diferenca_ic_inferior": inferior,
        "diferenca_ic_superior": superior,
        "exclui_zero": bool(inferior > 0.0 or superior < 0.0),
        "n_reamostragens": n_reamostragens,
        "seed": seed,
    }
    return tabela, resumo


def salvar_bootstrap(tabela: pd.DataFrame, caminho: Path | str | None = None) -> Path:
    """IC 95% por bootstrap: barras de erro do campeão, da regra atual e da diferença."""
    if caminho is None:
        caminho = PASTA_FIGURAS / "05_04_bootstrap.png"
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    rotulos = list(tabela["metrica"])
    medias = tabela["media"].to_numpy() / 1000.0
    erros = np.vstack(
        [
            (tabela["media"] - tabela["ic_inferior"]).to_numpy() / 1000.0,
            (tabela["ic_superior"] - tabela["media"]).to_numpy() / 1000.0,
        ]
    )

    figura, eixo = plt.subplots(figsize=(10, 6))
    cores = ["#1f4e79", "#c62828", "#2e7d32"]
    eixo.errorbar(
        medias,
        rotulos,
        xerr=erros,
        fmt="o",
        capsize=6,
        lw=2,
        ms=8,
        ecolor="0.3",
    )
    for posicao, cor in enumerate(cores[: len(rotulos)]):
        eixo.plot(medias[posicao], posicao, marker="o", ms=9, color=cor)
    eixo.axvline(0.0, color="0.4", lw=1, ls="--")
    eixo.set_xlabel("Resultado líquido no mês (R$ mil)")
    eixo.set_title(
        "Fase 5 — intervalo de confiança de 95% (2.000 reamostragens, SEED=42)"
    )
    eixo.grid(alpha=0.3, axis="x")
    figura.tight_layout()
    figura.savefig(caminho, dpi=150)
    plt.close(figura)
    return caminho


def grade_sensibilidade(
    tp_acumulado_campeao: np.ndarray,
    tp_acumulado_regra: np.ndarray,
    grade_custo_fraude_paga: tuple[float, ...] = GRADE_CUSTO_FRAUDE_PAGA,
    grade_custo_pericia: tuple[float, ...] = GRADE_CUSTO_PERICIA,
    grade_capacidade: tuple[int, ...] = GRADE_CAPACIDADE,
) -> pd.DataFrame:
    """Grade combinada de sensibilidade econômica (custo da fraude × perícia × capacidade).

    Em cada célula, `k` é a capacidade do cenário. Como o ranking do campeão e o da
    regra **não dependem dos custos**, o `TP(k)` de cada política é o mesmo em toda a
    grade — o que muda é a valoração. Cada linha traz o resultado das duas políticas
    na capacidade do cenário, o ganho (campeão − regra), o `k` ótimo sob aquele par
    de custos e o resultado nele.
    """
    linhas: list[dict[str, float | int | bool]] = []
    for custo_fraude in grade_custo_fraude_paga:
        for custo_pericia in grade_custo_pericia:
            resultado_campeao = resultado_por_k(tp_acumulado_campeao, custo_pericia, custo_fraude)
            resultado_regra = resultado_por_k(tp_acumulado_regra, custo_pericia, custo_fraude)
            posicao_otima = int(np.argmax(resultado_campeao))
            for capacidade in grade_capacidade:
                k = min(int(capacidade), len(tp_acumulado_campeao) - 1)
                linhas.append(
                    {
                        "custo_fraude_paga": float(custo_fraude),
                        "custo_pericia": float(custo_pericia),
                        "capacidade": int(capacidade),
                        "k": k,
                        "tp_campeao": int(tp_acumulado_campeao[k]),
                        "tp_regra": int(tp_acumulado_regra[k]),
                        "resultado_campeao": float(resultado_campeao[k]),
                        "resultado_regra": float(resultado_regra[k]),
                        "ganho": float(resultado_campeao[k] - resultado_regra[k]),
                        "k_otimo": posicao_otima,
                        "resultado_no_otimo": float(resultado_campeao[posicao_otima]),
                    }
                )
    tabela = pd.DataFrame(linhas)
    tabela["vence_regra"] = tabela["ganho"] > 0.0
    return tabela


def salvar_sensibilidade(grade: pd.DataFrame, caminho: Path | str | None = None) -> Path:
    """Heatmap do ganho sobre a regra atual: custo da fraude (linhas) × capacidade (colunas)."""
    if caminho is None:
        caminho = PASTA_FIGURAS / "05_02_sensibilidade.png"
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    recorte = grade[grade["custo_pericia"] == CUSTO_PERICIA]
    matriz = (
        recorte.pivot(index="custo_fraude_paga", columns="capacidade", values="ganho") / 1000.0
    )

    figura, eixo = plt.subplots(figsize=(10, 6))
    limite = float(np.max(np.abs(matriz.to_numpy())))
    imagem = eixo.imshow(
        matriz.to_numpy(),
        cmap="RdYlGn",
        aspect="auto",
        origin="lower",
        vmin=-limite,
        vmax=limite,
    )
    eixo.set_xticks(range(len(matriz.columns)), [str(coluna) for coluna in matriz.columns])
    eixo.set_yticks(range(len(matriz.index)), [f"{valor:,.0f}" for valor in matriz.index])
    eixo.set_xlabel("capacidade (perícias/mês)")
    eixo.set_ylabel("custo da fraude paga (R$)")
    for linha in range(matriz.shape[0]):
        for coluna in range(matriz.shape[1]):
            eixo.text(
                coluna, linha, f"{matriz.to_numpy()[linha, coluna]:,.0f}",
                ha="center", va="center", fontsize=9,
            )
    eixo.set_title(
        "Fase 5 — ganho do campeão sobre a regra atual (R$ mil)\n"
        f"custo da perícia fixo em R$ {CUSTO_PERICIA:,.0f}"
    )
    figura.colorbar(imagem, ax=eixo, label="ganho (R$ mil)")
    figura.tight_layout()
    figura.savefig(caminho, dpi=150)
    plt.close(figura)
    return caminho


def break_even(
    tp_campeao: int,
    tp_regra: int,
    k: int = CAPACIDADE_MENSAL,
    custo_pericia: float = CUSTO_PERICIA,
) -> dict[str, float | bool]:
    """Valores críticos em que a política — ou a vantagem sobre a regra — se anula.

    Com a **mesma capacidade** `k`, o ganho sobre a regra vale
    `custo_fraude × (tp_campeao − tp_regra)`: não depende do custo da perícia e só
    zera se o custo da fraude for zero. Já a **própria política** deixa de se pagar
    abaixo de `custo_pericia × k / tp_campeao` por fraude paga (razão
    `custo_fraude / custo_pericia = k / tp_campeao`). A regra atual exige muito mais.
    """
    custo_critico_politica = custo_pericia * k / tp_campeao if tp_campeao else float("inf")
    custo_critico_regra = custo_pericia * k / tp_regra if tp_regra else float("inf")
    return {
        "custo_fraude_critico_politica": float(custo_critico_politica),
        "razao_critica_politica": float(k / tp_campeao) if tp_campeao else float("inf"),
        "custo_fraude_critico_regra": float(custo_critico_regra),
        "razao_critica_regra": float(k / tp_regra) if tp_regra else float("inf"),
        "vantagem_sobre_regra_sempre_positiva": bool(tp_campeao > tp_regra),
    }


def sensibilidade_taxa_fraude(
    y_true,
    escore,
    k: int = CAPACIDADE_MENSAL,
    taxas_alvo: tuple[float, ...] = (0.002, 0.0035, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.0247),
    n_repeticoes: int = 200,
    seed: int = SEED,
    custo_pericia: float = CUSTO_PERICIA,
    custo_fraude_paga: float = CUSTO_FRAUDE_PAGA,
) -> pd.DataFrame:
    """Resultado@k sob prevalência de fraude simulada (drift) na janela de avaliação.

    O escore e a ordem da fila ficam **congelados**: muda apenas quantas fraudes
    existem no mês. Para cada taxa-alvo, sorteia `n_alvo` fraudes entre as observadas
    sem reposição (e com reposição quando o alvo excede o observado) e reavalia o
    `Resultado@k` no mesmo topo de fila, repetindo `n_repeticoes` vezes com `SEED`.
    """
    y = np.asarray(y_true, dtype=int)
    n = len(y)
    n_fraudes = int(y.sum())
    n_negativos = n - n_fraudes
    escore = np.asarray(escore, dtype=float)
    corte = min(k, n)
    selecao = np.argsort(-escore, kind="mergesort")[:corte]
    mascara_topo = np.zeros(n, dtype=bool)
    mascara_topo[selecao] = True
    fraude_no_topo = mascara_topo[np.nonzero(y == 1)[0]]

    gerador = np.random.default_rng(seed)
    linhas: list[dict[str, float | int | bool]] = []
    for taxa in taxas_alvo:
        n_alvo = int(round(taxa * n_negativos / (1.0 - taxa)))
        n_alvo = min(max(n_alvo, 0), n_fraudes)
        resultados = np.empty(n_repeticoes, dtype=float)
        for repeticao in range(n_repeticoes):
            sorteio = gerador.choice(n_fraudes, size=n_alvo, replace=False)
            no_topo = int(fraude_no_topo[sorteio].sum())
            resultados[repeticao] = custo_fraude_paga * no_topo - custo_pericia * corte
        linhas.append(
            {
                "taxa_alvo": float(taxa),
                "n_fraudes_alvo": int(n_alvo),
                "resultado_medio": float(resultados.mean()),
                "resultado_desvio": float(resultados.std(ddof=1)),
                "paga": bool(resultados.mean() > 0.0),
            }
        )
    return pd.DataFrame(linhas)


def taxa_fraude_de_break_even(
    y_true,
    escore,
    k: int = CAPACIDADE_MENSAL,
    custo_pericia: float = CUSTO_PERICIA,
    custo_fraude_paga: float = CUSTO_FRAUDE_PAGA,
) -> float:
    """Taxa de fraude mínima do fluxo para a política se pagar com o `k` de hoje.

    Com a fila congelada, o topo de `k` captura uma **cobertura** fixa das fraudes.
    Zerando `custo_fraude × n_alvo × cobertura − custo_pericia × k`, obtém-se o
    número mínimo de fraudes do mês e, daí, a taxa de fraude de equilíbrio.
    """
    y = np.asarray(y_true, dtype=int)
    n = len(y)
    n_fraudes = int(y.sum())
    n_negativos = n - n_fraudes
    corte = min(k, n)
    selecao = np.argsort(-np.asarray(escore, dtype=float), kind="mergesort")[:corte]
    tp_no_k = int(y[selecao].sum())
    cobertura = tp_no_k / n_fraudes if n_fraudes else 0.0
    if cobertura <= 0.0:
        return float("inf")
    n_alvo = custo_pericia * corte / (custo_fraude_paga * cobertura)
    return float(n_alvo / (n_negativos + n_alvo))


# ---------------------------------------------------------------------------
# Parte 6 — Triagem de disparidade da fila periciada
# ---------------------------------------------------------------------------
def _faixa_idade(
    serie: pd.Series,
    faixas: tuple[tuple[str, int, int], ...] = FAIXAS_IDADE_DISPARIDADE,
) -> pd.Series:
    """Discretiza a idade em faixas legíveis; idade ausente vira 'não informado'."""
    idade = pd.to_numeric(serie, errors="coerce")
    rotulo = pd.Series("não informado", index=serie.index, dtype=object)
    for nome, minimo, maximo in faixas:
        dentro = idade.between(minimo, maximo).fillna(False)
        rotulo[dentro] = nome
    return rotulo


def triagem_disparidade(
    y_true,
    escore,
    X_bruto: pd.DataFrame,
    k: int = CAPACIDADE_MENSAL,
    coluna_idade: str = COLUNA_IDADE_DISPARIDADE,
    coluna_uf: str = COLUNA_UF_DISPARIDADE,
) -> pd.DataFrame:
    """Triagem de disparidade da fila periciada — sinal para revisão, não veredito legal.

    Para cada grupo (faixa de idade e UF), mede a **taxa de seleção** (fração do grupo
    que entra nas `k` perícias) e a **precisão@k** do grupo (fraudes entre os periciados).
    A leitura de justiça é a razão entre a taxa de seleção do grupo e a taxa média: um
    grupo selecionado muito acima do resto concentra a fila e merece revisão humana.

    `X_bruto` é o chamado **cru** (antes do preparo), porque `idade_segurado` e `uf` vivem
    nas colunas originais. Devolve uma linha por `(dimensao, grupo)`.
    """
    y = np.asarray(y_true, dtype=int)
    n = len(y)
    corte = min(k, n)
    selecao = np.argsort(-np.asarray(escore, dtype=float), kind="mergesort")[:corte]
    marca_selecao = np.zeros(n, dtype=bool)
    marca_selecao[selecao] = True
    taxa_selecao_geral = corte / n if n else 0.0

    dimensoes = {
        "idade": _faixa_idade(X_bruto[coluna_idade]),
        "uf": X_bruto[coluna_uf].astype("string").fillna("não informado"),
    }

    linhas: list[dict] = []
    for dimensao, grupos in dimensoes.items():
        for grupo in sorted(grupos.unique(), key=str):
            no_grupo = (grupos == grupo).to_numpy()
            n_grupo = int(no_grupo.sum())
            if n_grupo == 0:
                continue
            n_selecionado = int((no_grupo & marca_selecao).sum())
            tp = int(y[no_grupo & marca_selecao].sum())
            n_fraudes = int(y[no_grupo].sum())
            taxa_selecao = n_selecionado / n_grupo
            precisao = (tp / n_selecionado) if n_selecionado else 0.0
            taxa_fraude_grupo = n_fraudes / n_grupo
            razao_selecao = taxa_selecao / taxa_selecao_geral if taxa_selecao_geral else 0.0
            linhas.append(
                {
                    "dimensao": dimensao,
                    "grupo": str(grupo),
                    "n_grupo": n_grupo,
                    "n_selecionado": n_selecionado,
                    "taxa_selecao": taxa_selecao,
                    "razao_selecao": razao_selecao,
                    "precisao": precisao,
                    "taxa_fraude_grupo": taxa_fraude_grupo,
                    "lift": precisao / taxa_fraude_grupo if taxa_fraude_grupo else 0.0,
                }
            )
    return pd.DataFrame(linhas)

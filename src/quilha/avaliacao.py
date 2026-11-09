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
        caminho = PASTA_FIGURAS / "05_curva_decisao.png"
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

"""Reamostragem do treino do fold — Parte 4 (undersampling controlado).

Por que este módulo existe
--------------------------
O desbalanceamento 31:1 tem dois tratamentos possíveis: mexer no **peso** de cada
erro (Parte 3) ou **jogar fora** exemplos do majoritário (aqui). A Parte 4 mede o
segundo, com número, para o comitê decidir.

Por que o reamostrador NÃO é uma etapa do Pipeline
--------------------------------------------------
Verificado nesta versão do scikit-learn (1.9.1) por sonda própria: o `Pipeline`
**não propaga o `y`** devolvido por uma etapa intermediária. A sonda descartou
linhas no `fit_transform` de uma etapa e mostrou o `X` chegando adulterado ao
modelo final, com o `y` intacto — ou seja, um reamostrador como etapa corromperia
o ajuste. Por isso ele entra como o **próprio passo `modelo`**, envolvendo o
estimador da família:

* o `X` já chega pré-processado (imputado e codificado), que é o que dá sentido à
  distância usada pelo NearMiss-1 — distância sobre texto cru não significa nada;
* o corte de linhas acontece dentro do `fit`, portanto dentro do **fold de treino**,
  nunca no holdout (zero vazamento).

Efeito colateral que precisa ficar registrado
---------------------------------------------
Reamostrar **muda a prevalência** da fraude no treino (de 2,49% para 1/6 ou 1/11).
As métricas de ranking (PR AUC, ROC AUC, Resultado@k) continuam comparáveis; a
**probabilidade crua não**. Se o corte de 3,2258% for aplicado na escala
reamostrada, ele precisa ser recalibrado — item da Parte 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEED = 42

METODO_CONTROLE = "controle"
METODO_ALEATORIO = "aleatorio"
METODO_NEARMISS = "nearmiss1"

N_VIZINHOS_PADRAO = 3


@dataclass(frozen=True)
class EstrategiaAmostragem:
    """Uma forma de montar o conjunto de treino de um fold."""

    rotulo: str
    metodo: str
    proporcao: float | None = None
    n_vizinhos: int = N_VIZINHOS_PADRAO

    def descricao(self) -> str:
        """Texto legível para tabela e relatório."""
        if self.metodo == METODO_CONTROLE:
            return "controle (sem reamostragem)"
        if self.metodo == METODO_ALEATORIO:
            return f"sorteio aleatorio 1:{self.proporcao:g}"
        return f"NearMiss-1 1:{self.proporcao:g} (n_vizinhos={self.n_vizinhos})"


def estrategias_parte4() -> list[EstrategiaAmostragem]:
    """As cinco estratégias do teste controlado, na ordem do prompt.

    O NearMiss-1 roda **nas mesmas proporções** do sorteio aleatório de propósito:
    assim a única variável que muda entre os dois é *quem* é escolhido (proximidade
    contra sorteio), e não *quanto* se mantém.
    """
    return [
        EstrategiaAmostragem("controle", METODO_CONTROLE),
        EstrategiaAmostragem("aleatorio_1_5", METODO_ALEATORIO, 5.0),
        EstrategiaAmostragem("aleatorio_1_10", METODO_ALEATORIO, 10.0),
        EstrategiaAmostragem("nearmiss1_1_5", METODO_NEARMISS, 5.0),
        EstrategiaAmostragem("nearmiss1_1_10", METODO_NEARMISS, 10.0),
    ]


def _selecionar_linhas(objeto, indices: np.ndarray):
    """Indexa respeitando pandas (rótulo) e numpy (posição)."""
    if isinstance(objeto, pd.DataFrame | pd.Series):
        return objeto.iloc[indices]
    return np.asarray(objeto)[indices]


def _para_matriz_numerica(X) -> np.ndarray:
    """Converte a matriz pré-processada em `float` para o cálculo de distância."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float)
    return np.asarray(X, dtype=float)


def _quantidade_negativos(
    n_positivos: int,
    n_negativos_disponiveis: int,
    proporcao: float,
) -> int:
    """Quantos negativos manter para uma proporção `1:proporcao`."""
    return min(n_negativos_disponiveis, int(round(proporcao * n_positivos)))


def indices_aleatorios(
    y,
    proporcao: float,
    seed: int = SEED,
) -> np.ndarray:
    """Mantém todos os positivos e sorteia negativos na proporção pedida."""
    alvo = np.asarray(y)
    indices_positivos = np.flatnonzero(alvo == 1)
    indices_negativos = np.flatnonzero(alvo == 0)

    quantidade = _quantidade_negativos(
        n_positivos=len(indices_positivos),
        n_negativos_disponiveis=len(indices_negativos),
        proporcao=proporcao,
    )
    gerador = np.random.default_rng(seed)
    escolhidos = gerador.choice(indices_negativos, size=quantidade, replace=False)
    return np.sort(np.concatenate([indices_positivos, escolhidos]))


def indices_nearmiss1(
    X,
    y,
    proporcao: float,
    n_vizinhos: int = N_VIZINHOS_PADRAO,
) -> np.ndarray:
    """NearMiss-1: mantém os negativos mais próximos da fronteira com os positivos.

    O critério é o do método original: cada exemplo do majoritário é pontuado pela
    **distância média aos `n_vizinhos` positivos mais próximos**, e sobrevivem os de
    menor pontuação. É o que dá sentido a chamar o undersampling de "inteligente".

    A padronização interna existe só para a distância ser comparável: sem ela,
    `valor_reclamado` (dezenas de milhares) engoliria as colunas binárias do one-hot.

    Não há sorteio: o método é determinístico.
    """
    alvo = np.asarray(y)
    indices_positivos = np.flatnonzero(alvo == 1)
    indices_negativos = np.flatnonzero(alvo == 0)

    quantidade = _quantidade_negativos(
        n_positivos=len(indices_positivos),
        n_negativos_disponiveis=len(indices_negativos),
        proporcao=proporcao,
    )

    matriz = _para_matriz_numerica(X)
    padronizada = StandardScaler().fit_transform(matriz)

    vizinhos = NearestNeighbors(n_neighbors=min(n_vizinhos, len(indices_positivos)))
    vizinhos.fit(padronizada[indices_positivos])
    distancias, _ = vizinhos.kneighbors(padronizada[indices_negativos])
    media_por_negativo = distancias.mean(axis=1)

    ordem = np.argsort(media_por_negativo, kind="stable")
    escolhidos = indices_negativos[ordem[:quantidade]]
    return np.sort(np.concatenate([indices_positivos, escolhidos]))


def selecionar_indices(
    X,
    y,
    estrategia: EstrategiaAmostragem | None,
    seed: int = SEED,
) -> np.ndarray | None:
    """Índices que sobrevivem no treino do fold; `None` quando não há corte."""
    if estrategia is None or estrategia.metodo == METODO_CONTROLE:
        return None
    if estrategia.proporcao is None:
        raise ValueError(f"Estratégia {estrategia.rotulo} exige proporção definida.")
    if estrategia.metodo == METODO_ALEATORIO:
        return indices_aleatorios(y=y, proporcao=estrategia.proporcao, seed=seed)
    if estrategia.metodo == METODO_NEARMISS:
        return indices_nearmiss1(
            X=X,
            y=y,
            proporcao=estrategia.proporcao,
            n_vizinhos=estrategia.n_vizinhos,
        )
    raise ValueError(f"Método de amostragem desconhecido: {estrategia.metodo}")


def reamostrar(
    X,
    y,
    estrategia: EstrategiaAmostragem | None,
    seed: int = SEED,
) -> tuple:
    """Devolve `(X, y)` do treino do fold já reamostrados."""
    indices = selecionar_indices(X=X, y=y, estrategia=estrategia, seed=seed)
    if indices is None:
        return X, y
    return _selecionar_linhas(X, indices), _selecionar_linhas(y, indices)


class ModeloComSubamostragem(BaseEstimator, ClassifierMixin):
    """Passo `modelo` do Pipeline que reamostra o treino do fold antes de ajustar.

    Envolve o estimador da família em vez de virar etapa própria do `Pipeline`
    porque o `Pipeline` do scikit-learn 1.9.1 não propaga o `y` entre etapas (ver
    o docstring do módulo). Também guarda a contagem de linhas para auditoria: é a
    prova de que o corte aconteceu no treino e nunca no holdout.
    """

    def __init__(
        self,
        estimador: BaseEstimator | None = None,
        estrategia: EstrategiaAmostragem | None = None,
        seed: int = SEED,
    ) -> None:
        self.estimador = estimador
        self.estrategia = estrategia
        self.seed = seed

    def fit(self, X, y):
        """Reamostra o treino recebido e ajusta o estimador interno."""
        X_treino, y_treino = reamostrar(
            X=X,
            y=y,
            estrategia=self.estrategia,
            seed=self.seed,
        )
        self.modelo_ = clone(self.estimador).fit(X_treino, y_treino)
        self.n_linhas_original_ = int(len(y))
        self.n_linhas_treino_ = int(len(y_treino))
        self.n_fraudes_treino_ = int(np.sum(np.asarray(y_treino) == 1))
        self.proporcao_treino_ = float(self.n_linhas_treino_ / max(self.n_fraudes_treino_, 1))
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Delega ao estimador interno — sem reamostragem na predição."""
        return self.modelo_.predict_proba(X)

    def predict(self, X) -> np.ndarray:
        """Delega ao estimador interno."""
        return self.modelo_.predict(X)

    @property
    def classes_(self) -> np.ndarray:
        """Classes vistas pelo estimador interno."""
        return self.modelo_.classes_


def estimador_interno(modelo: BaseEstimator) -> BaseEstimator:
    """Devolve o estimador da família, desembrulhando o reamostrador se houver."""
    if isinstance(modelo, ModeloComSubamostragem):
        return modelo.modelo_
    return modelo

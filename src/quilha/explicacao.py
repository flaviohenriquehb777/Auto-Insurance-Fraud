"""Explicação SHAP agregada por feature original (Fases 3 e 4 do CRISP-DM).

O `OneHotEncoder` dilui uma única variável em várias colunas (ex.: `tipo_sinistro`
vira `tipo_sinistro_Colisão`, `tipo_sinistro_Roubo`, ...). O SHAP explica coluna a
coluna, então a leitura crua entrega ao negócio nomes técnicos ilegíveis. Este
módulo soma as contribuições das colunas codificadas de volta à **feature
original** — é o nome que o negócio lê.

Regra de agregação
------------------
Para cada coluna que sai do `ColumnTransformer`, o nome vem prefixado pelo ramo
(`numericas__` / `categoricas__`). Retirando o prefixo:

* `missingindicator_<feature>` — coluna criada por `add_indicator=True`
  (marcador de "dado ausente") -> volta para `<feature>`. Com a base atual essa
  coluna não existe (não há nulo em campo numérico no treino — ver
  `pipeline.py`); a regra é defesa para o caso de passar a existir;
* nome idêntico a uma feature original (numérica ou booleana) -> ela mesma;
* qualquer outro caso é uma coluna de one-hot `<feature>_<categoria>` -> casa com
  a **feature mais longa** que seja prefixo do nome. O casamento pelo maior
  prefixo resolve sozinho eventuais nomes que são prefixo de outro
  (ex.: `faixa_apolice` vs. `faixa_valor_reclamado`).

A soma das contribuições é preservada: agregar é apenas reagrupar colunas, nunca
descartar valor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from quilha.preparacao import FEATURES

# Prefixo que o scikit-learn usa nas colunas de `add_indicator=True`.
PREFIXO_INDICADOR = "missingindicator_"


def mapear_features(
    nomes_transformados: list[str],
    features_originais: tuple[str, ...] | list[str] = FEATURES,
) -> dict[str, str]:
    """Mapeia cada coluna transformada para o nome da feature original.

    Recebe os nomes que saem do `ColumnTransformer` (ex.:
    `categoricas__tipo_sinistro_Colisão`) e devolve `{nome_transformado: feature}`.
    Levanta `ValueError` se um nome não puder ser atribuído a nenhuma feature —
    melhor falhar alto do que explicar a feature errada.
    """
    originais = list(features_originais)
    mapeamento: dict[str, str] = {}
    for nome in nomes_transformados:
        nucleo = nome.split("__", 1)[1] if "__" in nome else nome
        if nucleo.startswith(PREFIXO_INDICADOR):
            alvo = nucleo[len(PREFIXO_INDICADOR):]
        elif nucleo in originais:
            alvo = nucleo
        else:
            candidatos = [f for f in originais if nucleo.startswith(f + "_")]
            if not candidatos:
                raise ValueError(
                    f"Coluna '{nome}' não pôde ser atribuída a nenhuma feature original."
                )
            alvo = max(candidatos, key=len)
        mapeamento[nome] = alvo
    return mapeamento


def agregar_contribuicoes(
    valores: np.ndarray | list,
    nomes_transformados: list[str],
    features_originais: tuple[str, ...] | list[str] = FEATURES,
) -> pd.DataFrame:
    """Soma as contribuições das colunas codificadas de volta à feature original.

    `valores` tem forma `(n_amostras, n_colunas_transformadas)` (uma linha só
    também é aceita). Devolve um DataFrame com **uma coluna por feature original**,
    na ordem canônica de `features_originais`, e a soma total preservada.
    """
    v = np.asarray(valores, dtype=float)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    if v.shape[1] != len(nomes_transformados):
        raise ValueError(
            f"Matriz com {v.shape[1]} colunas, mas {len(nomes_transformados)} nomes informados."
        )

    mapeamento = mapear_features(nomes_transformados, features_originais)
    origem = np.array([mapeamento[n] for n in nomes_transformados], dtype=object)

    # Ordem de saída estável: a ordem canônica das features, só com as presentes.
    presentes = [f for f in features_originais if (origem == f).any()]
    saida = np.empty((v.shape[0], len(presentes)), dtype=float)
    for j, feature in enumerate(presentes):
        saida[:, j] = v[:, origem == feature].sum(axis=1)
    return pd.DataFrame(saida, columns=presentes)


def _dados_transformados(pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Aplica preparo + imputação e devolve a matriz densa com nomes de coluna."""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    transformadores = pipeline[:-1].set_output(transform="pandas")
    matriz = transformadores.transform(df)
    if not isinstance(matriz, pd.DataFrame):
        matriz = pd.DataFrame(np.asarray(matriz))
    return matriz


def _amostrar_fundo(fundo: pd.DataFrame, n_fundo: int, semente: int) -> pd.DataFrame:
    """Recorta o fundo do SHAP; ele é a referência contra a qual a contribuição é medida."""
    fundo_df = fundo if isinstance(fundo, pd.DataFrame) else pd.DataFrame(fundo)
    if len(fundo_df) > n_fundo:
        fundo_df = fundo_df.sample(n=n_fundo, random_state=semente)
    return fundo_df


def _extrair_classe_positiva(valores):
    """Fica com a coluna da classe 1 (fraude) de qualquer saída do SHAP.

    `predict_proba` devolve duas colunas; o SHAP pode entregar um vetor `(n, m)`,
    um arranjo `(n, m, 2)` ou uma lista por classe. Em todos os casos, o que
    interessa é a contribuição para a probabilidade de fraude.
    """
    if isinstance(valores, list):
        return np.asarray(valores[1])
    valores = np.asarray(valores)
    if valores.ndim == 3:
        return valores[:, :, 1]
    return valores


def _montar_explicador(pipeline: Pipeline, x_fundo: pd.DataFrame):
    """Cria o explicador model-agnóstico uma única vez, para reúso em lote."""
    import shap

    modelo = pipeline[-1]
    return shap.Explainer(modelo.predict_proba, shap.maskers.Independent(x_fundo.to_numpy()))


def explicar_sinistro(
    pipeline: Pipeline,
    linha: pd.DataFrame,
    fundo: pd.DataFrame,
    features_originais: tuple[str, ...] | list[str] = FEATURES,
    n_fundo: int = 100,
    semente: int = 42,
) -> pd.Series:
    """Contribuição de cada feature original para a probabilidade de UM sinistro.

    Recebe o Pipeline ajustado, **uma linha crua** (o chamado como chega no app) e
    um `fundo` de linhas cruas do treino (a referência contra a qual o SHAP mede a
    contribuição). Devolve uma `Series` indexada pelo nome da feature original,
    ordenada da maior contribuição para a menor (positivo = empurra para fraude).

    Modelo-agnóstico de propósito: usa `shap.Explainer` sobre `predict_proba`, então
    funciona igual com o placeholder da Fase 3 e com o modelo real da Fase 4.
    """
    linha_df = linha if isinstance(linha, pd.DataFrame) else pd.DataFrame([linha])
    linha_df = linha_df.head(1)

    x_linha = _dados_transformados(pipeline, linha_df)
    x_fundo = _dados_transformados(pipeline, _amostrar_fundo(fundo, n_fundo, semente))

    explicador = _montar_explicador(pipeline, x_fundo)
    valores = _extrair_classe_positiva(explicador(x_linha.to_numpy()).values)

    agregado = agregar_contribuicoes(valores, list(x_linha.columns), features_originais)
    return agregado.iloc[0].sort_values(ascending=False)


def contribuicoes_agregadas(
    pipeline: Pipeline,
    fundo: pd.DataFrame,
    amostra: pd.DataFrame,
    features_originais: tuple[str, ...] | list[str] = FEATURES,
    n_fundo: int = 100,
    semente: int = 42,
) -> pd.DataFrame:
    """SHAP de **várias linhas** de uma vez, já agregado por feature original.

    Mesma mecânica de `explicar_sinistro`, com o explicador criado uma única vez —
    é o que torna viável resumir a importância do campeão sobre uma amostra da fila.
    Devolve uma linha por sinistro e uma coluna por feature original.
    """
    amostra_df = amostra if isinstance(amostra, pd.DataFrame) else pd.DataFrame(amostra)
    x_amostra = _dados_transformados(pipeline, amostra_df)
    x_fundo = _dados_transformados(pipeline, _amostrar_fundo(fundo, n_fundo, semente))

    explicador = _montar_explicador(pipeline, x_fundo)
    valores = _extrair_classe_positiva(explicador(x_amostra.to_numpy()).values)

    return agregar_contribuicoes(valores, list(x_amostra.columns), features_originais)

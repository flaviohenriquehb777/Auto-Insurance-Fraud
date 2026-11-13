"""Alarme no silêncio — camada 3 da vírgula (Fase 6, Parte 1).

A camada 2 (`quilha.contrato`) recusa o dado inválido na porta do app. Esta camada
existe para o caso em que o dado **passa** pela porta e ainda assim é tratado em
silêncio: se um campo chega vazio por outro caminho — um script que chama
`pipeline.predict` direto, um campo novo da base que veio em branco — o Pipeline
**imputa a mediana do treino** e devolve uma predição de aparência normal.

`resumo_imputacoes` mede isso e devolve, por feature, a fração de linhas que foram
imputadas (nulos do `preparo` **mais** os marcadores `missingindicator_`, quando o
imputador os cria). A tela de governança do app (Parte 2) exibe o quadro e o
`alertas_imputacao` transforma o excesso em frase: **problema de dado, não de
fraude.** A fila do mês é sinalizada como suspeita — nunca como mais fraudulenta.
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from quilha.explicacao import PREFIXO_INDICADOR
from quilha.preparacao import FEATURES

# Acima desta fração de linhas imputadas, a feature é sinalizada. 5% é o valor de
# referência do prompt da fase: um em cada vinte sinistros entrando no modelo com
# valor inventado já não é ruído, é problema de origem do dado.
LIMIAR_IMPUTACAO_SUSPEITA = 0.05

# Prefixo que o `ColumnTransformer` do Pipeline dá à rama numérica na matriz final.
PREFIXO_RAMA_NUMERICA = "numericas__"

COLUNAS_RESUMO: tuple[str, ...] = (
    "feature",
    "n_linhas",
    "n_nulos",
    "n_indicador",
    "n_imputados",
    "fracao_imputada",
    "suspeita",
)


def resumo_imputacoes(
    pipeline: Pipeline,
    df: pd.DataFrame,
    limiar: float = LIMIAR_IMPUTACAO_SUSPEITA,
) -> pd.DataFrame:
    """Fração de linhas imputadas por feature, na ordem canônica de `FEATURES`.

    Conta como imputada a linha em que a feature saiu **nula** do `preparo` ou em
    que o marcador de ausente do imputador está ligado — as duas formas de o
    `SimpleImputer` substituir o valor por estatística do treino.
    """
    preparado = pipeline.named_steps["preparo"].transform(df)
    if not isinstance(preparado, pd.DataFrame):
        preparado = pd.DataFrame(preparado)

    imputacao = pipeline.named_steps["imputacao"]
    matriz = imputacao.set_output(transform="pandas").transform(preparado)

    n_linhas = len(preparado)
    linhas: list[dict] = []
    for feature in FEATURES:
        if feature in preparado.columns:
            nulos = preparado[feature].isna()
        else:
            nulos = pd.Series(False, index=preparado.index)

        nome_indicador = f"{PREFIXO_RAMA_NUMERICA}{PREFIXO_INDICADOR}{feature}"
        if nome_indicador in matriz.columns:
            indicador = matriz[nome_indicador].astype(bool)
        else:
            indicador = pd.Series(False, index=preparado.index)

        n_nulos = int(nulos.sum())
        n_indicador = int(indicador.sum())
        marcados = nulos.astype(bool) | indicador
        n_imputados = int(marcados.sum())
        fracao = n_imputados / n_linhas if n_linhas else 0.0

        linhas.append(
            {
                "feature": feature,
                "n_linhas": n_linhas,
                "n_nulos": n_nulos,
                "n_indicador": n_indicador,
                "n_imputados": n_imputados,
                "fracao_imputada": fracao,
                "suspeita": fracao > limiar,
            }
        )

    return pd.DataFrame(linhas, columns=list(COLUNAS_RESUMO))


def alertas_imputacao(
    resumo: pd.DataFrame,
    limiar: float = LIMIAR_IMPUTACAO_SUSPEITA,
) -> list[str]:
    """Transforma as features acima do limiar em avisos prontos para a tela.

    A frase é deliberada: o alerta acusa o **dado**, nunca a pessoa nem o sinistro.
    """
    suspeitas = resumo.loc[resumo["fracao_imputada"] > limiar]
    return [
        f"{registro.feature}: {registro.fracao_imputada:.1%} das linhas imputadas "
        f"(limiar {limiar:.0%}) — suspeita de problema de dado, não de fraude."
        for registro in suspeitas.itertuples()
    ]

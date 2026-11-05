"""Pipeline do scikit-learn: preparo -> imputação -> modelo (Fase 3 do CRISP-DM).

Este é o **único** objeto que o app carrega. Todo o tratamento mora dentro dele —
nada de transformação fora do objeto salvo. O app recebe o chamado cru, chama
`predict_proba` e pronto.

Estrutura
---------
    Pipeline([
        ("preparo",   FunctionTransformer(preparar)),   # 1. Preparo determinístico
        ("imputacao", ColumnTransformer([...])),        # 2. Imputação por tipo
        ("modelo",    ...),                             # 3. Modelo (Fase 4)
    ])

Imputação, com estratégia diferenciada por tipo
-----------------------------------------------
* Numéricas -> `SimpleImputer(strategy="median", add_indicator=True)`.
  Mediana por robustez: `dias_de_apolice` tem assimetria 1,50 e
  `valor_reclamado` 3,51 (Fase 2, seção 9) — a média seria arrastada pela cauda.
  O `add_indicator=True` fica como reserva para nulo real no treino (ver nota
  abaixo: com esta base ele não cria coluna nenhuma).
* Categóricas -> `SimpleImputer(strategy="most_frequent")`, **nunca** mediana:
  mediana de texto não existe. Seguido de
  `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` — categoria nova
  no app não quebra, e a matriz densa é o que o SHAP precisa.

Importante — o que o `add_indicator` **não** faz
-----------------------------------------------
Depois de `preparar`, a base **não tem** nulo em nenhuma coluna (a Fase 2 só
encontrou nulos em `boletim_ocorrencia`, que `preparar` transforma na categoria
"Não informado"). O `SimpleImputer` só cria coluna de indicador para as colunas
que tinham nulo **no `fit`**; sem nulo no treino, `indicator_.features_` sai
vazio e **nenhuma coluna de indicador é criada em lugar nenhum** — nem no treino,
nem em um chamado novo do app (verificado: matriz de um chamado com dois campos
vazios sai com o mesmo número de colunas da matriz de treino).

Consequência prática, dita sem maquiagem: um campo numérico em branco no app é
imputado pela **mediana do treino** e entra no modelo como valor típico — o modelo
não distingue "vazio" de "valor mediano", porque nunca viu uma linha vazia. A
bandeira fica como reserva para o dia em que uma coluna numérica tiver nulo real
no `fit`: aí ela passa a existir e a agregação do SHAP já sabe devolvê-la à
feature original (`explicacao.PREFIXO_INDICADOR`). O único campo com branco real
na base é o `boletim_ocorrencia`, que é categórico e tem tratamento próprio —
categoria explícita "Não informado" (`docs/03_preparacao_dados.md`, seção 3-d).

Nada de ajuste fora do treino: imputador e `OneHotEncoder` são ajustados só no
treino (é o `Pipeline` quem garante) e apenas aplicados no teste.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from quilha.dados import PASTA_MODELOS
from quilha.preparacao import (
    COLUNAS_BOOLEANAS,
    COLUNAS_CATEGORICAS,
    COLUNAS_NUMERICAS,
    FEATURES,
    preparar,
)

# Entrada da rama numérica: as contínuas/contagens mais a bandeira booleana
# (tratada como 0/1, com mediana = voto da maioria).
COLUNAS_RAMA_NUMERICA: tuple[str, ...] = COLUNAS_NUMERICAS + COLUNAS_BOOLEANAS

NOME_ARQUIVO_PIPELINE = "pipeline_fraude.joblib"
CAMINHO_PIPELINE: Path = PASTA_MODELOS / NOME_ARQUIVO_PIPELINE


# ---------------------------------------------------------------------------
# Construção do Pipeline
# ---------------------------------------------------------------------------
def construir_pipeline(modelo=None) -> Pipeline:
    """Monta o Pipeline serializável. `modelo=None` usa o placeholder da Fase 3.

    O placeholder (`DummyClassifier`) existe só para provar que o Pipeline roda de
    ponta a ponta; ele não aprende nada. O modelo de verdade entra na Fase 4.
    """
    if modelo is None:
        modelo = DummyClassifier(strategy="prior")

    rama_numerica = SimpleImputer(strategy="median", add_indicator=True)
    rama_categorica = Pipeline(
        [
            ("imputacao", SimpleImputer(strategy="most_frequent")),
            ("codificacao", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    imputacao = ColumnTransformer(
        [
            ("numericas", rama_numerica, list(COLUNAS_RAMA_NUMERICA)),
            ("categoricas", rama_categorica, list(COLUNAS_CATEGORICAS)),
        ]
    )

    pipeline = Pipeline(
        [
            ("preparo", FunctionTransformer(preparar)),
            ("imputacao", imputacao),
            ("modelo", modelo),
        ]
    )
    # Matriz densa com nomes de coluna: é o que o SHAP precisa para explicar.
    return pipeline.set_output(transform="pandas")


# ---------------------------------------------------------------------------
# Persistência e registro de versões
# ---------------------------------------------------------------------------
def versoes() -> dict:
    """Versões que precisam bater entre onde o Pipeline foi salvo e onde ele roda."""
    return {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
    }


def salvar_pipeline(pipeline: Pipeline, caminho: Path | str = CAMINHO_PIPELINE) -> Path:
    """Salva o Pipeline em disco e grava ao lado o registro de versões e features."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, caminho)

    registro = versoes()
    registro["features"] = list(FEATURES)
    caminho.with_suffix(".versoes.json").write_text(
        json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return caminho


def carregar_pipeline(caminho: Path | str = CAMINHO_PIPELINE) -> Pipeline:
    """Carrega o Pipeline salvo. O app usa exatamente isto."""
    return joblib.load(Path(caminho))

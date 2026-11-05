"""Testes da Fase 3 do CRISP-DM — preparação dos dados, Pipeline e agregação SHAP.

Cobre os critérios de aceite da fase: equivalência uma-linha vs. base inteira,
independência da ordem das colunas, tolerância a dado sujo, ausência de vazamento,
Pipeline ponta a ponta e round-trip do artefato salvo em disco.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quilha.dados import carregar_base_crua, dividir_temporal, encontrar_raiz
from quilha.explicacao import agregar_contribuicoes, mapear_features
from quilha.pipeline import (
    CAMINHO_PIPELINE,
    carregar_pipeline,
    construir_pipeline,
    salvar_pipeline,
)
from quilha.preparacao import COLUNAS_NUMERICAS, FEATURES, preparar

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def base_crua() -> pd.DataFrame:
    """Base inteira, do jeito que chega do CSV (tudo texto)."""
    return carregar_base_crua()


@pytest.fixture(scope="session")
def uma_linha_crua(base_crua: pd.DataFrame) -> pd.DataFrame:
    """Uma única linha crua, com o índice original preservado."""
    return base_crua.iloc[[7]].copy()


@pytest.fixture(scope="session")
def treino_teste(base_crua: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divisão temporal oficial da fase."""
    return dividir_temporal(base_crua)


def _sem_gabarito(df: pd.DataFrame) -> pd.DataFrame:
    """Remove as colunas que o app nunca recebe (gabarito e valor pago)."""
    return df.drop(columns=["fraude", "valor_indenizado"], errors="ignore")


# ---------------------------------------------------------------------------
# Parte 1 — a função `preparar`
# ---------------------------------------------------------------------------


def test_uma_linha_igual_a_linha_da_base(base_crua, uma_linha_crua):
    """Critério 1: preparar(uma_linha) == a linha correspondente da base inteira."""
    da_base = preparar(base_crua)
    isolada = preparar(uma_linha_crua)

    assert list(isolada.columns) == list(da_base.columns)
    assert list(isolada.dtypes) == list(da_base.dtypes)
    pd.testing.assert_frame_equal(isolada, da_base.iloc[[7]], check_dtype=True)


def test_ordem_das_colunas_nao_importa(base_crua):
    """A saída é idêntica se as colunas de entrada vierem embaralhadas."""
    referencia = preparar(base_crua)
    ordem = list(base_crua.columns)[::-1]
    embaralhada = preparar(base_crua[ordem])
    pd.testing.assert_frame_equal(embaralhada, referencia, check_dtype=True)


def test_idempotente_na_mesma_entrada(base_crua):
    """Rodar duas vezes sobre a MESMA entrada crua dá exatamente o mesmo resultado."""
    primeira = preparar(base_crua)
    segunda = preparar(base_crua)
    pd.testing.assert_frame_equal(primeira, segunda, check_dtype=True)


def test_nao_cria_nulos_nos_campos_numericos(base_crua):
    """Critério 2: `preparar` não cria nulo novo em campo numérico."""
    saida = preparar(base_crua)
    nulos = saida[list(COLUNAS_NUMERICAS)].isna().sum().sum()
    assert nulos == 0, f"apareceram {nulos} nulos em colunas numéricas"


def test_aceita_valor_reclamado_em_tres_formatos(uma_linha_crua):
    """`valor_reclamado` entra como "8837,07", "8837.07" ou 8837.07 — mesma saída."""
    linha = uma_linha_crua.copy()
    resultados = []
    for valor in ["8837,07", "8837.07", 8837.07]:
        linha["valor_reclamado"] = valor
        resultados.append(preparar(linha)["valor_reclamado"].iloc[0])

    assert all(pd.notna(r) for r in resultados)
    assert resultados[0] == resultados[1] == resultados[2] == pytest.approx(8837.07)


def test_sem_colunas_vazadas(base_crua):
    """Critério 3: `fraude` (gabarito) e `valor_indenizado` (vazamento) nunca saem."""
    saida = preparar(base_crua)
    assert "fraude" not in saida.columns
    assert "valor_indenizado" not in saida.columns


def test_ignora_coluna_extra(base_crua):
    """Coluna desconhecida que o app mandar não muda a saída nem quebra."""
    linha = base_crua.iloc[[7]].copy()
    linha["campo_de_controle_do_app"] = "qualquer coisa"
    pd.testing.assert_frame_equal(preparar(linha), preparar(base_crua.iloc[[7]]), check_dtype=True)


def test_robusto_a_dado_sujo():
    """Campo vazio, data malformada e texto inválido viram nulo — sem exceção."""
    linha = pd.DataFrame(
        {
            "id_sinistro": ["X1"],
            "data_ocorrencia": ["31/02/2026"],  # data inexistente
            "data_abertura": [""],  # vazio
            "idade_segurado": ["abc"],
            "dias_de_apolice": [""],
            "sinistros_12m": ["??"],
            "idade_veiculo": [None],
            "valor_fipe": ["R$ 50.000"],
            "valor_reclamado": [""],
            "hora_ocorrencia": ["99:99"],
            "canal_abertura": [""],
            "tipo_sinistro": [None],
            "uf": ["  "],
            "boletim_ocorrencia": [""],
            "terceiro_identificado": ["talvez"],
        }
    )
    saida = preparar(linha)  # não pode levantar

    assert saida.shape[0] == 1
    assert pd.isna(saida["hora"].iloc[0])
    assert pd.isna(saida["valor_reclamado"].iloc[0])
    assert pd.isna(saida["data_abertura"].iloc[0])
    # O branco do boletim vira a categoria explícita, nunca nulo (item d da fase).
    assert saida["boletim_ocorrencia"].iloc[0] == "Não informado"


def test_linha_completamente_vazia_nao_quebra():
    """Uma linha sem nenhum campo preenchido segue a mesma trilha, sem exceção."""
    vazia = pd.DataFrame([{c: None for c in FEATURES}])
    saida = preparar(vazia)
    assert saida.shape == (1, len(FEATURES))


# ---------------------------------------------------------------------------
# Parte 4 — agregação SHAP por feature original
# ---------------------------------------------------------------------------


NOMES_SINTETICOS = [
    "numericas__idade_segurado",
    "numericas__valor_fipe",
    "numericas__missingindicator_valor_fipe",
    "categoricas__tipo_sinistro_Colisão",
    "categoricas__tipo_sinistro_Roubo",
    "categoricas__uf_SP",
    "categoricas__faixa_valor_reclamado_40 a 50",
]
VALORES_SINTETICOS = np.array([[1.0, 2.0, 0.5, -3.0, 4.0, 1.5, -2.0]])


def test_mapeia_one_hot_e_indicador_para_a_feature_original():
    """O one-hot e o marcador de ausente voltam para o nome original."""
    mapa = mapear_features(NOMES_SINTETICOS)
    assert mapa["categoricas__tipo_sinistro_Colisão"] == "tipo_sinistro"
    assert mapa["categoricas__tipo_sinistro_Roubo"] == "tipo_sinistro"
    assert mapa["numericas__missingindicator_valor_fipe"] == "valor_fipe"
    assert mapa["categoricas__faixa_valor_reclamado_40 a 50"] == "faixa_valor_reclamado"


def test_mapeamento_falha_alto_em_nome_desconhecido():
    """Nome que não casa com nenhuma feature deve levantar, não chutar."""
    with pytest.raises(ValueError):
        mapear_features(["categoricas__coluna_inexistente_xyz"])


def test_agregacao_devolve_uma_coluna_por_feature_original():
    """Uma coluna por feature original, na ordem canônica, com a soma preservada."""
    agregado = agregar_contribuicoes(VALORES_SINTETICOS, NOMES_SINTETICOS)

    assert agregado.shape == (1, 5)
    assert list(agregado.columns) == [
        "idade_segurado",
        "valor_fipe",
        "tipo_sinistro",
        "uf",
        "faixa_valor_reclamado",
    ]
    # one-hot somado de volta à feature
    assert agregado["tipo_sinistro"].iloc[0] == pytest.approx(-3.0 + 4.0)
    # coluna de valor + marcador de ausente
    assert agregado["valor_fipe"].iloc[0] == pytest.approx(2.0 + 0.5)
    # agregar nunca descarta contribuição
    assert agregado.to_numpy().sum() == pytest.approx(VALORES_SINTETICOS.sum())


def test_agregacao_aceita_varias_linhas():
    """Duas amostras: agregação por linha, cada linha somando o próprio total."""
    valores = np.vstack([VALORES_SINTETICOS[0], VALORES_SINTETICOS[0] * 2])
    agregado = agregar_contribuicoes(valores, NOMES_SINTETICOS)
    assert agregado.shape == (2, 5)
    assert agregado.iloc[1].sum() == pytest.approx(valores[1].sum())


def test_agregacao_recusa_matriz_incompativel():
    """Matriz com nº de colunas diferente do nº de nomes deve falhar."""
    with pytest.raises(ValueError):
        agregar_contribuicoes(VALORES_SINTETICOS, NOMES_SINTETICOS[:-1])


# ---------------------------------------------------------------------------
# Parte 3 — Pipeline ponta a ponta e persistência
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pipeline_ajustado(treino_teste):
    """Pipeline da fase ajustado SÓ no treino (nunca no teste)."""
    treino, _ = treino_teste
    pipeline = construir_pipeline()
    pipeline.fit(_sem_gabarito(treino), treino["fraude"].astype(int))
    return pipeline


def test_pipeline_roda_de_ponta_a_ponta(pipeline_ajustado, treino_teste):
    """fit no treino e predict_proba em UMA linha crua do teste, sem erro."""
    _, teste = treino_teste
    linha = _sem_gabarito(teste).iloc[[0]]

    probabilidades = pipeline_ajustado.predict_proba(linha)

    assert probabilidades.shape == (1, 2)
    assert probabilidades.sum() == pytest.approx(1.0)
    assert 0.0 <= probabilidades[0, 1] <= 1.0


def test_pipeline_salvo_e_recarregado_preve_o_mesmo(pipeline_ajustado, treino_teste, tmp_path):
    """Critério 5: salvar, recarregar do disco e continuar prevendo o mesmo valor."""
    _, teste = treino_teste
    linha = _sem_gabarito(teste).iloc[[0]]
    esperado = pipeline_ajustado.predict_proba(linha)

    destino = salvar_pipeline(pipeline_ajustado, tmp_path / "pipeline.joblib")
    assert destino.exists()
    recarregado = carregar_pipeline(destino)

    obtido = recarregado.predict_proba(linha)
    np.testing.assert_allclose(obtido, esperado)


def test_pipeline_aceita_categoria_nunca_vista(pipeline_ajustado, treino_teste):
    """O app pode mandar categoria nova: `handle_unknown="ignore"` não quebra."""
    _, teste = treino_teste
    linha = _sem_gabarito(teste).iloc[[0]].copy()
    linha["tipo_sinistro"] = "Categoria Que Não Existe"
    linha["canal_abertura"] = "Canal Novo"

    probabilidades = pipeline_ajustado.predict_proba(linha)
    assert probabilidades.shape == (1, 2)


def test_pipeline_aceita_campo_vazio(pipeline_ajustado, treino_teste):
    """Chamado novo com campo em branco: `preparar` gera nulo e o imputador resolve."""
    _, teste = treino_teste
    linha = _sem_gabarito(teste).iloc[[0]].copy()
    for coluna in ["valor_reclamado", "valor_fipe", "idade_segurado", "boletim_ocorrencia"]:
        linha[coluna] = ""

    probabilidades = pipeline_ajustado.predict_proba(linha)
    assert probabilidades.shape == (1, 2)


def test_indicador_de_ausente_aparece_quando_ha_nulo_no_treino(treino_teste):
    """Caminho hoje **não** exercitado pelo treino real, testado de propósito.

    Com a base atual não existe nulo em campo numérico, então o `add_indicator`
    não cria coluna nenhuma. Aqui a coluna tem nulo real no `fit`: o indicador
    tem de aparecer e a agregação do SHAP tem de devolvê-lo à feature original.
    """
    treino, _ = treino_teste
    x_treino = _sem_gabarito(treino).copy()
    x_treino.loc[x_treino.index[:50], "valor_fipe"] = ""  # nulo real no fit

    pipeline = construir_pipeline()
    pipeline.fit(x_treino, treino["fraude"].astype(int))

    # Aplica o preparo e a imputação já ajustados, pedindo a matriz com nomes.
    imputacao = pipeline.named_steps["imputacao"].set_output(transform="pandas")
    nomes = list(imputacao.transform(preparar(x_treino.head(5))).columns)
    assert "numericas__missingindicator_valor_fipe" in nomes
    assert mapear_features(nomes)["numericas__missingindicator_valor_fipe"] == "valor_fipe"


def test_local_padrao_do_artefato_esta_dentro_do_repositorio():
    """Contrato do deploy: o `.joblib` mora **dentro** do repositório.

    O Streamlit Cloud clona o repositório e não enxerga nenhum armazenamento de
    artefatos separado — é a razão da exceção no `.gitignore`. Se a pasta dos
    modelos apontar para fora da raiz do projeto, o app quebra no deploy.
    """
    raiz = encontrar_raiz(Path(__file__))

    assert CAMINHO_PIPELINE.is_relative_to(raiz)
    assert CAMINHO_PIPELINE.parent == raiz / "models"


def test_salvar_pipeline_cria_a_pasta_e_grava_o_registro_de_versoes(pipeline_ajustado, tmp_path):
    """`salvar_pipeline` cria a pasta que faltar e grava o registro ao lado.

    O registro de versões é o que documenta com qual ambiente o artefato foi
    gerado; sem ele, o `.joblib` não é reproduzível.
    """
    destino = tmp_path / "pasta" / "que" / "ainda" / "nao" / "existe" / "pipeline.joblib"

    caminho = salvar_pipeline(pipeline_ajustado, destino)

    assert caminho.exists()
    registro = json.loads(caminho.with_suffix(".versoes.json").read_text(encoding="utf-8"))
    assert registro["features"] == list(FEATURES)
    assert {"python", "scikit_learn", "pandas", "numpy", "gerado_em"} <= set(registro)

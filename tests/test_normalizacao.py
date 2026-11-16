"""Testes da normalização de representação — melhoria B (spec 07).

O contrato de teste deste arquivo trava **o limite** da melhoria B:

* o que é **representação** (`"1.234,56"`, `" sp "`, `"15/08/2026"`) é reescrito na
  forma canônica **e relatado** — nada às escondidas;
* o que **não** é representação (`"abc"`, `"99:99"`, `"Telefone"`, `"1.234"`) sai
  intacto do normalizador e é recusado pelo contrato, como sempre foi;
* a ordem `normalizar → contrato → modelo` é o que **fecha a falha silenciosa**: sem
  ela, `" sp "` chega ao encoder como categoria desconhecida e o escore muda em
  silêncio (o teste `..._fecha_a_falha_silenciosa` mede exatamente isso).
"""

from __future__ import annotations

import pandas as pd
import pytest

from quilha.contrato import CAMPOS_ABERTURA, validar_payload
from quilha.normalizacao import (
    COLUNAS_RELATORIO,
    REGRA_CATEGORICA,
    REGRA_DATA,
    REGRA_MILHAR,
    normalizar,
)
from quilha.pipeline import CAMINHO_PIPELINE, carregar_pipeline

# ---------------------------------------------------------------------------
# Um sinistro de referência, válido por construção
# ---------------------------------------------------------------------------
SINISTRO_VALIDO: dict[str, str] = {
    "data_ocorrencia": "2026-07-18",
    "hora_ocorrencia": "13:53",
    "data_abertura": "2026-07-18",
    "canal_abertura": "App",
    "tipo_sinistro": "Colisão",
    "uf": "SP",
    "idade_segurado": "45",
    "dias_de_apolice": "120",
    "sinistros_12m": "0",
    "idade_veiculo": "3",
    "valor_fipe": "50000,00",
    "valor_reclamado": "8837,07",
    "boletim_ocorrencia": "Sim",
    "terceiro_identificado": "Não",
}


def _lote(**alteracoes: str) -> pd.DataFrame:
    """Lote de uma linha, com os campos que se quiser trocar."""
    return pd.DataFrame([{**SINISTRO_VALIDO, **alteracoes}])


@pytest.fixture(scope="session")
def campeao():
    """O artefato auditado — o mesmo que o app carrega."""
    return carregar_pipeline(CAMINHO_PIPELINE)


# ---------------------------------------------------------------------------
# B1 · número pt-BR com os dois separadores
# ---------------------------------------------------------------------------
def test_b1_milhar_pt_br_e_reescrito_e_relatado():
    limpo, relatorio = normalizar(_lote(valor_reclamado="1.234,56"))

    assert limpo.loc[0, "valor_reclamado"] == "1234.56"
    assert relatorio.to_dict(orient="records") == [
        {
            "linha": 0,
            "campo": "valor_reclamado",
            "de": "1.234,56",
            "para": "1234.56",
            "regra": REGRA_MILHAR,
        }
    ]


def test_b1_aceita_varios_grupos_de_milhar():
    limpo, _ = normalizar(_lote(valor_reclamado="1.234.567,89"))

    assert limpo.loc[0, "valor_reclamado"] == "1234567.89"


@pytest.mark.parametrize("valor", ["8837,07", "8837.07", "1.234"])
def test_b1_nao_toca_no_numero_que_nao_e_o_milhar_inequivoco(valor):
    """A vírgula corrente não muda e o ponto sozinho (`"1.234"`) continua ambíguo."""
    limpo, relatorio = normalizar(_lote(valor_reclamado=valor))

    assert limpo.loc[0, "valor_reclamado"] == valor
    assert relatorio.empty


# ---------------------------------------------------------------------------
# B2 · categórica com caixa/espaço
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "campo, entrada, canonico",
    [
        ("uf", " sp ", "SP"),
        ("canal_abertura", "app", "App"),
        ("tipo_sinistro", "COLISÃO", "Colisão"),
        ("terceiro_identificado", "não se aplica", "Não se aplica"),
    ],
)
def test_b2_categorica_com_caixa_ou_espaco_e_reescrita(campo, entrada, canonico):
    limpo, relatorio = normalizar(_lote(**{campo: entrada}))

    assert limpo.loc[0, campo] == canonico
    assert relatorio.to_dict(orient="records") == [
        {"linha": 0, "campo": campo, "de": entrada, "para": canonico, "regra": REGRA_CATEGORICA}
    ]


# ---------------------------------------------------------------------------
# B3 · data DD/MM/AAAA (convenção pt-BR)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "entrada, iso", [("15/08/2026", "2026-08-15"), ("03/04/2026", "2026-04-03")]
)
def test_b3_data_pt_br_e_reescrita_para_iso(entrada, iso):
    limpo, relatorio = normalizar(_lote(data_ocorrencia=entrada))

    assert limpo.loc[0, "data_ocorrencia"] == iso
    assert relatorio.to_dict(orient="records") == [
        {"linha": 0, "campo": "data_ocorrencia", "de": entrada, "para": iso, "regra": REGRA_DATA}
    ]


def test_b3_iso_ja_canonico_nao_aparece_no_relatorio():
    _, relatorio = normalizar(_lote(data_ocorrencia="2026-07-18"))

    assert relatorio.empty


# ---------------------------------------------------------------------------
# O que NÃO é representação passa intacto e o contrato recusa
# ---------------------------------------------------------------------------
def test_irrecuperaveis_passam_intactos_e_o_contrato_recusa():
    lote = _lote(idade_segurado="abc", hora_ocorrencia="99:99", canal_abertura="Telefone")

    limpo, relatorio = normalizar(lote)

    assert relatorio.empty, "nada aqui é representação: não há o que reescrever"
    assert limpo.loc[0, "idade_segurado"] == "abc"
    assert limpo.loc[0, "hora_ocorrencia"] == "99:99"
    assert limpo.loc[0, "canal_abertura"] == "Telefone"

    erros = validar_payload(limpo)
    assert {erro.campo for erro in erros} == {
        "idade_segurado",
        "hora_ocorrencia",
        "canal_abertura",
    }


def test_data_inexistente_nao_e_reescrita_nem_aceita():
    limpo, relatorio = normalizar(_lote(data_ocorrencia="31/02/2026"))

    assert relatorio.empty
    assert limpo.loc[0, "data_ocorrencia"] == "31/02/2026"
    assert {erro.campo for erro in validar_payload(limpo)} == {"data_ocorrencia"}


# ---------------------------------------------------------------------------
# Contrato do próprio normalizador
# ---------------------------------------------------------------------------
def test_relatorio_vazio_vem_com_as_colunas():
    """A tela distingue "nada mudou" de "não houve normalização" pela forma."""
    _, relatorio = normalizar(_lote())

    assert relatorio.empty
    assert list(relatorio.columns) == list(COLUNAS_RELATORIO)


def test_normalizar_nao_altera_a_entrada():
    cru = _lote(valor_reclamado="1.234,56", uf=" sp ", data_ocorrencia="15/08/2026")

    normalizar(cru)

    assert cru.loc[0, "valor_reclamado"] == "1.234,56"
    assert cru.loc[0, "uf"] == " sp "
    assert cru.loc[0, "data_ocorrencia"] == "15/08/2026"


def test_normalizar_e_idempotente():
    primeira, relatorio_1 = normalizar(_lote(valor_reclamado="1.234,56", uf=" sp "))
    segunda, relatorio_2 = normalizar(primeira)

    pd.testing.assert_frame_equal(primeira, segunda)
    assert not relatorio_1.empty
    assert relatorio_2.empty


def test_normalizar_ignora_colunas_fora_do_contrato():
    lote = _lote()
    lote["id_sinistro"] = " sp "
    lote["fraude"] = "1.234,56"

    limpo, relatorio = normalizar(lote)

    assert relatorio.empty
    assert limpo.loc[0, "id_sinistro"] == " sp "


def test_normalizar_exige_dataframe():
    with pytest.raises(TypeError):
        normalizar(SINISTRO_VALIDO)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# A ordem `normalizar → contrato → modelo` fecha a falha silenciosa
# ---------------------------------------------------------------------------
def test_normalizacao_fecha_a_falha_silenciosa_da_categoria(campeao):
    """Sem normalizar, `" sp "` vira categoria desconhecida e o escore muda sem aviso.

    Com a ordem do app (normalizar antes do modelo), o escore de `" sp "` é
    **idêntico** ao de `"SP"` — e o relatório diz por quê.
    """
    cru = _lote(uf=" sp ")
    canonico = _lote(uf="SP")
    limpo, relatorio = normalizar(cru)

    escore_cru = float(campeao.predict_proba(cru[list(CAMPOS_ABERTURA)].astype(str))[0, 1])
    escore_canonico = float(campeao.predict_proba(canonico[list(CAMPOS_ABERTURA)])[0, 1])
    escore_limpo = float(campeao.predict_proba(limpo[list(CAMPOS_ABERTURA)])[0, 1])

    assert escore_cru != pytest.approx(escore_canonico), (
        "o teste perdeu o sentido: a categoria crua deixou de mudar o escore"
    )
    assert escore_limpo == pytest.approx(escore_canonico)
    assert len(relatorio) == 1

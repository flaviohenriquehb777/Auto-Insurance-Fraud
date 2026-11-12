"""Testes do contrato de dados e do monitor de imputação — Fase 6, Parte 1.

Camada 2 (contrato) e camada 3 (alarme no silêncio) da "falha silenciosa da
vírgula". O objetivo deste arquivo não é só provar que o contrato funciona hoje:
é **travar** o comportamento. Se alguém afrouxar `preparacao._numero` para aceitar
`"21.700,00"`, ou passar a imputar em vez de recusar na fronteira, o CI fica
vermelho — que é exatamente a hora em que se quer descobrir isso.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from quilha.contrato import (
    CAMPOS_ABERTURA,
    CAMPOS_CATEGORICOS,
    CAMPOS_DATA,
    CAMPOS_HORA,
    CAMPOS_NUMERICOS,
    DOMINIOS_CATEGORICOS,
    validar_payload,
)
from quilha.dados import carregar_base_crua, dividir_temporal
from quilha.monitoramento import (
    COLUNAS_RESUMO,
    LIMIAR_IMPUTACAO_SUSPEITA,
    alertas_imputacao,
    resumo_imputacoes,
)
from quilha.pipeline import CAMINHO_PIPELINE, carregar_pipeline
from quilha.preparacao import _numero

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


def _payload(**alteracoes: str) -> pd.DataFrame:
    """Payload de uma linha, com os campos que se quiser trocar."""
    campos = {**SINISTRO_VALIDO, **alteracoes}
    return pd.DataFrame([campos])


def _campos(erros) -> set[str]:
    """Conjunto dos campos recusados — o resumo legível de um lote de erros."""
    return {erro.campo for erro in erros}


def _texto(erros) -> str:
    return " | ".join(str(erro) for erro in erros)


@pytest.fixture(scope="session")
def base_crua() -> pd.DataFrame:
    """Base inteira, do jeito que chega do CSV (tudo texto)."""
    return carregar_base_crua()


@pytest.fixture(scope="session")
def campeao():
    """O artefato auditado da Parte 0 — o mesmo que o app carrega."""
    return carregar_pipeline(CAMINHO_PIPELINE)


# ---------------------------------------------------------------------------
# A bateria adversarial (o coração da Parte 1)
# ---------------------------------------------------------------------------
# Entrada em `_numero`, o que ele devolve hoje, se o CONTRATO aceita, e a lição.
BATERIA_ADVERSARIAL: tuple[tuple[str | None, float | None, bool, str], ...] = (
    ("8837,07", 8837.07, True, "formato pt-BR correto: aceito"),
    ("8837.07", 8837.07, True, "formato com ponto também converte"),
    ("21.700,00", None, False, "milhar pt-BR não é tratado — decisão do projeto"),
    ("R$ 100,00", None, False, "texto com moeda não é número"),
    ("1.234", 1.234, True, "lido como 1,234 — risco de 1.000× documentado"),
    ("", None, False, "vazio: o contrato não imputa na fronteira"),
    (None, None, False, "ausente: o contrato não imputa na fronteira"),
)


@pytest.mark.parametrize(
    "entrada, numero_esperado, aceito, licao",
    BATERIA_ADVERSARIAL,
    ids=[caso[3] for caso in BATERIA_ADVERSARIAL],
)
def test_bateria_adversarial_do_numero(entrada, numero_esperado, aceito, licao):
    """Cada linha da bateria: o que o parser faz e o que o contrato decide."""
    convertido = _numero(pd.Series([entrada], dtype=object)).iloc[0]

    if numero_esperado is None:
        assert pd.isna(convertido), f"{licao}: esperava nulo, veio {convertido!r}"
    else:
        assert convertido == pytest.approx(numero_esperado)

    erros = validar_payload(_payload(valor_reclamado=entrada))  # type: ignore[arg-type]
    if aceito:
        assert erros == [], f"{licao}: esperava aceite, veio recusa — {_texto(erros)}"
    else:
        assert erros, f"{licao}: esperava recusa e o payload passou"
        assert _campos(erros) == {"valor_reclamado"}
        assert erros[0].linha == 0


# ---------------------------------------------------------------------------
# Payload válido e equivalência com a linha crua
# ---------------------------------------------------------------------------
def test_sinistro_de_referencia_e_aceito():
    assert validar_payload(_payload()) == []


def test_lote_real_da_base_inteira_e_aceito(base_crua):
    """As 60.000 linhas do case passam pelo contrato sem uma única recusa."""
    lote = base_crua[list(CAMPOS_ABERTURA)]
    erros = validar_payload(lote)
    assert erros == [], _texto(erros[:5])


def test_payload_aprovado_preve_igual_a_linha_crua_completa(base_crua, campeao):
    """O contrato não muda o dado: as 14 colunas bastam para o mesmo escore."""
    linha = base_crua.iloc[[7]]
    aprovado = linha[list(CAMPOS_ABERTURA)]
    assert validar_payload(aprovado) == []

    np.testing.assert_allclose(
        campeao.predict_proba(aprovado), campeao.predict_proba(linha)
    )


def test_dicionario_de_um_sinistro_e_aceito():
    """A tela individual manda um dicionário; o contrato aceita os dois formatos."""
    assert validar_payload(SINISTRO_VALIDO) == []


def test_payload_de_tipo_invalido_e_recusado():
    with pytest.raises(TypeError):
        validar_payload("valor_reclamado=8837,07")


# ---------------------------------------------------------------------------
# Estrutura do contrato: as 14 colunas e os domínios
# ---------------------------------------------------------------------------
def test_as_14_colunas_de_abertura_cobrem_dominios_numeros_datas_e_horas():
    assert len(CAMPOS_ABERTURA) == 14
    assert set(CAMPOS_ABERTURA) == (
        set(CAMPOS_CATEGORICOS) | set(CAMPOS_DATA) | set(CAMPOS_HORA) | set(CAMPOS_NUMERICOS)
    )


def test_dominios_batem_com_a_base(base_crua):
    """Os domínios são os da base real — nem mais frouxos, nem mais estreitos."""
    for campo, dominio in DOMINIOS_CATEGORICOS.items():
        na_base = set(base_crua[campo].dropna().astype(str).str.strip().unique())
        assert na_base <= set(dominio), f"{campo}: a base tem valor fora do domínio"
        assert len(na_base) == len(set(dominio) - {""}), f"{campo}: domínio com valor a mais"


def test_coluna_ausente_e_recusada_com_marcador_de_ausente():
    erros = validar_payload(_payload().drop(columns=["valor_reclamado"]))

    assert len(erros) == 1
    assert erros[0].campo == "valor_reclamado"
    assert erros[0].valor_recebido == "(ausente)"
    assert "ausente" in erros[0].motivo
    assert erros[0].linha is None


def test_varias_colunas_ausentes_saem_todas_de_uma_vez():
    payload = _payload().drop(columns=["uf", "valor_fipe", "canal_abertura"])
    assert _campos(validar_payload(payload)) == {"uf", "valor_fipe", "canal_abertura"}


# ---------------------------------------------------------------------------
# Recusas por domínio, vazio, data e hora
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "campo, valor",
    [
        ("canal_abertura", "E-mail"),
        ("tipo_sinistro", "Batida"),
        ("uf", "XX"),
        ("terceiro_identificado", "Talvez"),
        ("boletim_ocorrencia", "Em análise"),
    ],
)
def test_valor_fora_do_dominio_e_recusado(campo, valor):
    erros = validar_payload(_payload(**{campo: valor}))

    assert _campos(erros) == {campo}
    assert erros[0].valor_recebido == valor
    assert "domínio" in erros[0].motivo
    assert erros[0].linha == 0


@pytest.mark.parametrize("campo", ["canal_abertura", "uf", "terceiro_identificado"])
def test_categorico_vazio_e_recusado(campo):
    erros = validar_payload(_payload(**{campo: ""}))

    assert _campos(erros) == {campo}
    assert erros[0].valor_recebido == "(vazio)"
    assert "vazio" in erros[0].motivo


def test_boletim_em_branco_e_aceito_e_vira_nao_informado(campeao):
    """O branco do boletim é caso de negócio: passa no contrato e nunca é imputado."""
    payload = _payload(boletim_ocorrencia="")
    assert validar_payload(payload) == []

    preparado = campeao.named_steps["preparo"].transform(payload)
    assert preparado["boletim_ocorrencia"].iloc[0] == "Não informado"


@pytest.mark.parametrize(
    "campo, valor",
    [
        ("data_ocorrencia", "31/02/2026"),  # dia inexistente
        ("data_ocorrencia", "ontem"),
        ("data_ocorrencia", "18/07/2026"),  # dia/mês/ano: formato não explícito
        ("data_ocorrencia", "03/04/2026"),  # ambíguo: o pandas leria 4 de março
        ("data_abertura", ""),
        ("data_abertura", "13:53"),  # hora trocada de campo virava "hoje às 13:53"
    ],
)
def test_data_invalida_e_recusada(campo, valor):
    erros = validar_payload(_payload(**{campo: valor}))

    assert _campos(erros) == {campo}
    assert erros[0].linha == 0


@pytest.mark.parametrize(
    "valor",
    ["2026-07-18", dt.date(2026, 7, 18), pd.Timestamp("2026-07-18")],
    ids=["texto-iso", "objeto-date", "objeto-timestamp"],
)
def test_data_explicita_e_aceita(valor):
    """Texto ISO ou objeto de data (o que um seletor de data devolve)."""
    assert validar_payload(_payload(data_ocorrencia=valor, data_abertura=valor)) == []


def test_recusa_de_data_ensina_o_formato():
    erro = validar_payload(_payload(data_abertura="18/07/2026"))[0]

    assert erro.valor_recebido == "18/07/2026"
    assert "AAAA-MM-DD" in erro.motivo


@pytest.mark.parametrize(
    "valor",
    [
        "99:99",  # hora impossível: `preparar` zeraria e imputaria em silêncio
        "9:30",  # sem o zero à esquerda o slice de 2 caracteres devolveria nulo
        "08h30",
        "meio-dia",
        "",
    ],
)
def test_hora_invalida_e_recusada(valor):
    erros = validar_payload(_payload(hora_ocorrencia=valor))

    assert _campos(erros) == {"hora_ocorrencia"}
    assert erros[0].linha == 0


@pytest.mark.parametrize("valor", ["00:00", "13:53", "23:59"])
def test_hora_valida_e_aceita(valor):
    assert validar_payload(_payload(hora_ocorrencia=valor)) == []


def test_hora_como_objeto_de_hora_e_aceita():
    """Um seletor de hora devolve `datetime.time`, não texto."""
    assert validar_payload(_payload(hora_ocorrencia=dt.time(13, 53))) == []


def test_numerico_vazio_e_recusado_com_o_motivo_da_fronteira():
    """A recusa diz que o contrato não imputa — e não que o dado 'é inválido'."""
    erros = validar_payload(_payload(valor_reclamado=""))

    assert len(erros) == 1
    assert "não imputa na fronteira" in erros[0].motivo


# ---------------------------------------------------------------------------
# Lote: o erro aponta a linha
# ---------------------------------------------------------------------------
def test_lote_com_uma_linha_ruim_aponta_a_linha():
    """Sem o número da linha, um lote de 12.000 sinistros não é acionável."""
    lote = pd.DataFrame(
        [
            dict(SINISTRO_VALIDO),
            {**SINISTRO_VALIDO, "uf": "SP"},
            {**SINISTRO_VALIDO, "valor_reclamado": "21.700,00"},
        ]
    )
    erros = validar_payload(lote)

    assert len(erros) == 1
    assert erros[0].linha == 2
    assert erros[0].valor_recebido == "21.700,00"
    assert "[linha 2]" in str(erros[0])


# ---------------------------------------------------------------------------
# Camada 3 — o monitor de imputação
# ---------------------------------------------------------------------------
def test_monitor_nao_encontra_imputacao_no_mes_real(base_crua, campeao):
    """No mês de teste real, nenhuma das 18 features precisa de imputação."""
    _, teste = dividir_temporal(base_crua)
    resumo = resumo_imputacoes(campeao, teste[list(CAMPOS_ABERTURA)])

    assert list(resumo.columns) == list(COLUNAS_RESUMO)
    assert len(resumo) == 18
    assert resumo["n_linhas"].eq(len(teste)).all()
    assert int(resumo["n_imputados"].sum()) == 0
    assert alertas_imputacao(resumo) == []


def test_monitor_denuncia_o_campo_e_as_features_derivadas_dele(base_crua, campeao):
    """Um branco em `valor_reclamado` contamina as três features que dependem dele."""
    lote = base_crua[list(CAMPOS_ABERTURA)].head(100).copy()
    lote.loc[:, "valor_reclamado"] = ""

    resumo = resumo_imputacoes(campeao, lote)
    suspeitas = set(resumo.loc[resumo["suspeita"], "feature"])

    assert suspeitas == {"valor_reclamado", "razao_reclamado_fipe", "faixa_valor_reclamado"}
    assert resumo.set_index("feature").loc["valor_reclamado", "fracao_imputada"] == 1.0

    alertas = alertas_imputacao(resumo)
    assert len(alertas) == 3
    assert any("valor_reclamado" in alerta for alerta in alertas)
    assert all("não de fraude" in alerta for alerta in alertas)


def test_monitor_respeita_limiar_informado(base_crua, campeao):
    """Com o limiar acima da fração medida, nada é sinalizado."""
    lote = base_crua[list(CAMPOS_ABERTURA)].head(100).copy()
    lote.loc[lote.index[:2], "valor_reclamado"] = ""  # 2% das linhas

    resumo = resumo_imputacoes(campeao, lote, limiar=LIMIAR_IMPUTACAO_SUSPEITA)

    assert resumo.set_index("feature").loc["valor_reclamado", "fracao_imputada"] == pytest.approx(
        0.02
    )
    assert alertas_imputacao(resumo) == []

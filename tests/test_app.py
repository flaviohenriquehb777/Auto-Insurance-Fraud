"""Testes do app de triagem — nível de função, sem subir a interface.

O que esta suíte protege, na ordem em que importa para o negócio:

1. **A porta.** Um payload inválido nunca chega ao modelo (`fail closed`, decisão D1);
   o teste prova isso com um estimador que **explode se for chamado**.
2. **A fila.** A ordenação respeita o escore (e é estável no empate), e o corte respeita
   a capacidade `k` — que é restrição de negócio, não escolha do analista.
3. **A linguagem.** O semáforo é de prioridade e nenhum rótulo acusa ninguém.
4. **O número.** O painel de valor bate com a Fase 5 e o artefato implantado reproduz o
   mês auditado (é a prova de que o app lê o `.joblib` certo).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quilha.app import (
    ALTA_PRIORIDADE,
    GANHO_IC_INFERIOR,
    GANHO_IC_SUPERIOR,
    GANHO_POLITICA,
    MEDIA_PRIORIDADE,
    MENSAGEM_PERMANENTE,
    RESULTADO_POLITICA,
    SEM_PRIORIDADE,
    PayloadRecusado,
    avaliar_fila,
    carregar_modelo,
    carregar_treino_e_teste,
    drift_das_features,
    faixa_prioridade,
    hash_do_artefato,
    k_do_lote,
    medir_valor_da_politica,
    montar_fila,
    painel_valor,
    pontuar_lote,
    posicao_na_fila,
    psi,
    resumo_fila,
    saude_do_modelo,
    valor_esperado,
)
from quilha.contrato import CAMPOS_ABERTURA
from quilha.pipeline import CAMINHO_PIPELINE
from quilha.preparacao import preparar

# Sinistro completo e válido: qualquer campo trocado por valor ruim deve ser recusado.
SINISTRO_VALIDO = {
    "data_ocorrencia": "2026-07-01",
    "hora_ocorrencia": "13:53",
    "data_abertura": "2026-07-02",
    "canal_abertura": "App",
    "tipo_sinistro": "Colisão",
    "uf": "SP",
    "idade_segurado": "40",
    "dias_de_apolice": "120",
    "sinistros_12m": "0",
    "idade_veiculo": "3",
    "valor_fipe": "60000",
    "valor_reclamado": "8837,07",
    "boletim_ocorrencia": "Sim",
    "terceiro_identificado": "Não",
}


class _EstimadorFixo:
    """Estimador de mentira com escore fixo por posição da linha — ordena como quisermos."""

    def __init__(self, escores: list[float]) -> None:
        self._escores = np.asarray(escores, dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        escore = self._escores[: len(X)]
        return np.column_stack([1.0 - escore, escore])


class _EstimadorProibido:
    """Explode se for chamado: prova que o dado inválido não chega ao modelo."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise AssertionError("o modelo foi chamado com um payload que o contrato recusaria")


@pytest.fixture
def lote() -> pd.DataFrame:
    """Lote mínimo válido, com a coluna de gabarito como no mês de teste."""
    return pd.DataFrame([SINISTRO_VALIDO] * 5)


# ---------------------------------------------------------------------------
# 1. A porta: contrato antes do modelo
# ---------------------------------------------------------------------------
def test_payload_invalido_e_recusado_antes_de_chegar_ao_modelo(lote: pd.DataFrame) -> None:
    lote.loc[2, "valor_reclamado"] = "21.700,00"  # milhar pt-BR: não converte

    with pytest.raises(PayloadRecusado) as recusa:
        pontuar_lote(lote, _EstimadorProibido())

    tabela = recusa.value.como_tabela()
    assert list(tabela["campo"]) == ["valor_reclamado"]
    assert tabela.loc[0, "linha"] == 2
    assert tabela.loc[0, "valor_recebido"] == "21.700,00"


def test_lote_vazio_de_erros_pontua_normalmente() -> None:
    fila = pontuar_lote(pd.DataFrame([SINISTRO_VALIDO]), _EstimadorFixo([0.9]), k=1)

    assert fila["escore"].tolist() == [0.9]


# ---------------------------------------------------------------------------
# 2. A fila: escore, empate estável e corte em k
# ---------------------------------------------------------------------------
def test_fila_ordena_por_escore_decrescente_e_preserva_a_linha_do_lote() -> None:
    fila = montar_fila(
        pd.DataFrame([SINISTRO_VALIDO] * 3),
        _EstimadorFixo([0.1, 0.9, 0.5]),
        k=3,
    )

    assert fila["linha_no_lote"].tolist() == [1, 2, 0]  # 0,9 · 0,5 · 0,1
    assert fila["escore"].tolist() == [0.9, 0.5, 0.1]
    assert fila["posicao"].tolist() == [1, 2, 3]


def test_empate_preserva_a_ordem_original_do_lote() -> None:
    fila = montar_fila(pd.DataFrame([SINISTRO_VALIDO] * 3), _EstimadorFixo([0.5, 0.5, 0.5]), k=2)

    assert fila["linha_no_lote"].tolist() == [0, 1]


def test_k_do_lote_e_proporcional_a_capacidade_mensal() -> None:
    assert k_do_lote(12_024) == 900  # um mês cheio
    assert k_do_lote(3_006) == 225  # uma semana de um mês cheio
    assert k_do_lote(0) == 0


def test_corte_em_k_respeita_a_capacidade_sem_k_informado() -> None:
    lote = pd.DataFrame([SINISTRO_VALIDO] * 3_006)

    fila = montar_fila(lote, _EstimadorFixo(list(np.linspace(0, 1, 3_006))))

    assert len(fila) == 225
    assert fila.attrs["k"] == 225
    assert fila.attrs["n_lote"] == 3_006


def test_k_maior_que_o_lote_devolve_o_lote_inteiro() -> None:
    fila = montar_fila(pd.DataFrame([SINISTRO_VALIDO] * 3), _EstimadorFixo([0.1, 0.2, 0.3]), k=99)

    assert len(fila) == 3


def test_lote_vazio_nao_quebra() -> None:
    fila = montar_fila(pd.DataFrame(columns=list(CAMPOS_ABERTURA)), _EstimadorFixo([]))

    assert fila.empty
    assert resumo_fila(fila) == "Lote vazio: nada a priorizar."


def test_posicao_na_fila_conta_quantos_escores_estao_a_frente() -> None:
    escores = np.array([0.9, 0.5, 0.5, 0.1])

    assert posicao_na_fila(0.95, escores) == 1
    assert posicao_na_fila(0.5, escores) == 2  # empate herda a primeira posição do grupo
    assert posicao_na_fila(0.05, escores) == 5


def test_avaliar_fila_conta_as_fraudes_dentro_do_corte() -> None:
    lote = pd.DataFrame([SINISTRO_VALIDO] * 4)
    lote["fraude"] = ["0", "1", "0", "1"]
    fila = montar_fila(lote, _EstimadorFixo([0.9, 0.8, 0.1, 0.05]), k=2)

    acerto = avaliar_fila(fila, lote["fraude"])

    assert acerto == {"periciados": 2, "fraudes": 1, "precisao": 0.5}
    assert "1 sinistros eram fraude" in resumo_fila(fila, acerto)


# ---------------------------------------------------------------------------
# 3. A linguagem: prioridade, nunca culpa
# ---------------------------------------------------------------------------
def test_faixa_de_prioridade_muda_nas_bordas_da_capacidade() -> None:
    assert faixa_prioridade(1, k=900) == ALTA_PRIORIDADE
    assert faixa_prioridade(900, k=900) == ALTA_PRIORIDADE
    assert faixa_prioridade(901, k=900) == MEDIA_PRIORIDADE
    assert faixa_prioridade(1_800, k=900) == MEDIA_PRIORIDADE
    assert faixa_prioridade(1_801, k=900) == SEM_PRIORIDADE


def test_rotulos_e_mensagem_permanente_nao_acusam_ninguem() -> None:
    textos = [ALTA_PRIORIDADE, MEDIA_PRIORIDADE, SEM_PRIORIDADE, MENSAGEM_PERMANENTE]

    for texto in textos:
        assert "fraude" not in texto.lower()
        assert "culpado" not in texto.lower()
    assert MENSAGEM_PERMANENTE == "O modelo ordena, a política decide, o perito confirma."
    assert "prioridade" in ALTA_PRIORIDADE.lower()


# ---------------------------------------------------------------------------
# 4. O valor: painel, valor por sinistro e drift
# ---------------------------------------------------------------------------
def test_painel_de_valor_bate_com_a_fase_5() -> None:
    painel = painel_valor()

    assert painel["politica"] == RESULTADO_POLITICA == 3_731_700.00
    assert painel["regra_atual"] == -326_200.00
    assert painel["ganho"] == GANHO_POLITICA == 4_057_900.00
    assert (painel["ganho_ic_inferior"], painel["ganho_ic_superior"]) == (
        GANHO_IC_INFERIOR,
        GANHO_IC_SUPERIOR,
    )
    assert round(painel["piso"] * 100, 4) == 3.2258


def test_valor_esperado_usa_a_conta_do_anexo_e_a_taxa_observada_da_faixa() -> None:
    valores = valor_esperado(0.0)  # escore mínimo: a faixa observada também é ~0

    assert valores["valor_bruto"] == pytest.approx(-700.0, abs=1e-6)
    assert valores["taxa_observada"] == 0.0
    assert valores["valor_com_taxa_observada"] == pytest.approx(-700.0, abs=1e-6)

    # No topo da curva (escore alto), a taxa observada passa a valer muito mais que 700.
    topo = valor_esperado(0.18)
    assert topo["taxa_observada"] > 0.15
    assert topo["valor_com_taxa_observada"] > 2_000.0


def test_psi_zera_na_mesma_distribuicao_e_cresce_com_o_deslocamento() -> None:
    rng = np.random.default_rng(42)
    referencia = pd.Series(rng.normal(size=2_000))

    assert psi(referencia, referencia.copy()) == pytest.approx(0.0, abs=1e-9)
    assert psi(referencia, referencia + 0.10) < 0.25  # deslocamento leve: ainda estável
    assert psi(referencia, referencia + 5.0) > 0.25  # deslocamento material: re-treinar


def test_drift_do_mes_contra_o_treino_fica_dentro_do_limiar() -> None:
    treino, teste = carregar_treino_e_teste()

    tabela = drift_das_features(
        treino.sample(2_000, random_state=42), teste.sample(2_000, random_state=42)
    )

    assert (tabela["psi"] < 0.25).all()
    assert set(tabela["veredito"]) <= {"estável", "atenção", "re-treinar (deriva de população)"}


# ---------------------------------------------------------------------------
# 5. O artefato: hash auditado, saúde e reprodução do mês
# ---------------------------------------------------------------------------
def test_hash_do_artefato_e_o_hash_auditado() -> None:
    assert (
        hash_do_artefato(CAMINHO_PIPELINE)
        == "65c7eb5ea1325f4911f9223fa3bf1d2f7aec909e77a30865a47f25a837bb10a3"
    )


def test_saude_do_modelo_mede_vazios_e_imputacoes_do_lote() -> None:
    treino, _ = carregar_treino_e_teste()
    lote = treino.sample(50, random_state=42).reset_index(drop=True)

    saude = saude_do_modelo(carregar_modelo(), lote[list(CAMPOS_ABERTURA)])

    assert saude["n_linhas"] == 50
    assert set(saude["vazios_por_campo"]) == set(CAMPOS_ABERTURA)
    assert saude["alertas_imputacao"] == []  # lote limpo: nada a alarmar


def test_artefato_implantado_reproduz_o_mes_auditado() -> None:
    """A prova de que o app lê o `.joblib` certo: 201 fraudes e R$ 3.731.700,00."""
    _, teste = carregar_treino_e_teste()

    conferencia = medir_valor_da_politica(carregar_modelo(), teste)

    assert conferencia["confere"] is True
    assert conferencia["tp"] == 201
    assert conferencia["periciados"] == 900
    assert conferencia["resultado"] == RESULTADO_POLITICA


def test_preparar_do_app_nao_toca_no_numero_do_endpoint() -> None:
    """O app não reimplementa conversão: quem resolve a vírgula é o `preparo` do Pipeline."""
    features = preparar(pd.DataFrame([SINISTRO_VALIDO]))

    assert float(features.loc[0, "valor_reclamado"]) == 8837.07

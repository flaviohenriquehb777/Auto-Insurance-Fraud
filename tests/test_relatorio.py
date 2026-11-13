"""Testes do resumo executivo em `.docx` — Fase 6, Parte 4.

O ponto sensível desta parte é a **regra de ouro**: todo número do documento
tem de existir impresso no notebook consolidado. Como os dois bebem do **mesmo**
CSV (`reports/tabelas/avaliacao_quadro_respostas.csv`), o teste trava a ligação:
se alguém mudar a fonte, o documento muda junto — e nunca em separado.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from docx import Document

from quilha.relatorio import (
    ARQUIVO_QUADRO,
    COLUNAS_QUADRO,
    LEITURAS,
    TEMAS,
    carregar_quadro,
    gerar_relatorio,
    montar_documento,
    partes_do_resumo,
)

# ---------------------------------------------------------------------------
# Fonte única: o CSV medido na Fase 5
# ---------------------------------------------------------------------------


def test_carregar_quadro_le_o_csv_medido() -> None:
    """O CSV da Fase 5 tem as três colunas e os 24 campos do Quadro."""
    quadro = carregar_quadro()

    assert list(quadro.columns) == list(COLUNAS_QUADRO)
    assert len(quadro) == 24
    assert quadro["pergunta"].between(1, 5).all()
    assert (quadro["resposta"] != "não informado").all()


def test_carregar_quadro_falha_com_mensagem_clara(tmp_path: Path) -> None:
    """Arquivo ausente não vira documento vazio: falha alto e claro."""
    with pytest.raises(FileNotFoundError, match=ARQUIVO_QUADRO):
        carregar_quadro(tmp_path / ARQUIVO_QUADRO)


def test_carregar_quadro_exige_as_colunas(tmp_path: Path) -> None:
    """CSV sem as colunas certas é recusado, não interpretado no escuro."""
    caminho = tmp_path / ARQUIVO_QUADRO
    pd.DataFrame({"pergunta": [1], "campo": ["x"]}).to_csv(caminho, index=False)

    with pytest.raises(ValueError, match="resposta"):
        carregar_quadro(caminho)


# ---------------------------------------------------------------------------
# As cinco partes
# ---------------------------------------------------------------------------


def test_partes_do_resumo_cobre_as_cinco_perguntas() -> None:
    """As cinco perguntas do comitê aparecem, em ordem, sem perder linha."""
    partes = partes_do_resumo(carregar_quadro())

    assert [numero for numero, _, _ in partes] == [1, 2, 3, 4, 5]
    assert sum(len(pares) for _, _, pares in partes) == 24
    for numero, titulo, _ in partes:
        assert titulo == TEMAS[numero]
    assert set(LEITURAS) == {1, 2, 3, 4, 5}


# ---------------------------------------------------------------------------
# O documento
# ---------------------------------------------------------------------------


def test_documento_tem_cinco_partes_e_o_quadro() -> None:
    """5 cabeçalhos de pergunta + 1 tabela com cabeçalho e os 24 campos."""
    quadro = carregar_quadro()
    documento = montar_documento(quadro, gerado_em=datetime(2026, 8, 1, 12, 0))

    cabecalhos = [p.text for p in documento.paragraphs if p.style.name == "Heading 1"]
    partes = [texto for texto in cabecalhos if texto.startswith("Pergunta ")]
    assert len(partes) == 5

    assert len(documento.tables) == 1
    tabela = documento.tables[0]
    assert len(tabela.rows) == 25  # cabeçalho + 24 campos
    assert [c.text for c in tabela.rows[0].cells] == ["Pergunta", "Campo", "Resposta"]

    # Cada resposta do CSV aparece no documento (documento e CSV não divergem).
    texto = "\n".join(c.text for row in tabela.rows for c in row.cells)
    for resposta in quadro["resposta"]:
        assert resposta in texto


def test_gerar_relatorio_grava_o_arquivo(tmp_path: Path) -> None:
    """O gerador devolve o caminho e grava um `.docx` legível."""
    destino = tmp_path / "Relatorio_Quilha_Fraude.docx"

    gravado = gerar_relatorio(caminho_saida=destino)

    assert gravado == destino
    assert destino.exists()
    assert len(Document(destino).tables[0].rows) == 25

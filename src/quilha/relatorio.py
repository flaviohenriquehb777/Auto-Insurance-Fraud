"""Resumo executivo da Fase 6 em `.docx` (5 partes + Quadro de Respostas).

O documento é montado a partir de uma **fonte única**: o CSV
`reports/tabelas/avaliacao_quadro_respostas.csv`, o mesmo que o notebook
consolidado (`06_consolidado.ipynb`) imprime ao final. Pela regra de ouro da
Fase 6, **todo número do `.docx` existe impresso no `.ipynb`** — como os dois
bebem do mesmo arquivo, a divergência é impossível por construção.

As **cinco partes** do resumo correspondem às cinco perguntas do comitê e são
montadas agrupando as linhas do CSV por pergunta. Cada parte abre com uma
frase de leitura (sem números: só interpretação) e lista, em marcadores, o
campo do Quadro e a resposta medida. O **Quadro completo (24 campos)** fecha o
documento, fora do limite de 2 páginas.

Fora do Quadro, o único dado que não vem do CSV é o carimbo de geração
("gerado em") — metadado do documento, não métrica do case.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from docx import Document
from docx.document import Document as Documento

from quilha.dados import RAIZ_PROJETO
from quilha.modelagem import PASTA_TABELAS

# ---------------------------------------------------------------------------
# Fonte única e destino do entregável
# ---------------------------------------------------------------------------

ARQUIVO_QUADRO = "avaliacao_quadro_respostas.csv"
CAMINHO_QUADRO: Path = PASTA_TABELAS / ARQUIVO_QUADRO
NOME_ARQUIVO_SAIDA = "Relatorio_Quilha_Fraude.docx"
CAMINHO_SAIDA: Path = RAIZ_PROJETO / NOME_ARQUIVO_SAIDA

COLUNAS_QUADRO = ("pergunta", "campo", "resposta")

TITULO = "Detecção de fraude em sinistros de automóvel — Quilha Seguros"
SUBTITULO = "Resumo executivo da política de priorização da perícia (CRISP-DM, Fases 1 a 6)"

# Título de cada uma das cinco partes (perguntas do comitê).
TEMAS: dict[int, str] = {
    1: "Quais variáveis entram no modelo (e idade/UF ficam?)",
    2: "O que a base revela sobre o risco",
    3: "Qual modelo, e como ele performa",
    4: "Qual política adotar, e quanto ela rende",
    5: "Quantas perícias manter, e até onde vale a pena",
}

# Frase de leitura de cada parte: interpretação, **sem números** — os números
# ficam nos campos do Quadro logo abaixo, sempre vindos do CSV.
LEITURAS: dict[int, str] = {
    1: (
        "O modelo usa apenas o que a seguradora já sabe na abertura do sinistro. "
        "Idade e UF permanecem por empate técnico, não por ganho econômico."
    ),
    2: (
        "A base não é homogênea: há grupos com taxa de fraude muito acima da média, "
        "e é essa concentração que a fila de perícia explora."
    ),
    3: (
        "O campeão é escolhido pelo resultado econômico, não pela métrica estatística "
        "isolada; a referência simples serve de piso de comparação."
    ),
    4: (
        "A política recomendada troca um critério fixo de valor por uma ordenação por "
        "escore, mantendo a mesma capacidade mensal de perícia."
    ),
    5: (
        "O dimensionamento da equipe vem do custo marginal da perícia: existe um teto "
        "econômico para quantas periciar antes de a conta virar."
    ),
}


def carregar_quadro(caminho: Path = CAMINHO_QUADRO) -> pd.DataFrame:
    """Lê o Quadro de Respostas medido na Fase 5 — a fonte única do documento."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Quadro não encontrado em {caminho}. Rode a Fase 5 para gerar o CSV."
        )
    quadro = pd.read_csv(caminho)
    faltando = set(COLUNAS_QUADRO).difference(quadro.columns)
    if faltando:
        raise ValueError(f"CSV do Quadro sem as colunas: {sorted(faltando)}")
    quadro = quadro[list(COLUNAS_QUADRO)].copy()
    quadro["pergunta"] = quadro["pergunta"].astype(int)
    return quadro


def partes_do_resumo(
    quadro: pd.DataFrame,
) -> list[tuple[int, str, list[tuple[str, str]]]]:
    """Agrupa as linhas do Quadro nas cinco perguntas, na ordem 1 a 5."""
    partes: list[tuple[int, str, list[tuple[str, str]]]] = []
    for pergunta in sorted(quadro["pergunta"].unique()):
        recorte = quadro.loc[quadro["pergunta"] == pergunta, ["campo", "resposta"]]
        pares = [(str(campo), str(resposta)) for campo, resposta in recorte.to_numpy()]
        numero = int(pergunta)
        partes.append((numero, TEMAS.get(numero, f"Pergunta {numero}"), pares))
    return partes


def _adicionar_cabecalho(documento: Documento, gerado_em: datetime) -> None:
    """Título, subtítulo e carimbo de geração (metadado do documento)."""
    documento.add_heading(TITULO, level=0)
    documento.add_paragraph(SUBTITULO)
    carimbo = documento.add_paragraph()
    texto = carimbo.add_run(
        f"Gerado em {gerado_em:%d/%m/%Y %H:%M} · "
        f"fonte única: reports/tabelas/{ARQUIVO_QUADRO}"
    )
    texto.italic = True


def _adicionar_parte(
    documento: Documento,
    pergunta: int,
    titulo: str,
    pares: list[tuple[str, str]],
) -> None:
    """Uma das cinco partes: cabeçalho, frase de leitura e campos em marcadores."""
    documento.add_heading(f"Pergunta {pergunta} — {titulo}", level=1)
    leitura = LEITURAS.get(pergunta)
    if leitura:
        documento.add_paragraph(leitura)
    for campo, resposta in pares:
        paragrafo = documento.add_paragraph(style="List Bullet")
        rotulo = paragrafo.add_run(f"{campo}: ")
        rotulo.bold = True
        paragrafo.add_run(resposta)


def _adicionar_quadro(documento: Documento, quadro: pd.DataFrame) -> None:
    """O Quadro de Respostas completo (24 campos), fora do limite de 2 páginas."""
    documento.add_page_break()
    documento.add_heading("Quadro de Respostas do comitê", level=1)
    documento.add_paragraph(
        "Respostas medidas na Fase 5. Cada número desta tabela existe impresso no "
        "notebook consolidado (06_consolidado.ipynb)."
    )
    tabela = documento.add_table(rows=1, cols=len(COLUNAS_QUADRO))
    tabela.style = "Table Grid"
    for celula, texto in zip(
        tabela.rows[0].cells, ("Pergunta", "Campo", "Resposta"), strict=True
    ):
        celula.text = texto
        for paragrafo in celula.paragraphs:
            for run in paragrafo.runs:
                run.bold = True
    for numero, campo, resposta in quadro.to_numpy():
        celulas = tabela.add_row().cells
        celulas[0].text = str(numero)
        celulas[1].text = str(campo)
        celulas[2].text = str(resposta)


def montar_documento(
    quadro: pd.DataFrame, gerado_em: datetime | None = None
) -> Documento:
    """Monta o documento em memória (testável sem tocar o disco)."""
    documento = Document()
    _adicionar_cabecalho(documento, gerado_em or datetime.now())
    for pergunta, titulo, pares in partes_do_resumo(quadro):
        _adicionar_parte(documento, pergunta, titulo, pares)
    _adicionar_quadro(documento, quadro)
    return documento


def gerar_relatorio(
    caminho_saida: Path | None = None,
    caminho_quadro: Path | None = None,
    gerado_em: datetime | None = None,
) -> Path:
    """Gera o `.docx` a partir do Quadro medido e devolve o caminho gravado."""
    quadro = carregar_quadro(caminho_quadro or CAMINHO_QUADRO)
    documento = montar_documento(quadro, gerado_em=gerado_em)
    destino = Path(caminho_saida) if caminho_saida is not None else CAMINHO_SAIDA
    destino.parent.mkdir(parents=True, exist_ok=True)
    documento.save(destino)
    return destino


if __name__ == "__main__":
    caminho = gerar_relatorio()
    print(f"Relatório gerado: {caminho.relative_to(RAIZ_PROJETO)}")

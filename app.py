"""Lançador do app de triagem — `streamlit run app.py` a partir da raiz do projeto.

Só existe para uma coisa: colocar `src/` no caminho de importação. O pacote `quilha` não é
instalado no ambiente virtual (o projeto roda com `PYTHONPATH=src`, que o `pytest` já injeta
pelo `pyproject.toml`), então sem estas linhas o `import quilha` falharia ao subir a tela.

Toda a lógica e as telas moram em `src/quilha/app.py`; aqui não há regra de negócio nenhuma.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `Path(__file__).parent` é a raiz do projeto (este arquivo mora nela).
RAIZ = Path(__file__).resolve().parent
SRC = RAIZ / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quilha.app import main  # noqa: E402 - precisa do sys.path acima

main()

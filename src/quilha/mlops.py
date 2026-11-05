"""Infraestrutura de MLOps da Fase 4 (CRISP-DM) — tracking com degradação limpa.

Regra de ouro deste módulo: **o treino nunca pode quebrar por causa do tracking**.
Se as credenciais do DagsHub não estiverem no ambiente (ou se o `mlflow` nem estiver
instalado), o módulo emite um aviso, cai para o armazenamento local em `mlruns/` e o
notebook segue até o fim. É o pré-requisito do notebook consolidado, que a equipe vai
executar sem token nenhum.

O que mora aqui
---------------
* `experimento()`  — nome único do experimento da fase (`quilha-fraude-fase4`).
* `nome_run(...)`  — convenção `<familia>__<config>__seed<seed>`.
* `contexto_run(...)` — *context manager* que abre (ou finge abrir) uma run e devolve um
  `Registro` com métodos de logging. Com o tracking desligado, é tudo no-op.
* `reprodutibilidade()` — hash do `uv.lock` e commit curto do git.

Segurança e reprodutibilidade
-----------------------------
As credenciais são lidas **apenas** de variáveis de ambiente (`MLFLOW_TRACKING_URI`,
`MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`) — nunca em código, nunca no
repositório. O Pipeline logado é sempre o objeto **inteiro** (`preparo + imputacao +
modelo`), que é o que o app carrega; a assinatura é inferida sobre o chamado CRU.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quilha.dados import RAIZ_PROJETO

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
EXPERIMENTO = "quilha-fraude-fase4"
ARQUIVO_TRACKING_LOCAL = RAIZ_PROJETO / "mlflow.db"
PASTA_ARTEFATOS_LOCAL = RAIZ_PROJETO / "mlartifacts"
VARIAVEIS_AMBIENTE = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
)

_AVISO_EMITIDO = False


# ---------------------------------------------------------------------------
# Ambiente e nomes
# ---------------------------------------------------------------------------
def _mlflow():
    """Importa o `mlflow` só quando é usado; devolve `None` se não estiver instalado."""
    try:
        import mlflow
    except ImportError:
        return None
    return mlflow


def credenciais_presentes() -> bool:
    """`True` quando as três variáveis de ambiente do MLflow estão preenchidas."""
    return all(os.environ.get(nome) for nome in VARIAVEIS_AMBIENTE)


def experimento() -> str:
    """Nome do experimento da Fase 4 no MLflow (todos os runs ficam sob ele)."""
    return EXPERIMENTO


def nome_run(familia: str, config: str, seed: int) -> str:
    """Convenção única de nome de run: `<familia>__<config>__seed<seed>`."""
    return f"{familia}__{config}__seed{seed}"


def _avisar(mensagem: str) -> None:
    """Emite o aviso de degradação uma única vez por processo."""
    global _AVISO_EMITIDO
    if not _AVISO_EMITIDO:
        warnings.warn(mensagem, stacklevel=2)
        _AVISO_EMITIDO = True


def configurar_tracking() -> tuple[str, bool]:
    """Aponta o MLflow para o DagsHub quando há credencial; senão, para o SQLite local.

    Devolve `(uri, local)`. Nunca levanta exceção: sem `mlflow` instalado devolve
    `("sem-mlflow", True)` e o treino continua com o logging desligado.

    O fallback local usa **SQLite** (`sqlite:///mlflow.db`) e não o `./mlruns`: o MLflow
    3.x colocou o *file store* em modo de manutenção, e só o backend de banco habilita os
    recursos atuais — inclusive o Model Registry com *alias*, que esta fase exige.
    """
    mlflow = _mlflow()
    if mlflow is None:
        _avisar(
            "Pacote `mlflow` não instalado: o tracking está desligado. "
            "O treino segue normalmente (instale o grupo `mlops` para registrar as runs)."
        )
        return "sem-mlflow", True

    if credenciais_presentes():
        uri = os.environ["MLFLOW_TRACKING_URI"]
        return uri, False

    uri = f"sqlite:///{ARQUIVO_TRACKING_LOCAL.as_posix()}"
    _avisar(
        "Credenciais do MLflow ausentes (MLFLOW_TRACKING_URI/USERNAME/PASSWORD): "
        f"gravando as runs localmente em `{ARQUIVO_TRACKING_LOCAL}`. "
        "O treino segue normalmente."
    )
    return uri, True


def reprodutibilidade() -> dict[str, str]:
    """Hash do `uv.lock` e commit curto do git — sem isso a run não é reproduzível."""
    dados: dict[str, str] = {}
    lock = RAIZ_PROJETO / "uv.lock"
    if lock.exists():
        dados["uv_lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()[:16]
    commit = _commit_git()
    if commit:
        dados["git_commit"] = commit
    return dados


def _commit_git() -> str | None:
    """Commit curto do HEAD, ou `None` se não houver repositório/commit."""
    try:
        saida = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=RAIZ_PROJETO,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return saida.stdout.strip() or None


# ---------------------------------------------------------------------------
# Handle de logging devolvido pelo contexto_run
# ---------------------------------------------------------------------------
def _parametro_logavel(valor: Any) -> str | int | float | bool:
    """MLflow aceita só escalar; o resto vira texto (ex.: `class_weight=None`)."""
    if isinstance(valor, str | int | float | bool):
        return valor
    return str(valor)


@dataclass
class Registro:
    """Handle de logging de uma run. Com `ativa=False`, todo método é no-op.

    Existe para o notebook não precisar de `if` nenhum em volta do logging: com ou sem
    credencial, o código do treino é idêntico.
    """

    ativa: bool
    uri: str
    run_id: str | None = None
    mlflow: Any | None = None
    motivo: str | None = None

    def log_parametros(self, parametros: dict) -> None:
        if self.ativa:
            self.mlflow.log_params({k: _parametro_logavel(v) for k, v in parametros.items()})

    def log_metricas(self, metricas: dict, prefixo: str = "") -> None:
        if self.ativa:
            self.mlflow.log_metrics(
                {f"{prefixo}{k}": float(v) for k, v in metricas.items() if v is not None}
            )

    def log_artefato(self, caminho: Path | str) -> None:
        if self.ativa:
            self.mlflow.log_artifact(str(caminho))

    def log_json(self, objeto: Any, nome: str) -> None:
        """Grava um objeto (tabela, dicionário) como artefato JSON da run."""
        if not self.ativa:
            return
        with tempfile.TemporaryDirectory() as tmp:
            destino = Path(tmp) / nome
            destino.write_text(
                json.dumps(objeto, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            self.mlflow.log_artifact(str(destino))

    def log_pipeline(self, pipeline, exemplo_cru) -> str | None:
        """Loga o Pipeline **inteiro** com assinatura inferida sobre o chamado CRU.

        `exemplo_cru` é um DataFrame de uma linha com as 15 colunas que o app envia
        (as 17 do CSV menos `fraude` e `valor_indenizado`). Devolve a URI do modelo
        logado, ou `None` se o tracking estiver desligado.

        Serialização em `cloudpickle` de propósito: o Pipeline atravessa sklearn,
        XGBoost, LightGBM e uma `FunctionTransformer` com função nossa
        (`quilha.preparacao.preparar`). O `skops` — padrão do MLflow 3 — exige declarar
        cada tipo como confiável e travaria a cada família de modelo nova; o cloudpickle
        é o formato estável para Pipeline heterogêneo e tem o mesmo nível de confiança do
        `pipeline_fraude.joblib` que o projeto já versiona.
        """
        if not self.ativa:
            return None
        from mlflow.models import infer_signature

        assinatura = infer_signature(exemplo_cru, pipeline.predict_proba(exemplo_cru)[:, 1])
        self.mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="pipeline",
            signature=assinatura,
            input_example=exemplo_cru,
            serialization_format="cloudpickle",
        )
        return f"runs:/{self.run_id}/pipeline"

    def registrar_modelo(
        self,
        nome_registrado: str,
        descricao: str | None = None,
        tags: dict | None = None,
        alias: str = "champion",
    ) -> str | None:
        """Registra o Pipeline no Model Registry e aplica o alias (ex.: `champion`).

        Usa **alias**, não *stages* — stages estão descontinuados no MLflow atual.
        Se o registro falhar (ex.: armazenamento local sem backend de registry), emite
        aviso e devolve `None`: o `.joblib` já está logado como artefato e o treino segue.
        """
        if not self.ativa:
            return None
        try:
            versao = self.mlflow.register_model(f"runs:/{self.run_id}/pipeline", nome_registrado)
            cliente = self.mlflow.MlflowClient()
            if descricao:
                cliente.update_model_version(nome_registrado, versao.version, description=descricao)
            if tags:
                for chave, valor in tags.items():
                    cliente.set_model_version_tag(
                        nome_registrado, versao.version, chave, str(valor)
                    )
            if alias:
                cliente.set_registered_model_alias(nome_registrado, alias, versao.version)
        except Exception as erro:  # noqa: BLE001 - registry indisponível não pode derrubar o treino
            _avisar(f"Model Registry indisponível ({erro!r}); o Pipeline segue logado na run.")
            return None
        return versao.version


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------
def _garantir_experimento(mlflow, local: bool) -> None:
    """Cria o experimento da fase se ainda não existir e o seleciona.

    No modo local, fixa o diretório de artefatos na raiz do projeto para o resultado não
    depender do diretório de onde o notebook foi aberto.
    """
    if mlflow.get_experiment_by_name(EXPERIMENTO) is not None:
        mlflow.set_experiment(EXPERIMENTO)
        return

    if local:
        PASTA_ARTEFATOS_LOCAL.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(EXPERIMENTO, artifact_location=PASTA_ARTEFATOS_LOCAL.as_uri())
    else:
        mlflow.create_experiment(EXPERIMENTO)
    mlflow.set_experiment(EXPERIMENTO)


@contextmanager
def contexto_run(
    nome: str, parametros: dict | None = None, tags: dict | None = None
) -> Iterator[Registro]:
    """Abre uma run do MLflow (ou finge abrir) e garante o fechamento no fim.

    Sem credencial — ou sem `mlflow` instalado — devolve um `Registro` inativo: o corpo do
    `with` roda igual e o treino nunca quebra por causa do tracking.
    """
    mlflow = _mlflow()
    if mlflow is None:
        yield Registro(ativa=False, uri="sem-mlflow", motivo="pacote mlflow ausente")
        return

    uri, local = configurar_tracking()
    mlflow.set_tracking_uri(uri)
    _garantir_experimento(mlflow, local)
    with mlflow.start_run(run_name=nome) as run:
        registro = Registro(ativa=True, uri=uri, run_id=run.info.run_id, mlflow=mlflow)
        if tags:
            mlflow.set_tags(tags)
        if parametros:
            registro.log_parametros(parametros)
        registro.log_parametros(reprodutibilidade())
        yield registro

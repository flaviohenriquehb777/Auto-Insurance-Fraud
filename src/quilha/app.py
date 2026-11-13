"""App de triagem de perícia — a interface entre a política e as pessoas (Fase 6).

O app **não decide fraude**. Ele faz três coisas: **prioriza** a fila, **explica** por
que um sinistro subiu na fila e **mostra o valor** da política. A decisão de periciar é
humana; o perito confirma.

Por que este módulo é separado da tela
--------------------------------------
Toda a lógica (contrato, ordenação, corte em `k`, faixa de prioridade, painel de valor,
drift, saúde do modelo) mora aqui como **função pura**, testável sem subir UI
(`tests/test_app.py`). A tela Streamlit fica em `main()`, e o `streamlit` só é importado
lá dentro — assim os testes rodam sem o pacote instalado e sem navegador.

As três telas
-------------
1. **Fila do mês** — recebe o lote, passa pelo contrato (camada 2 da Parte 1), pontua,
   ordena por escore e corta em `k`. Exporta a fila em CSV para o gestor da perícia.
2. **Sinistro individual** — escore, posição na fila, faixa de prioridade e os sinais
   SHAP em linguagem de priorização (nunca "evidências de fraude").
3. **Valor e governança** — o ganho em reais com o intervalo de confiança, o valor por
   sinistro, a saúde do modelo e os gatilhos de circuit breaker.

Decisões de implantação travadas aqui (e levadas ao resumo da diretoria)
-----------------------------------------------------------------------
**D1 — Lote com linha inválida: recusa o lote inteiro (`fail closed`).**
A fila é uma ordenação sobre o **universo fechado do mês**: o corte em `k`, a posição de
cada sinistro e o resumo do mês só valem se *todos* os sinistros do lote foram pontuados.
Processar "as linhas boas" e omitir as ruins em silêncio é exatamente a falha que esta fase
existe para matar: a linha omitida pode ser a fraude e ninguém saberá. O custo de bloquear é
minutos de correção na origem; o custo de uma fila incompleta é dinheiro e risco de auditoria.
O app devolve a **lista de erros por linha** (campo, valor recebido, motivo) e permite baixá-la
em CSV, para a correção ser rápida. Nenhuma predição parcial, nunca.
No **sinistro individual** o efeito é o mesmo, naturalmente restrito àquele sinistro.

**D2 — Semáforo por posição na fila, jamais por culpa.**
🔴 alta prioridade = dentro da capacidade (`posição ≤ k`); 🟡 periciar se houver folga =
até `2 × k` (a capacidade do mês seguinte); 🟢 sem prioridade de perícia = o resto.
É posição em uma fila de trabalho, não veredito sobre a pessoa. O texto na tela nunca diz
"fraude" para um sinistro individual.

**D3 — Valor por sinistro como faixa, com a taxa observada ao lado.**
`valor esperado = chance × 21.700 − 700`, mas a probabilidade do campeão **não é calibrada**
(Brier 0,014702 sem calibração contra 0,014688 com isotonic: ganho de 0,000011, ou seja, nada;
sigmoid piora para 0,015210). Então o número aparece como estimativa, acompanhado da **taxa
observada** naquela faixa de escore — o que transforma "chance de 18%" em "na faixa de escore
equivalente, 17,9% dos sinistros eram fraude".

**D4 — Capacidade proporcional ao lote.**
`k = 900 × n_sinistros / 12.024` (`modelagem.calcular_k_proporcional`): a equipe pericia 900/mês,
e um lote que não é um mês cheio não pode ser periciado como se fosse.

Rastreabilidade
---------------
A fila carrega a coluna `linha_no_lote` (0-based), na **mesma numeração** que o contrato usa
para apontar erros — é o que permite sair da linha de erro para a linha da fila e do arquivo.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from quilha.contrato import CAMPOS_ABERTURA, ErroContrato, validar_payload
from quilha.dados import (
    DATA_CORTE,
    RAIZ_PROJETO,
    carregar_base_crua,
    dividir_temporal,
)
from quilha.explicacao import explicar_sinistro
from quilha.modelagem import (
    CAPACIDADE_MENSAL,
    CUSTO_FRAUDE_PAGA,
    CUSTO_PERICIA,
    SEED,
    SINISTROS_POR_MES,
    calcular_k_proporcional,
)
from quilha.monitoramento import LIMIAR_IMPUTACAO_SUSPEITA, alertas_imputacao, resumo_imputacoes
from quilha.pipeline import CAMINHO_PIPELINE, carregar_pipeline
from quilha.preparacao import FEATURES, preparar

# ---------------------------------------------------------------------------
# Constantes de tela e de política
# ---------------------------------------------------------------------------
# Semáforo reetiquetado por prioridade (nunca por culpa).
ALTA_PRIORIDADE = "🔴 Periciar — alta prioridade"
MEDIA_PRIORIDADE = "🟡 Periciar se houver folga"
SEM_PRIORIDADE = "🟢 Sem prioridade de perícia"

# Mensagem permanente das telas: o app ordena, ninguém é acusado.
MENSAGEM_PERMANENTE = "O modelo ordena, a política decide, o perito confirma."

# A faixa amarela é uma capacidade de folga: o mês seguinte de perícia.
FATOR_FOLGA = 2

# Limiares de drift do plano de monitoramento da Fase 5 (docs/05_avaliacao.md, §9.5).
LIMIAR_PSI_ALERTA = 0.25
LIMIAR_PSI_ATENCAO = 0.10

# Features acompanhadas quanto a deriva: as de maior importância SHAP de cada natureza.
# A lista sai de `reports/tabelas/avaliacao_shap_importancia.csv`.
FEATURES_DRIFT_NUMERICAS: tuple[str, ...] = (
    "razao_reclamado_fipe",
    "dias_de_apolice",
    "atraso_abertura_dias",
    "valor_reclamado",
    "sinistros_12m",
)
FEATURES_DRIFT_CATEGORICAS: tuple[str, ...] = (
    "boletim_ocorrencia",
    "tipo_sinistro",
    "canal_abertura",
)
FEATURES_DRIFT: tuple[str, ...] = FEATURES_DRIFT_NUMERICAS + FEATURES_DRIFT_CATEGORICAS

# O gabarito só existe no lote do mês de teste; quando ele vem, a fila se autoavalia.
COLUNA_GABARITO = "fraude"

# Colunas do chamado que acompanham a fila para o perito abrir o caso com contexto.
COLUNAS_CONTEXTO_FILA: tuple[str, ...] = (
    "tipo_sinistro",
    "uf",
    "dias_de_apolice",
    "boletim_ocorrencia",
)

# ---------------------------------------------------------------------------
# Números auditados da política (Fase 5) — fonte única, com procedência declarada
# ---------------------------------------------------------------------------
# Ponto (resultado realizado no mês de teste de 11.904 sinistros, agosto/2026):
#   `docs/06_implantacao.md` (reauditoria da Parte 0) e
#   `reports/tabelas/avaliacao_quadro_respostas.csv` (campos da Pergunta 4).
# Intervalo de confiança do ganho: `reports/tabelas/avaliacao_bootstrap.csv`
#   (linha `diferenca`, 2.000 reamostragens, SEED=42).
RESULTADO_POLITICA = 3_731_700.00
RESULTADO_REGRA_ATUAL = -326_200.00
GANHO_POLITICA = RESULTADO_POLITICA - RESULTADO_REGRA_ATUAL  # R$ 4.057.900,00
GANHO_IC_INFERIOR = 3_537_100.00
GANHO_IC_SUPERIOR = 4_665_500.00
TP_POLITICA_MES_TESTE = 201
PERICIAS_POLITICA_MES_TESTE = CAPACIDADE_MENSAL
N_SINISTROS_MES_TESTE = 11_904

# Piso de viabilidade: a partir daqui a perícia se paga em valor esperado.
PISO_VIABILIDADE = CUSTO_PERICIA / CUSTO_FRAUDE_PAGA  # 3,2258%


def formatar_reais(valor: float, casas: int = 0) -> str:
    """Formata um número no padrão monetário brasileiro, sem o prefixo R$."""
    formatado = f"{valor:,.{casas}f}"
    return formatado.replace(",", "#").replace(".", ",").replace("#", ".")


# Gatilhos do plano de monitoramento (docs/05_avaliacao.md, §9.5) expostos na tela 3.
GATILHOS_MONITORAMENTO: tuple[tuple[str, str, str], ...] = (
    ("Precisão@900", "< 3,2258% por 2 meses seguidos", "Recalibrar; se persistir, re-treinar"),
    ("Resultado@900 realizado", "< R$ 0", "Circuit breaker: fila volta à regra de R$ 50 mil"),
    ("Taxa de fraude do fluxo", "< 0,3480% (equilíbrio)", "Revisar a política com o comitê"),
    (
        "% de boletim em branco",
        "> 10% (baseline 4,83%)",
        "Alerta; acima de 20% aciona a contingência",
    ),
    ("PSI das features principais", "> 0,25", "Re-treinar (deriva de população)"),
    (
        "Distribuição do escore",
        "share no top-900 desviando > 20%",
        "Investigar a deriva antes de periciar",
    ),
)

# Curva de calibração da Fase 5: dá a taxa observada por faixa de escore.
CAMINHO_CURVA_CALIBRACAO = RAIZ_PROJETO / "reports" / "tabelas" / "parte5_curva_calibracao.csv"


# ---------------------------------------------------------------------------
# Contrato na porta: um lote inválido não chega ao modelo
# ---------------------------------------------------------------------------
class PayloadRecusado(Exception):
    """Lote (ou sinistro) barrado pelo contrato da camada 2, com os erros por linha.

    `fail closed` (decisão D1): o app prefere não pontuar a pontuar sobre dado que ele
    não entende. Quem recebe esta exceção sabe exatamente **qual linha**, **qual campo**
    e **qual valor** corrigir.
    """

    def __init__(self, erros: list[ErroContrato]) -> None:
        self.erros = erros
        super().__init__(f"{len(erros)} problema(s) de contrato no lote")

    def como_tabela(self) -> pd.DataFrame:
        """Erros em tabela pronta para a tela e para o CSV de correção."""
        return pd.DataFrame(
            {
                "linha": [erro.linha if erro.linha is not None else -1 for erro in self.erros],
                "campo": [erro.campo for erro in self.erros],
                "valor_recebido": [erro.valor_recebido for erro in self.erros],
                "motivo": [erro.motivo for erro in self.erros],
            }
        )


# ---------------------------------------------------------------------------
# Fila: capacidade, ordenação e faixa de prioridade
# ---------------------------------------------------------------------------
def k_do_lote(n_sinistros: int, capacidade_mensal: int = CAPACIDADE_MENSAL) -> int:
    """Capacidade de perícia proporcional ao tamanho do lote (decisão D4).

    A equipe executa `capacidade_mensal` perícias por mês; um lote de `n` sinistros recebe
    a fração correspondente. Sem isso, um lote de três dias consumiria as perícias do mês.
    """
    return calcular_k_proporcional(n_sinistros, capacidade_mensal=capacidade_mensal)


def faixa_prioridade(posicao: int, k: int, fator_folga: int = FATOR_FOLGA) -> str:
    """Rótulo do semáforo pela posição na fila (decisão D2).

    `posicao` é 1-based. Dentro da capacidade = alta prioridade; até `fator_folga × k` =
    folga; o resto = sem prioridade. É posição em fila de trabalho, não veredito.
    """
    if posicao <= k:
        return ALTA_PRIORIDADE
    if posicao <= fator_folga * k:
        return MEDIA_PRIORIDADE
    return SEM_PRIORIDADE


def escore_da_politica(pipeline, df: pd.DataFrame) -> np.ndarray:
    """Escore de cada sinistro = probabilidade da classe positiva do campeão.

    A vírgula decimal de `valor_reclamado` **não** é tratada aqui: quem converte é o
    `preparo` que já mora dentro do Pipeline. O app não reimplementa parser.
    """
    return pipeline.predict_proba(df[list(CAMPOS_ABERTURA)])[:, 1]


def posicao_na_fila(escore: float, escores_do_lote: np.ndarray) -> int:
    """Posição (1-based) que um escore ocuparia na fila: quantos estão à frente + 1.

    Empate recebe a primeira posição do grupo, que é o que a ordenação estável faz.
    """
    return int(np.count_nonzero(escores_do_lote > escore)) + 1


def montar_fila(df: pd.DataFrame, pipeline, k: int | None = None) -> pd.DataFrame:
    """Ordena o lote pelo escore e corta em `k`, devolvendo a fila de perícia.

    Pressupõe lote **já aprovado pelo contrato** (ver `pontuar_lote`). A ordenação é
    estável e igual à da Fase 5 (`argsort` sobre o escore negativo), então a fila do app
    reproduz o `Resultado@900` auditado.
    """
    if len(df) == 0:
        fila = pd.DataFrame()
        fila.attrs["k"] = 0
        fila.attrs["n_lote"] = 0
        return fila

    if k is None:
        k = k_do_lote(len(df))
    k = int(min(k, len(df)))

    lote = df.reset_index(drop=True)
    escore = escore_da_politica(pipeline, lote)
    # `mergesort` é estável: escore idêntico preserva a ordem original do lote, que é
    # exatamente o critério que reproduz o número auditado na Fase 5.
    ordem = np.argsort(-escore, kind="mergesort")[:k]

    fila = pd.DataFrame(
        {
            "posicao": np.arange(1, len(ordem) + 1),
            "linha_no_lote": ordem,  # mesma numeração (0-based) dos erros do contrato
            "escore": escore[ordem],
            "valor_reclamado": lote["valor_reclamado"].to_numpy(dtype=object)[ordem],
            "faixa_prioridade": [
                faixa_prioridade(posicao, k) for posicao in range(1, len(ordem) + 1)
            ],
        }
    )
    if "id_sinistro" in lote.columns:
        fila.insert(1, "id_sinistro", lote["id_sinistro"].to_numpy(dtype=object)[ordem])
    for coluna in COLUNAS_CONTEXTO_FILA:
        if coluna in lote.columns:
            fila[coluna] = lote[coluna].to_numpy(dtype=object)[ordem]

    fila.attrs["k"] = k
    fila.attrs["n_lote"] = len(lote)
    return fila


def pontuar_lote(df: pd.DataFrame, pipeline, k: int | None = None) -> pd.DataFrame:
    """Rota única de pontuação da tela de fila: **contrato primeiro**, escore depois.

    Levanta `PayloadRecusado` se qualquer linha violar o contrato — e nesse caso **nada**
    é pontuado (decisão D1).
    """
    erros = validar_payload(df)
    if erros:
        raise PayloadRecusado(erros)
    return montar_fila(df, pipeline, k=k)


def avaliar_fila(fila: pd.DataFrame, gabarito: pd.Series | np.ndarray) -> dict[str, float | int]:
    """Autoavaliação da fila quando o lote traz o gabarito (o mês de teste traz).

    É leitura de auditoria do mês: quantas das fraudes ficaram dentro do corte. Sem
    gabarito no lote, a tela simplesmente não mostra a linha — o silêncio é honesto.
    """
    y = np.asarray(gabarito, dtype=int)
    posicoes = fila["linha_no_lote"].to_numpy(dtype=int)
    periciados = int(len(posicoes))
    fraudes = int(y[posicoes].sum()) if periciados else 0
    return {
        "periciados": periciados,
        "fraudes": fraudes,
        "precisao": float(fraudes / periciados) if periciados else 0.0,
    }


def resumo_fila(fila: pd.DataFrame, acerto: dict[str, float | int] | None = None) -> str:
    """Frase de cabeçalho da fila, em linguagem de fila — nunca de acusação."""
    if fila.empty:
        return "Lote vazio: nada a priorizar."
    k = int(fila.attrs.get("k", len(fila)))
    n_lote = int(fila.attrs.get("n_lote", len(fila)))
    if acerto is None:
        return (
            f"Fila de {k} perícias sobre {n_lote} sinistros do lote, ordenada pelo escore. "
            "Sem gabarito no lote, o acerto só pode ser medido no mês de teste."
        )
    return (
        f"Fila de {k} perícias sobre {n_lote} sinistros do lote. Nestas {k} primeiras "
        f"posições, {acerto['fraudes']} sinistros eram fraude "
        f"({acerto['precisao'] * 100:.2f}% de acerto) — leitura de auditoria do mês, "
        "não um veredito sobre cada caso."
    )


# ---------------------------------------------------------------------------
# Valor: por sinistro e do mês
# ---------------------------------------------------------------------------
def ler_curva_calibracao(caminho: Path | str = CAMINHO_CURVA_CALIBRACAO) -> pd.DataFrame:
    """Curva de calibração da Fase 5: probabilidade média × fração observada."""
    return pd.read_csv(caminho)


def valor_esperado(escore: float, curva: pd.DataFrame | None = None) -> dict[str, float]:
    """Valor esperado de periciar um sinistro — como **faixa**, não como certeza (D3).

    `valor_bruto = escore × 21.700 − 700` é a conta direta com a probabilidade do campeão.
    Como essa probabilidade não é calibrada, o número vem acompanhado da **taxa observada**
    da faixa de escore equivalente — é a leitura que sobrevive à auditoria.
    """
    if curva is None:
        curva = ler_curva_calibracao()

    bruto = float(escore) * CUSTO_FRAUDE_PAGA - CUSTO_PERICIA
    # Faixa de escore mais próxima na curva: é onde estão os sinistros comparáveis.
    linha = curva.loc[(curva["probabilidade_media"] - float(escore)).abs().idxmin()]
    taxa_observada = float(linha["fracao_positivos"])
    return {
        "escore": float(escore),
        "valor_bruto": bruto,
        "faixa_escore": float(linha["probabilidade_media"]),
        "taxa_observada": taxa_observada,
        "valor_com_taxa_observada": taxa_observada * CUSTO_FRAUDE_PAGA - CUSTO_PERICIA,
        "piso_viabilidade": PISO_VIABILIDADE,
    }


def painel_valor() -> dict[str, float | int]:
    """Painel da diretoria: o mês da política contra o mês da regra atual (Fase 5)."""
    return {
        "politica": RESULTADO_POLITICA,
        "regra_atual": RESULTADO_REGRA_ATUAL,
        "ganho": GANHO_POLITICA,
        "ganho_ic_inferior": GANHO_IC_INFERIOR,
        "ganho_ic_superior": GANHO_IC_SUPERIOR,
        "tp": TP_POLITICA_MES_TESTE,
        "pericias": PERICIAS_POLITICA_MES_TESTE,
        "n_sinistros": N_SINISTROS_MES_TESTE,
        "precisao": TP_POLITICA_MES_TESTE / PERICIAS_POLITICA_MES_TESTE,
        "piso": PISO_VIABILIDADE,
        "custo_pericia": CUSTO_PERICIA,
        "custo_fraude_paga": CUSTO_FRAUDE_PAGA,
    }


def medir_valor_da_politica(pipeline, teste: pd.DataFrame) -> dict[str, float | int]:
    """Recalcula o `Resultado@900` do campeão no mês de teste — conferência do painel.

    É a prova viva de que o app está lendo o artefato auditado: se este número divergir de
    `RESULTADO_POLITICA`, o painel perde validade e a tela avisa.
    """
    if COLUNA_GABARITO not in teste.columns:
        return {"erro": "lote sem coluna de gabarito: não há como medir o acerto"}

    lote = teste.reset_index(drop=True)
    fila = montar_fila(lote, pipeline, k=CAPACIDADE_MENSAL)
    acerto = avaliar_fila(fila, lote[COLUNA_GABARITO])
    resultado = acerto["fraudes"] * CUSTO_FRAUDE_PAGA - acerto["periciados"] * CUSTO_PERICIA
    return {
        "tp": acerto["fraudes"],
        "periciados": acerto["periciados"],
        "precisao": acerto["precisao"],
        "resultado": float(resultado),
        "confere": acerto["fraudes"] == TP_POLITICA_MES_TESTE,
    }


# ---------------------------------------------------------------------------
# Saúde do modelo e deriva de população
# ---------------------------------------------------------------------------
def hash_do_artefato(caminho: Path | str = CAMINHO_PIPELINE) -> str:
    """SHA-256 do `.joblib` implantado — a identidade do modelo em produção."""
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def _vazios(serie: pd.Series) -> int:
    """Contagem de vazios de um campo cru: nulo ou texto em branco."""
    return int((serie.isna() | (serie.astype("string").str.strip() == "")).sum())


def saude_do_modelo(pipeline, df: pd.DataFrame, caminho: Path | str = CAMINHO_PIPELINE) -> dict:
    """Bloco de saúde: identidade do artefato, vazios do lote e imputações (camada 3)."""
    resumo = resumo_imputacoes(pipeline, df)
    return {
        "hash": hash_do_artefato(caminho),
        "caminho": str(caminho),
        "n_linhas": int(len(df)),
        "vazios_por_campo": {
            campo: _vazios(df[campo]) for campo in CAMPOS_ABERTURA if campo in df.columns
        },
        "imputacoes": resumo,
        "alertas_imputacao": alertas_imputacao(resumo),
        "limiar_imputacao": LIMIAR_IMPUTACAO_SUSPEITA,
    }


def psi(referencia: pd.Series, atual: pd.Series, n_bandas: int = 10) -> float:
    """Population Stability Index entre a referência e a população atual.

    O PSI mede o quanto a distribuição de uma feature se moveu:
    `Σ (atual% − ref%) × ln(atual% / ref%)`. É o padrão de monitoramento já adotado pelo
    plano da Fase 5 (`docs/05_avaliacao.md`, §9.5), que manda re-treinar acima de 0,25.
    """
    ref = referencia.dropna()
    atu = atual.dropna()
    if ref.empty or atu.empty:
        return 0.0

    if pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(atu):
        # Bandas pelas quantis da referência: cada caixa tem a mesma massa no treino.
        cortes = np.unique(np.quantile(ref, np.linspace(0, 1, n_bandas + 1)))
        if len(cortes) < 3:
            return 0.0
        cortes[0], cortes[-1] = -np.inf, np.inf
        ref_share = pd.cut(ref, cortes, duplicates="drop").value_counts(normalize=True)
        atu_share = pd.cut(atu, cortes, duplicates="drop").value_counts(normalize=True)
    else:
        ref_share = ref.astype(str).value_counts(normalize=True)
        atu_share = atu.astype(str).value_counts(normalize=True)

    tamanho = 1e-6  # evita log(0) em banda que sumiu na população atual
    bandas = ref_share.index.union(atu_share.index)
    ref_pct = ref_share.reindex(bandas, fill_value=0.0).clip(lower=tamanho)
    atu_pct = atu_share.reindex(bandas, fill_value=0.0).clip(lower=tamanho)
    return float(((atu_pct - ref_pct) * np.log(atu_pct / ref_pct)).sum())


def drift_das_features(
    referencia: pd.DataFrame,
    atual: pd.DataFrame,
    features: tuple[str, ...] = FEATURES_DRIFT,
) -> pd.DataFrame:
    """PSI das features principais, com o veredito de cada uma.

    Recebe os dois lados **crus** (o chamado como chega) e aplica `preparar` neles; a
    comparação é feita sobre as features derivadas, que é o que o modelo enxerga.
    """
    ref = preparar(referencia)
    atu = preparar(atual)
    linhas = []
    for feature in features:
        valor = psi(ref[feature], atu[feature])
        if valor > LIMIAR_PSI_ALERTA:
            veredito = "re-treinar (deriva de população)"
        elif valor > LIMIAR_PSI_ATENCAO:
            veredito = "atenção"
        else:
            veredito = "estável"
        linhas.append({"feature": feature, "psi": valor, "veredito": veredito})
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Dados do app: artefato, lote de demonstração e fundo do SHAP
# ---------------------------------------------------------------------------
def _recurso(func):
    """Cacheia um objeto (ex.: modelo). Usa st.cache_resource no app e
    functools.lru_cache nos testes (onde não há sessão Streamlit)."""
    try:
        import streamlit as st

        return st.cache_resource(func)
    except Exception:
        return functools.lru_cache(maxsize=1)(func)


def _dados(func):
    """Cacheia dados (ex.: DataFrames). Usa st.cache_data no app e
    functools.lru_cache nos testes."""
    try:
        import streamlit as st

        return st.cache_data(func)
    except Exception:
        return functools.lru_cache(maxsize=1)(func)


@_recurso
def carregar_modelo(caminho: Path | str = CAMINHO_PIPELINE):
    """Carrega o artefato auditado. O app nunca treina — só usa."""
    return carregar_pipeline(caminho)


@_dados
def carregar_treino_e_teste() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treino e teste crus, pela mesma divisão temporal da Fase 3/4."""
    return dividir_temporal(carregar_base_crua(), data_corte=DATA_CORTE)


@_dados
def amostra_de_fundo(n: int = 100, semente: int = SEED) -> pd.DataFrame:
    """Amostra do treino que serve de referência ao SHAP (100 linhas bastam)."""
    treino, _ = carregar_treino_e_teste()
    return treino.sample(n=min(n, len(treino)), random_state=semente)


def explicar(
    pipeline,
    linha: pd.DataFrame,
    fundo: pd.DataFrame,
    n_fundo: int = 100,
) -> pd.Series:
    """Sinais SHAP de um sinistro, agregados por feature original (Fases 3 e 4)."""
    return explicar_sinistro(pipeline, linha, fundo, features_originais=FEATURES, n_fundo=n_fundo)


# ---------------------------------------------------------------------------
# Tela (Streamlit) — o `streamlit` entra só aqui, para os testes rodarem sem ele
# ---------------------------------------------------------------------------
def main() -> None:  # pragma: no cover - a tela é exercitada à mão e no smoke test
    """Renderiza as três telas do app de triagem."""
    import streamlit as st

    st.set_page_config(
        page_title="Quilha · Triagem de perícia",
        page_icon=str(RAIZ_PROJETO / "assets" / "brand" / "favicon.png"),
        layout="wide",
    )
    st.title("Triagem de perícia — Quilha Seguros")
    st.info(MENSAGEM_PERMANENTE)

    tela = st.sidebar.radio("Tela", ("Fila do mês", "Sinistro individual", "Valor e governança"))
    st.sidebar.caption(
        f"Capacidade: {CAPACIDADE_MENSAL} perícias/mês · piso de viabilidade "
        f"{PISO_VIABILIDADE * 100:.4f}%"
    )

    if tela == "Valor e governança":
        _tela_valor(st)
        return

    lote = _carregar_lote(st)
    if tela == "Fila do mês":
        _tela_fila(st, lote)
    else:
        _tela_individual(st, lote, _carregar_lote(st, chave="lote_individual"))


def _carregar_lote(st, chave: str = "lote_fila") -> pd.DataFrame:  # pragma: no cover
    """Lote de entrada: o CSV enviado ou o mês de teste como demonstração."""
    arquivo = st.sidebar.file_uploader("Lote de sinistros abertos (CSV)", type=["csv"], key=chave)
    if arquivo is not None:
        return pd.read_csv(arquivo, dtype=str)
    st.sidebar.caption("Sem arquivo: usando o mês de teste (agosto/2026) como demonstração.")
    _, teste = carregar_treino_e_teste()
    return teste


def _tela_fila(st, lote: pd.DataFrame) -> None:  # pragma: no cover - depende da UI
    """Tela 1 — fila do mês: contrato, escore, corte em `k` e exportação."""
    st.subheader("Fila do mês")
    pipeline = carregar_modelo()

    k_padrao = k_do_lote(len(lote))
    k = st.slider("Perícias a executar (k)", 0, max(len(lote), 1), k_padrao)
    st.caption(
        f"k proporcional ao lote: 900 × {len(lote)} / {SINISTROS_POR_MES} = {k_padrao}. "
        "A capacidade é restrição de negócio, não escolha do analista."
    )

    try:
        fila = pontuar_lote(lote, pipeline, k=k)
    except PayloadRecusado as recusa:
        st.error(
            f"{recusa}. O lote **não foi pontuado** (fail closed): uma fila incompleta "
            "omitiria sinistros sem avisar."
        )
        erros = recusa.como_tabela()
        st.dataframe(erros, width="stretch")
        st.download_button(
            "Exportar erros (CSV)",
            erros.to_csv(index=False).encode("utf-8"),
            file_name="erros_de_contrato.csv",
        )
        return

    lote_limpo = lote.reset_index(drop=True)
    tem_gabarito = COLUNA_GABARITO in lote_limpo.columns
    gabarito = lote_limpo[COLUNA_GABARITO] if tem_gabarito else None
    acerto = avaliar_fila(fila, gabarito) if gabarito is not None else None
    mensagem = resumo_fila(fila, acerto)
    (st.success if acerto is not None else st.info)(mensagem)

    st.caption(
        f"{len(lote)} sinistros no lote · {k} periciados · corte no escore "
        f"{fila['escore'].min():.6f}"
    )
    st.dataframe(fila, width="stretch")
    st.download_button(
        "Exportar fila (CSV)",
        fila.to_csv(index=False).encode("utf-8"),
        file_name="fila_pericia.csv",
    )


def _tela_individual(st, lote: pd.DataFrame, lote_individual: pd.DataFrame) -> None:
    """Tela 2 — sinistro individual: escore, posição, faixa, SHAP e governança."""
    # pragma: no cover - depende da UI
    st.subheader("Sinistro individual")
    pipeline = carregar_modelo()
    st.caption(
        "Governança: o modelo usa atributos demográficos (idade e UF) e a seleção é "
        "auditada. Eles aparecem no escore, mas não são destacados como 'razões'."
    )

    origem = st.radio("Origem do sinistro", ("Da fila do mês", "Digitar um chamado"))
    if origem == "Da fila do mês":
        linha_crua = lote_individual.head(1).reset_index(drop=True)
    else:
        valores = {campo: st.text_input(campo, "") for campo in CAMPOS_ABERTURA}
        linha_crua = pd.DataFrame([valores])

    if not st.button("Priorizar este sinistro"):
        return

    erros = validar_payload(linha_crua)
    if erros:
        st.error("O contrato recusou o chamado — corrija e reenvie:")
        st.dataframe(PayloadRecusado(erros).como_tabela(), width="stretch")
        return

    escores = escore_da_politica(pipeline, lote)
    escore = float(escore_da_politica(pipeline, linha_crua)[0])
    posicao = posicao_na_fila(escore, escores)

    col1, col2, col3 = st.columns(3)
    col1.metric("Escore do campeão", f"{escore:.6f}")
    col2.metric("Posição na fila", f"#{posicao} de {len(lote)}")
    col3.metric("Faixa de prioridade", faixa_prioridade(posicao, CAPACIDADE_MENSAL))

    valores = valor_esperado(escore)
    st.write(
        f"**Valor esperado da perícia:** R$ {formatar_reais(valores['valor_bruto'], 2)} "
        "— estimativa com a "
        f"probabilidade bruta do campeão. Na faixa de escore equivalente "
        f"({valores['faixa_escore']:.4f}), a **taxa observada** foi de "
        f"{valores['taxa_observada'] * 100:.2f}%, o que dá "
        f"R$ {formatar_reais(valores['valor_com_taxa_observada'], 2)}. Piso de viabilidade: "
        f"{valores['piso_viabilidade'] * 100:.4f}%."
    )

    st.write("### O que fez o modelo priorizar este sinistro")
    st.caption("Priorização, não evidência de irregularidade. A ordem é sugestão ao perito.")
    sinais = explicar(pipeline, linha_crua, amostra_de_fundo())
    st.bar_chart(sinais.head(10))


def _tela_valor(st) -> None:  # pragma: no cover - depende da UI
    """Tela 3 — valor e governança para a diretoria."""
    st.subheader("Valor e governança")
    painel = painel_valor()

    col1, col2, col3 = st.columns(3)
    col1.metric("Política de escore (mês)", f"R$ {formatar_reais(painel['politica'])}")
    col2.metric("Regra atual > R$ 50 mil (mês)", f"R$ {formatar_reais(painel['regra_atual'])}")
    col3.metric("Ganho por mês", f"R$ {formatar_reais(painel['ganho'])}")
    st.caption(
        f"IC 95% do ganho (bootstrap, 2.000 reamostragens): "
        f"R$ {formatar_reais(painel['ganho_ic_inferior'])} a "
        f"R$ {formatar_reais(painel['ganho_ic_superior'])}. "
        f"{painel['tp']} fraudes em {painel['pericias']} perícias "
        f"({painel['precisao'] * 100:.2f}% de acerto) no mês de teste de "
        f"{painel['n_sinistros']} sinistros."
    )
    st.caption(
        f"Estes {painel['pericias']} são a **capacidade cheia** de um mês de referência "
        f"({SINISTROS_POR_MES} sinistros) — é com ela que o número auditado foi medido. A fila "
        "operacional escala essa capacidade pelo tamanho do lote recebido (ver tela de fila): "
        "a **taxa** de perícia é que se mantém."
    )

    st.write("### Valor por sinistro")
    st.write(
        f"`valor esperado = chance × R$ {formatar_reais(painel['custo_fraude_paga'])} − "
        f"R$ {formatar_reais(painel['custo_pericia'])}`, exibido como faixa com a taxa "
        "observada ao lado: "
        "a calibração do campeão não é confiável (Brier 0,014702 sem calibração contra "
        "0,014688 com isotonic) e por isso a probabilidade bruta não é apresentada como previsão."
    )
    st.write(f"Piso de viabilidade de uma perícia: **{painel['piso'] * 100:.4f}%**.")

    st.write("### Saúde do modelo")
    pipeline = carregar_modelo()
    saude = saude_do_modelo(pipeline, _carregar_lote(st, chave="lote_saude"))
    st.write(f"Artefato: `{saude['caminho']}`")
    st.code(saude["hash"], language=None)
    st.write(f"Linhas avaliadas: {saude['n_linhas']}")
    st.dataframe(saude["imputacoes"], width="stretch")
    for alerta in saude["alertas_imputacao"]:
        st.warning(alerta)

    _, teste = carregar_treino_e_teste()
    conferencia = medir_valor_da_politica(pipeline, teste)
    if conferencia.get("confere"):
        st.success(
            f"Conferência do painel: o artefato reproduz o mês auditado — "
            f"{conferencia['tp']} fraudes em {conferencia['periciados']} perícias, "
            f"R$ {formatar_reais(conferencia['resultado'], 2)}."
        )
    else:
        st.error(f"O artefato NÃO reproduz o mês auditado: {conferencia}")

    st.write("### Deriva das features principais (PSI)")
    treino, _ = carregar_treino_e_teste()
    st.dataframe(
        drift_das_features(treino, _carregar_lote(st, chave="lote_drift")),
        width="stretch",
    )

    st.write("### Gatilhos de monitoramento e circuit breaker")
    st.table(
        pd.DataFrame(
            GATILHOS_MONITORAMENTO, columns=["Métrica mensal", "Gatilho de alerta", "Ação"]
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()

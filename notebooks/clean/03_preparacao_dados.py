# %% [markdown]
# # Fase 3 — Preparação dos Dados
#
# **Projeto:** Detecção de fraude em sinistros de automóvel — Quilha Seguros
# **Fonte:** `data/raw/quilha_sinistros.csv` (60.000 sinistros, 17 colunas, 01/04/2026 a 31/08/2026)
# **Objetivo da fase:** transformar a base crua em atributos prontos para o modelo,
# separar passado (treino) de futuro (teste) e montar o Pipeline que o app do
# Streamlit vai carregar — preparo, imputação e modelo em um único objeto.
#
# **Regra de ouro desta fase:** o CSV nunca é editado. Todo tratamento acontece em
# código, dentro de `src/quilha/preparacao.py`, e a decisão de cada tratamento é
# sustentada por um número da Fase 2.

# %%
# =============================================================================
# Configuração do ambiente, semente e caminhos do projeto
# =============================================================================
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Semente global do projeto: qualquer etapa aleatória (amostragem do fundo SHAP,
# validação cruzada da Fase 4) usa exatamente este valor.
SEED = 42


def encontrar_raiz(inicio: Path) -> Path:
    """Sobe na árvore de diretórios até encontrar a raiz do projeto (pyproject.toml)."""
    for pasta in [inicio.resolve(), *inicio.resolve().parents]:
        if (pasta / "pyproject.toml").exists():
            return pasta
    raise FileNotFoundError("Raiz do projeto não encontrada (pyproject.toml ausente).")


RAIZ = encontrar_raiz(Path.cwd())
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from quilha.dados import (  # noqa: E402
    DATA_CORTE,
    carregar_base_crua,
    dividir_temporal,
)
from quilha.explicacao import agregar_contribuicoes, mapear_features  # noqa: E402
from quilha.pipeline import (  # noqa: E402
    CAMINHO_PIPELINE,
    carregar_pipeline,
    construir_pipeline,
    salvar_pipeline,
    versoes,
)
from quilha.preparacao import (  # noqa: E402
    COLUNAS_BOOLEANAS,
    COLUNAS_CATEGORICAS,
    COLUNAS_NUMERICAS,
    FEATURES,
    preparar,
)

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 120)

print(f"Raiz do projeto : {RAIZ}")
print(f"Semente (SEED)  : {SEED}")

# %% [markdown]
# ## 1. Carga, tratamento e conferência de dimensões
#
# A base é lida **crua** (`dtype=str`), exatamente como foi entregue. Isso importa
# porque o app recebe um chamado novo pelo mesmo formato: a função `preparar`
# precisa funcionar com o texto que chega, não com um dado já tipado por nós.

# %%
base_crua = carregar_base_crua()

print(f"Dimensões : {base_crua.shape[0]:,} linhas x {base_crua.shape[1]} colunas")
print(f"Colunas   : {list(base_crua.columns)}")

# %%
# `preparar` é a função pública da fase: pura, determinística, sem estatística
# aprendida. Roda igual com a base inteira ou com uma linha só.
base_preparada = preparar(base_crua)

print(f"Entrada : {base_crua.shape[0]:,} linhas x {base_crua.shape[1]} colunas (schema cru)")
print(f"Saída   : {base_preparada.shape[0]:,} linhas x {base_preparada.shape[1]} colunas")
print(f"Saída pode conter: 3 colunas de controle + {len(FEATURES)} features")
print()
print("Colunas de saída:")
print(list(base_preparada.columns))

# %%
# Nenhuma linha é descartada: a contagem de linhas tem de ser idêntica.
print(f"Linhas descartadas por nulo ou outlier: {len(base_crua) - len(base_preparada)}")
print(f"Nulos novos em campos numéricos: {int(base_preparada[list(COLUNAS_NUMERICAS)].isna().sum().sum())}")

# %% [markdown]
# ## 2. Antes x depois, coluna por coluna
#
# Duas evidências exigidas pela fase: o **dtype** e a **contagem de nulos** antes e
# depois. A coluna `boletim_ocorrencia` é o caso interessante — o branco não é
# imputado, vira categoria explícita (item d), então o nulo do cru desaparece
# virando sinal, não virando média.

# %%
def resumo_coluna(nome: str) -> dict:
    """dtype e nº de nulos de uma coluna, antes e depois de `preparar`."""
    if nome in base_crua.columns:
        antes_dtype, antes_nulos = str(base_crua[nome].dtype), int(base_crua[nome].isna().sum())
    else:
        antes_dtype, antes_nulos = "—", "—"
    if nome in base_preparada.columns:
        depois_dtype = str(base_preparada[nome].dtype)
        depois_nulos = int(base_preparada[nome].isna().sum())
    else:
        depois_dtype, depois_nulos = "—", "—"
    return {
        "coluna": nome,
        "dtype_antes": antes_dtype,
        "nulos_antes": antes_nulos,
        "dtype_depois": depois_dtype,
        "nulos_depois": depois_nulos,
    }


nomes = ["id_sinistro", "data_ocorrencia", "data_abertura", *FEATURES]
resumo = pd.DataFrame([resumo_coluna(n) for n in nomes])
display(resumo)

# %%
# Colunas que existem no CSV e NÃO saem em `preparar`. Cada uma tem um motivo.
descartadas = {
    "fraude": "gabarito do case — nunca é atributo",
    "valor_indenizado": "VAZAMENTO: 13,44% dos sinistros com 0,00 e taxa de fraude "
    "de 17,37% neles contra 0,15% nos demais (115x); só existe no encerramento",
    "terceiro_identificado": "redundante: só varia dentro de 'Colisão'; o sinal útil "
    "foi recuperado em `tem_terceiro_identificado`",
}
pd.DataFrame(
    {"coluna": list(descartadas), "motivo": list(descartadas.values())}
)

# %% [markdown]
# ## 3. Divisão treino/teste por tempo
#
# A decisão do ponto de corte sai de evidência: a taxa de fraude e o volume de cada
# mês de abertura. Um corte mal escolhido cria um teste com fraude de menos ou com
# um período atípico, e aí a avaliação da Fase 5 mede a sorte do mês, não o modelo.

# %%
base_tipada = base_crua.copy()
base_tipada["data_abertura"] = pd.to_datetime(base_tipada["data_abertura"])
base_tipada["fraude"] = base_tipada["fraude"].astype(int)

por_mes = base_tipada.groupby(base_tipada["data_abertura"].dt.to_period("M").astype(str)).agg(
    sinistros=("fraude", "size"), fraudes=("fraude", "sum")
)
por_mes["taxa_%"] = (por_mes["fraudes"] / por_mes["sinistros"] * 100).round(2)
display(por_mes)

print(f"Taxa de fraude da base inteira (referência): {base_tipada['fraude'].mean() * 100:.2f}%")
print(f"Amplitude da taxa mensal: {por_mes['taxa_%'].max() - por_mes['taxa_%'].min():.2f} ponto percentual")
print(f"Corte em {DATA_CORTE.date()} — teste = período mais recente, sem shuffle e sem stratify.")

# %%
treino, teste = dividir_temporal(base_crua)

periodo_treino = (treino["data_abertura"].min(), treino["data_abertura"].max())
periodo_teste = (teste["data_abertura"].min(), teste["data_abertura"].max())
fraudes_treino = int(treino["fraude"].astype(int).sum())
fraudes_teste = int(teste["fraude"].astype(int).sum())

divisao = pd.DataFrame(
    [
        {
            "conjunto": "treino",
            "de": periodo_treino[0],
            "ate": periodo_treino[1],
            "sinistros": len(treino),
            "fraudes": fraudes_treino,
            "taxa_%": round(fraudes_treino / len(treino) * 100, 2),
        },
        {
            "conjunto": "teste",
            "de": periodo_teste[0],
            "ate": periodo_teste[1],
            "sinistros": len(teste),
            "fraudes": fraudes_teste,
            "taxa_%": round(fraudes_teste / len(teste) * 100, 2),
        },
    ]
)
display(divisao)

# %% [markdown]
# ## 4. Taxa de fraude das categorias e faixas criadas
#
# É o teste de utilidade de cada atributo derivado: se a taxa de fraude não se move
# entre as categorias, o atributo não separa nada.

# %%
def taxa_por(rotulo: str, coluna: str, base_alvo: pd.DataFrame = base_preparada) -> None:
    """Taxa de fraude por categoria de `coluna` (a coluna `fraude` vem da base crua)."""
    alvo = base_crua["fraude"].astype(int)
    tabela = pd.DataFrame({"sinistros": base_alvo[coluna].value_counts()})
    tabela["fraudes"] = alvo.groupby(base_alvo[coluna], observed=True).sum()
    tabela["taxa_%"] = (tabela["fraudes"] / tabela["sinistros"] * 100).round(2)
    print(f"=== {rotulo} ===")
    display(tabela.sort_index())
    print()


taxa_por("boletim_ocorrencia (categoria 'Não informado' incluída)", "boletim_ocorrencia")
taxa_por("tem_terceiro_identificado", "tem_terceiro_identificado")
taxa_por("faixa_apolice", "faixa_apolice")
taxa_por("faixa_valor_reclamado", "faixa_valor_reclamado")
taxa_por("periodo_dia", "periodo_dia")
taxa_por("fim_de_semana", "fim_de_semana")

# %% [markdown]
# ## 5. O Pipeline — preparo, imputação e modelo em um único objeto
#
# O app do Streamlit recebe o chamado cru, carrega **um** arquivo e chama
# `predict_proba`. Nenhum tratamento pode morar fora do objeto salvo. O modelo aqui
# ainda é um `DummyClassifier`: ele existe só para provar que o Pipeline roda de
# ponta a ponta. O modelo de verdade entra na Fase 4.

# %%
# ── Ajuste SÓ no treino ──────────────────────────────────────────────────────
# Nenhum transformador com `fit` pode ver o teste. O `Pipeline` garante isso: o
# `fit` recebe o treino e o `predict_proba` recebe o teste logo depois.
sem_gabarito = ["fraude", "valor_indenizado"]

pipeline = construir_pipeline()
pipeline.fit(treino.drop(columns=sem_gabarito), treino["fraude"].astype(int))

print("Estrutura do Pipeline:")
for nome, etapa in pipeline.steps:
    print(f"  {nome:10s} -> {type(etapa).__name__}")

# %%
# ── Prova de ponta a ponta: UMA linha crua, como o app vai mandar ────────────
linha_crua = teste.drop(columns=sem_gabarito).iloc[[13]]
print("Chamado cru (uma linha, 15 colunas):")
display(linha_crua.T.head(20))

probabilidade = pipeline.predict_proba(linha_crua)
print(f"predict_proba -> classe legítimo = {probabilidade[0, 0]:.6f} | fraude = {probabilidade[0, 1]:.6f}")
print(f"Gabarito do sinistro {linha_crua['id_sinistro'].iloc[0]}: fraude = {teste['fraude'].iloc[13]}")

# %%
# ── Robustez: categoria nova e campo em branco não podem quebrar o app ───────
casos = {
    "categoria nunca vista": {"tipo_sinistro": "Categoria Nova", "canal_abertura": "Canal Novo"},
    "campos em branco": {
        "valor_reclamado": "",
        "valor_fipe": "",
        "idade_segurado": "",
        "boletim_ocorrencia": "",
    },
    "data malformada": {"data_ocorrencia": "31/02/2026", "data_abertura": "sem-data"},
}
for rotulo, alteracoes in casos.items():
    chamado = linha_crua.copy()
    for coluna, valor in alteracoes.items():
        chamado[coluna] = valor
    theta = pipeline.predict_proba(chamado)
    print(f"{rotulo:24s} -> fraude = {theta[0, 1]:.6f}  (sem erro)")

# %% [markdown]
# ## 6. Persistência e round-trip do artefato
#
# O `.gitignore` passou a versionar `models/*.joblib`: o Streamlit Cloud clona o
# repositório e não tem nenhum armazenamento de artefatos separado, então o app só
# funciona se o Pipeline estiver no repo.

# %%
caminho_salvo = salvar_pipeline(pipeline)
print(f"Artefato salvo em : {caminho_salvo}")
print(f"Registro de versões: {caminho_salvo.with_suffix('.versoes.json').name}")
print()
print("Versões que precisam bater entre local treinado e nuvem que serve:")
for chave, valor in versoes().items():
    print(f"  {chave:14s}: {valor}")
print()
print(f"Tamanho do arquivo: {caminho_salvo.stat().st_size / 1024:.1f} KB")

# %%
recarregado = carregar_pipeline()
antes = pipeline.predict_proba(linha_crua)[0, 1]
depois = recarregado.predict_proba(linha_crua)[0, 1]
print(f"Antes de salvar  : {antes:.10f}")
print(f"Depois de carregar: {depois:.10f}")
print(f"Mesmo valor       : {antes == depois}")

# %% [markdown]
# ## 7. Agregação SHAP por feature original
#
# O `OneHotEncoder` dilui uma variável em várias colunas. O SHAP explica coluna a
# coluna, então a explicação crua não é legível para o negócio. A função de
# agregação soma as colunas codificadas de volta à feature original — inclusive as
# colunas criadas pelo marcador de dado ausente (`add_indicator`).
#
# Nesta fase validamos a **lógica** de agregação com uma matriz sintética. A
# validação ponta a ponta com o modelo real acontece na Fase 4.

# %%
nomes_transformados = [
    "numericas__idade_segurado",
    "numericas__valor_fipe",
    "numericas__missingindicator_valor_fipe",
    "categoricas__tipo_sinistro_Colisão",
    "categoricas__tipo_sinistro_Roubo",
    "categoricas__uf_SP",
    "categoricas__faixa_valor_reclamado_40 a 50",
]
valores_sinteticos = np.array([[1.0, 2.0, 0.5, -3.0, 4.0, 1.5, -2.0]])

mapa = mapear_features(nomes_transformados)
display(pd.Series(mapa, name="feature_original").to_frame())

# %%
agregado = agregar_contribuicoes(valores_sinteticos, nomes_transformados)
display(agregado.T.rename(columns={0: "contribuicao"}))

print(f"Soma das colunas transformadas : {valores_sinteticos.sum():.4f}")
print(f"Soma depois da agregação       : {agregado.to_numpy().sum():.4f}")
print(f"Contribuição preservada        : {np.isclose(valores_sinteticos.sum(), agregado.to_numpy().sum())}")
print()
print("O one-hot de `tipo_sinistro` virou UMA linha: -3,0 + 4,0 = 1,0")
print("A coluna de indicador somou na feature de origem: 2,0 + 0,5 = 2,5")

# %% [markdown]
# ## 8. Features finais do modelo
#
# A lista abaixo é o conjunto de atributos que entra na modelagem, na ordem
# canônica. Nada dela existe depois da decisão de perícia — o modelo decide no
# momento da abertura.

# %%
features = pd.DataFrame(
    {
        "feature": list(FEATURES),
        "tipo": (
            ["numérica"] * len(COLUNAS_NUMERICAS)
            + ["booleana"] * len(COLUNAS_BOOLEANAS)
            + ["categórica"] * len(COLUNAS_CATEGORICAS)
        ),
    }
)
display(features)

print(f"Total de features : {len(FEATURES)}")
print(f"Vazamento presente: {'valor_indenizado' in FEATURES or 'fraude' in FEATURES}")

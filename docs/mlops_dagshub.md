# Runbook — DagsHub (tracking de experimentos)

**Projeto:** Detecção de fraude em sinistros de automóvel — Quilha Seguros
**Objetivo:** enviar as runs do MLflow (experimento `quilha-fraude-fase4`) para o DagsHub,
mantendo a degradação limpa — sem credencial, o treino continua rodando local.

---

## Antes de tudo: quem faz o quê

DagsHub, GitHub Secrets e GitHub Actions são ferramentas diferentes, com trabalhos
diferentes. Confundir as três é o erro mais comum:

| Ferramenta | Para que serve | Necessária para levar experimentos ao DagsHub? |
|---|---|---|
| **DagsHub** | Servidor de tracking/MLflow na nuvem | **Sim** — é o destino |
| **GitHub Secrets** | Guardar credenciais para o CI | **Não** — só se o treino rodar na nuvem |
| **GitHub Actions** | Rodar código automaticamente no push | **Não** — o treino roda local |

O único passo no lado do GitHub é **autorizar o DagsHub como GitHub App**.

### Aviso técnico que muda o planejamento

O MLflow **não sincroniza runs entre servidores**. As runs já gravadas no `mlflow.db`
local **não sobem sozinhas** para o DagsHub. Levar histórico existente exige
**re-executar** o código com o tracking apontado para o DagsHub.

---

## Como o código decide o destino

`quilha.mlops.configurar_tracking()` lê três variáveis de ambiente. Com as **três**
preenchidas, usa o DagsHub; sem elas, cai no SQLite local (`sqlite:///mlflow.db`).

- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`

As credenciais vivem **apenas** em variável de ambiente — nunca no código, nunca no
repositório.

---

## Passo a passo

### Parte 1 — DagsHub

1. Acesse `dagshub.com` → **Sign in with GitHub**.
2. Quando o GitHub pedir, **autorize o DagsHub GitHub App** — conceda acesso apenas ao
   repositório `Auto-Insurance-Fraud`, não a "todos" (princípio do menor privilégio).
3. **Create → Connect a repository → Import from GitHub** → selecione
   `flaviohenriquehb777/Auto-Insurance-Fraud`.
4. No repositório, abra a aba **Remote** e copie a **MLflow Tracking URI**:

   ```
   https://dagshub.com/flaviohenriquehb777/Auto-Insurance-Fraud.mlflow
   ```

### Parte 2 — Token

5. Perfil → **Settings → Tokens → Create new token** (nome sugerido: `quilha-mlflow`).
   Copie o valor — ele é exibido **uma única vez**.

### Parte 3 — Variáveis de ambiente (máquina local, onde o treino roda)

6. No PowerShell, fora do repositório:

   ```powershell
   [Environment]::SetEnvironmentVariable('MLFLOW_TRACKING_URI', 'https://dagshub.com/flaviohenriquehb777/Auto-Insurance-Fraud.mlflow', 'User')
   [Environment]::SetEnvironmentVariable('MLFLOW_TRACKING_USERNAME', 'flaviohenriquehb777', 'User')
   [Environment]::SetEnvironmentVariable('MLFLOW_TRACKING_PASSWORD', '<seu-token-do-dagshub>', 'User')
   ```

7. **Feche e reabra** o terminal/editor para herdar as variáveis.

### Parte 4 — Verificação

8. Confirme que o código saiu do local e aponta para o DagsHub:

   ```powershell
   $env:PYTHONPATH = 'src'; uv run python -c "from quilha.mlops import configurar_tracking, credenciais_presentes; print(credenciais_presentes()); print(configurar_tracking())"
   ```

   Esperado: `True` e `('https://dagshub.com/flaviohenriquehb777/Auto-Insurance-Fraud.mlflow', False)`.
   Se vier `False` e `sqlite:///...`, a variável não foi herdada pelo processo.

### Parte 5 — GitHub (somente se um dia treinarmos na nuvem)

9. **Não é necessário agora.** GitHub Secrets + Actions só entram se o treino for
   executado automaticamente no push. Como o treino precisa da base (6,57 MB) e leva
   ~30 min, isso é caro e desnecessário — o `ci.yml` do projeto faz lint + testes, que
   é o escopo correto de CI. Se um dia for preciso:
   `Settings → Secrets and variables → Actions → New repository secret`, com os três
   nomes acima.

---

## Erros comuns

| Sintoma | Causa provável | Ação |
|---|---|---|
| `credenciais_presentes()` devolve `False` | Variáveis não herdadas pelo processo | Reabrir o terminal/editor |
| Aviso "Credenciais do MLflow ausentes" no treino | Faltou uma das três variáveis | Conferir nome e valor das três |
| `401`/`403` ao logar | Token incorreto ou expirado | Gerar novo token no DagsHub |
| Runs aparecem em `mlflow.db` mesmo com credencial | `MLFLOW_TRACKING_URI` vazia | Revalidar a Parte 4 |

---

## Regra de segurança

- Credenciais **somente** em variável de ambiente — nunca em código, nunca no repositório.
- O `mlflow.db` e o `mlartifacts/` já estão no `.gitignore`.
- A autorização do GitHub App deve ser restrita ao repositório necessário.

---

## Estado sem credencial (garantia de portabilidade)

Sem as três variáveis (ou sem o pacote `mlflow`), `contexto_run` devolve um `Registro`
inativo: todos os métodos de logging viram *no-op* e o treino segue até o fim. É o que
permite o notebook consolidado rodar sem token nenhum.

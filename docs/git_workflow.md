# Fluxo Git do projeto

## Objetivo

Manter a `main` como trilha estavel do projeto e usar branches curtas para cada bloco de trabalho.
Assim, a validacao automatica roda antes do merge e cada marco importante fica mais auditavel.

## Padrao adotado

1. Abrir uma branch de trabalho a partir da `main`.
2. Desenvolver e validar localmente.
3. Subir a branch para o remoto.
4. Abrir um Pull Request para `main`.
5. Deixar o CI rodar no PR.
6. Fazer o merge somente quando o PR estiver validado.

## Convencao de nomes

- `feature/fase-4-modelagem` para nova entrega
- `fix/...` para correcao
- `docs/...` para documentacao
- `chore/...` para manutencao

## Comandos principais

```bash
git switch main
git pull origin main
git switch -c feature/fase-4-modelagem
```

Depois de desenvolver:

```bash
uv run ruff check src tests
uv run pytest -q
git push -u origin feature/fase-4-modelagem
gh pr create --base main --head feature/fase-4-modelagem
```

## Por que isso e mais profissional

- A `main` deixa de ser area de experimentacao e vira trilha de entrega.
- O CI roda antes do merge, reduzindo regressao silenciosa.
- O PR concentra contexto tecnico, evidencias e riscos.
- O historico fica mais claro: branch para trabalho, tag para marco, release para entrega.

## O que muda na pratica neste projeto

Da Fase 4 em diante, o fluxo recomendado e:

1. trabalhar em `feature/...`;
2. abrir PR para `main`;
3. usar tag e release apenas quando o marco estiver fechado.

Esse padrao e suficiente para projeto individual e ja conversa bem com repositorio de portfolio,
CI do GitHub Actions e futura colaboracao com revisores.

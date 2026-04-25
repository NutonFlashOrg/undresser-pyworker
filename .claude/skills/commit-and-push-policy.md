---
name: commit-and-push-policy
description: Defines how agents must commit and push code. Conventional Commits format, Paperclip co-author trailer, no bypass flags, no force push, PRs require board approval.
---

# Commit and Push Policy

## Commit message format

Follow Conventional Commits (https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

<optional body>

Co-Authored-By: Paperclip <noreply@paperclip.ing>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`

Scope is optional — use the module or file area (e.g. `auth`, `generation`, `worker`).

## Required trailer

Every commit made by an agent MUST include:

```
Co-Authored-By: Paperclip <noreply@paperclip.ing>
```

Do not put the agent name in the trailer. Use exactly `Co-Authored-By: Paperclip <noreply@paperclip.ing>`.

## Forbidden flags

- Never use `--no-verify` (bypasses hooks)
- Never use `--amend` on a commit that has already been pushed to the remote
- Never use `--force-with-lease` or `--force` on any push
- Never use `-c commit.gpgsign=false` or `--no-gpg-sign`

## Push target rules

- Only push to branches with the `paperclip/` prefix
- Never push to `main`, `staging`, or any `feature/*` branch
- Target remote: `origin`

## PR policy

- PRs from agent branches always target `main`
- PRs require board review and approval before merge
- Agents MUST NOT merge their own PRs
- Do not open PRs until the CEO signals the feature set is ready for board review

## Workflow

```bash
git add <specific files>          # never git add -A or git add .
git commit -m "$(cat <<'EOF'
feat(scope): summary line

Body if needed.

Co-Authored-By: Paperclip <noreply@paperclip.ing>
EOF
)"
git push origin paperclip/<role>/<slug>
```

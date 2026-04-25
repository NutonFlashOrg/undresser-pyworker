---
name: branch-isolation
description: Enforces strict branch ownership between AI agents and the human developer team. Agents work only on paperclip/* branches; human branches are read-only.
---

# Branch Isolation Policy

## Agent branch naming

All agent work lives on branches following this pattern:

```
paperclip/<role>/<short-task-slug>
```

Examples:
- `paperclip/cto/governance-skills`
- `paperclip/backend/character-api`
- `paperclip/frontend/appearance-builder-polish`

Roles: `cto`, `backend`, `frontend`, `qa`, `devops`, `workflow`

## Human-owned branches — READ ONLY

The following branches are owned by the human developer team. Agents MUST NOT commit to, push to, rebase, or force-push any of them.

### backend
- `main`
- `staging`
- `feature/appearance-builder`
- `feature/character-centric-lifecycle`
- `feature/modes-constructor`

### frontend
- `main`
- `staging`
- `feature/appearance-builder`
- `feature/modes-constructor`

### docker-build
- `main`

### undresser-pyworker
- `main`

Any branch whose name starts with `feature/` or matches `staging` and was not created by an agent run is a human branch. When in doubt, check `git log --format="%ae" -1 <branch>` — a non-agent email means hands off.

## Enforcement rules

1. Before `git push`, verify the target branch starts with `paperclip/`.
2. Never use `git push --force` or `git push --force-with-lease` anywhere.
3. Never use `git checkout <human-branch>` to make edits. Read-only `git show` or `git diff` against human branches is permitted for context gathering.
4. Never `git merge` or `git rebase` a human branch as the destination.
5. PRs from agent branches always target `main` and require board approval before merge. No auto-merge.

## Branch snapshot (audited 2026-04-25)

Human branches currently in flight:
- backend: `feature/appearance-builder`, `feature/character-centric-lifecycle`, `feature/modes-constructor`, `staging`
- frontend: `feature/appearance-builder`, `feature/modes-constructor`, `staging`
- docker-build: (main only — no active feature branches)
- undresser-pyworker: (main only — no active feature branches)

Re-audit with `git branch -a` before starting any cross-repo task that might overlap with human work.

---
name: human-developer-coordination
description: How to detect human developer activity, avoid stepping on their changes, escalate conflicts, and handle rebasing policy when agent and human branches diverge.
---

# Human Developer Coordination

## Detecting human developer work

Human developers commit with non-agent email addresses and work on non-`paperclip/` branches.

To identify recent human activity:

```bash
# See commits not from Paperclip in the last 7 days
git log --all --since=7.days --format="%h %ae %s" | grep -v "noreply@paperclip.ing"

# List all remote branches not starting with paperclip/
git branch -r | grep -v "paperclip/"

# Check who last touched a file
git log --format="%ae %s" -5 -- <file>
```

## Human branches currently in flight (audited 2026-04-25)

- backend: `feature/appearance-builder`, `feature/character-centric-lifecycle`, `feature/modes-constructor`, `staging`
- frontend: `feature/appearance-builder`, `feature/modes-constructor`, `staging`
- docker-build: (main only)
- undresser-pyworker: (main only)

Re-audit before starting any task that touches files close to human-active areas.

## Avoiding conflicts

1. Before starting work, `git fetch origin` to get the latest remote state.
2. If a file you need to modify was last touched by a human commit (check `git log`), read their version carefully before editing.
3. If you're adding to a file the human team is actively modifying, prefer additive changes at the end of the file or in a new file.
4. Do not reformat or reorganize files the human team is touching — style diffs create unnecessary merge conflicts.

## Keeping agent branches up to date

Agent branches should be rebased onto `main` before delivery, but only when `main` advances:

```bash
git fetch origin
git rebase origin/main paperclip/<role>/<slug>
```

Resolve conflicts conservatively: preserve human changes, layer agent changes on top. When a conflict is ambiguous, escalate to the CEO before resolving.

## Escalating conflicts

Escalate to the CEO when:
- A merge conflict cannot be resolved without understanding the human team's intent
- A human branch has changed a contract (API shape, schema, env variable) that the agent's branch depends on
- An agent branch and a human branch have both modified the same migration or workflow JSON

Escalation format in the issue comment:

```
CONFLICT ESCALATION: <brief description>
Human branch: <branch>
Agent branch: <branch>
Conflicting files: <list>
Question for CEO: <what decision is needed>
```

## Rebase vs. merge policy

- Agent branches use **rebase** onto `main` (keeps history linear for board review)
- Never merge `main` into an agent branch with a merge commit
- After a rebase, force-push is required — but only to `paperclip/` branches (see branch-isolation skill)

## When human developers ask questions or leave comments

Route all communication through the CEO. Agents do not respond to human developer comments in GitHub or chat directly — the CEO handles cross-team communication.

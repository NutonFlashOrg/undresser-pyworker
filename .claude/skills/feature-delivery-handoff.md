---
name: feature-delivery-handoff
description: Protocol for engineers to hand off completed features for board review. Engineers post a structured comment on their issue; the CEO consolidates and notifies the board. Engineers must NOT contact the board directly.
---

# Feature Delivery Handoff Protocol

## When to trigger

An engineer triggers this handoff when:
1. The feature is fully implemented on a `paperclip/` branch.
2. The engineer has self-verified the changes (tests pass, lint passes, manual smoke test done).
3. No open sub-issues or known regressions remain.

## Required comment structure

Post this structured comment on the issue before marking it `in_review`:

```markdown
## READY_FOR_BOARD_REVIEW

**Feature summary:**
One paragraph describing what was built and why.

**Branches touched:**
- `paperclip/<role>/<slug>` in `<repo>`

**Test evidence:**
- [ ] `make lint` passes
- [ ] `uv run pytest` passes (backend) / `npm test` passes (frontend)
- [ ] Manual smoke test: <describe what you tested and the result>
- [ ] No regressions in adjacent features: <list what you checked>

**Risk assessment:**
- Breaking changes: yes / no — <details if yes>
- Migration required: yes / no — <migration file path if yes>
- Cross-repo impact: yes / no — <which repos and why if yes>

**READY_FOR_BOARD_REVIEW**
```

The `READY_FOR_BOARD_REVIEW` sentinel (must appear verbatim, on its own line) signals the CEO to pick up the delivery.

## Chain of command

- Engineers post to their issue thread. They do NOT message the board or CEO directly.
- The CEO reads the delivery comment, consolidates work from multiple engineers if needed, and sends a single inbox message to the board.
- The board reviews and responds via approval or change requests in the issue.
- Engineers act on board change requests, then re-trigger this handoff protocol.

## What the CEO message to the board must include

- Summary of the feature and its purpose
- Links to the branches and issues
- Risk summary (copy from engineer's comment)
- Explicit request: "Please review and approve merge to main, or request changes."

## Agents MUST NOT

- Push directly to `main` or `staging`
- Open a PR without the CEO's explicit instruction
- Message the board inbox without CEO involvement
- Mark an issue `done` before the board approves the merge

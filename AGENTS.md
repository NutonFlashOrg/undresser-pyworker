## Mandatory Agent Skills

All agents working in this repository MUST apply the following governance skills before writing code, committing, or pushing:

| Skill | Purpose |
|-------|---------|
| `branch-isolation` | Agent vs human branch separation — agents stay on `paperclip/*` only |
| `commit-and-push-policy` | Conventional Commits + Paperclip trailer; no force-push; PRs require board approval |
| `senior-dev-style-backend` | Python/FastAPI code quality bar (backend repo) |
| `senior-dev-style-frontend` | TypeScript/React code quality bar (frontend repo) |
| `feature-delivery-handoff` | Structured handoff comment before marking ready for board review |
| `human-developer-coordination` | Detecting and avoiding conflicts with the human developer team |

Skills are located at `.claude/skills/` in each repo and at the workspace root (`/srv/ai-company/workspaces/nudelab/.claude/skills/`).

# AGENTS.md — undresser-pyworker

Cloud Code guidance for the `undresser-pyworker/` repository (NudeLab Python worker for Vast.ai).

---

## Branch

Current working branch:
- `main`

## Role In Product

This repo owns the Python worker logic running beside the model backend in Vast serverless environments.

It is responsible for:
- request parsing
- workload calculation
- payload transformation
- calibration scripts
- worker bootstrap behavior

## Current Reality

For the current NudeLab generation stack, the most important worker family is:
- `workers/comfyui-json/`

Key files:
- `start_server.sh`
- `workers/comfyui-json/worker.py`
- `workers/comfyui-json/workflow_transform.py`
- `scripts/calibrate_vast_workload_multi_lane.py`

## Product Relationship

This repo is downstream of product decisions only through concrete workflow/runtime contracts.

That means:
- UI-level changes do not matter here unless they change request shape
- backend orchestration changes matter if they change payload fields
- docker-build changes matter if they change startup/runtime assumptions

## Practical Rule

If the app/backend introduces:
- character-aware generation payloads
- new dataset-generation workflow payloads
- different LoRA binding patterns

check whether the worker transform layer must explicitly support them.

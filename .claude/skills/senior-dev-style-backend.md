---
name: senior-dev-style-backend
description: Python/FastAPI code quality bar for the backend repo. Typed everywhere, async-first, reversible migrations, no global state, structured logging, fail-loud config.
---

# Backend Code Style — Python / FastAPI

## Language and typing

- Python 3.11+. All public functions and methods must have full type annotations.
- No untyped `dict`, `list`, or `Any` without a specific justification.
- Use `TypedDict`, `dataclass`, or Pydantic models for structured data.
- Prefer `Optional[X]` → `X | None` (Python 3.10+ union syntax).

## Async

- Prefer `async def` for all FastAPI route handlers and service calls.
- Use `httpx.AsyncClient`, not `requests`, for outbound HTTP.
- Do not call blocking I/O inside an `async def` without `asyncio.to_thread`.

## FastAPI conventions

- Route handlers should be thin: validate input, call a service function, return a response.
- Use Pydantic v2 models for request/response schemas.
- Dependency injection via `Depends()` for auth, db sessions, config.
- Do not import app-level dependencies at module top level if they require configuration to be loaded.

## Database / Alembic

- Every schema change gets an Alembic migration.
- Migrations must be reversible: write the `downgrade()` function completely.
- Never modify an existing migration that has already been applied to `main` or `staging`.
- Do not use `op.execute("ALTER TABLE ...")` raw SQL unless Alembic helpers don't cover the case.

## State and config

- No global mutable state. Use FastAPI `lifespan` for startup/shutdown resources.
- Config is loaded from environment variables via a settings object (`pydantic_settings.BaseSettings`).
- Fail loudly on missing required config at startup — no silent fallbacks to `None`.
- No secrets or API keys committed to source.

## Logging

- Use Python's `logging` module with structured output (JSON via `python-json-logger` or equivalent).
- Do not use `print()` for debugging. Remove all print statements before committing.
- Log at `WARNING` or above for anything that indicates degraded behaviour.

## Testing

- All public service functions must have at least one test.
- Use `pytest` + `pytest-asyncio` for async tests.
- Lint: `make lint` (runs `ruff check app/`)
- Format: `make format` (runs `ruff format app/`)
- Type check: `uv run mypy app/` (if mypy is present in pyproject.toml)
- Tests: `uv run pytest`

## What not to do

- Do not commit commented-out code.
- Do not use `except Exception: pass`.
- Do not shadow built-in names (`id`, `type`, `input`, `list`, etc.).
- Do not add `# type: ignore` without a comment explaining why.

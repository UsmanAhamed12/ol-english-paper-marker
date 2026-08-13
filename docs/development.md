# Development workflow

## Phase discipline

Development follows the ordered phases in `AGENTS.md`. Each phase uses:

```text
Inspect -> Plan -> Implement -> Test -> Validate -> Document -> Stop
```

Do not start the next phase until the current phase is complete and the user
provides `CONTINUE`.

## Dependency management

Use uv exclusively for project dependencies and command execution:

```bash
uv add package-name
uv add --dev package-name
uv sync
uv run command
```

Do not use Poetry, Conda, or direct `pip` installs. Add a dependency only in the
phase that needs it. The current dependency set intentionally excludes AI and
OCR engines, database, vector-store, and UI integrations.

## Code boundaries

The implemented package currently includes `app.core`, focused paper domain
models, PDF ingestion, provider-independent OCR contracts, and the OCR benchmark
boundary. Future modules should be created only when their phase requires them.
Keep configuration, logging, and shared exceptions small; introduce
domain-specific exceptions with the domain feature that needs them.

Validate the ignored private OCR benchmark manifest without running OCR:

```bash
uv run python -m scripts.benchmark_ocr validate
```

## Configuration and logging

Settings live in `app/core/config.py` and are loaded from environment variables
or a local `.env`. Never hard-code credentials. Structured logging lives in
`app/core/logging.py` and uses the standard library; production code should use
`logging.getLogger(__name__)` instead of `print()`.

Logs must not include sensitive student data unless explicitly required and
protected. Prefer identifiers such as internal run IDs over names or admission
numbers.

## Tests and quality

Tests are organized beneath `tests/`, with fast unit tests under `tests/unit/`.
Run before completing a change:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy app
```

Tests should verify behavior and failure modes rather than increase counts
without value. Unit tests must remain deterministic and must not require future
external services.

## Data safety

`data/raw/` contains immutable, sensitive historical examination scripts and is
ignored by Git. Generated outputs must use ignored processed, evaluation,
runtime, or Chroma directories. Never overwrite source documents or expose
student-identifying filenames in logs and fixtures.

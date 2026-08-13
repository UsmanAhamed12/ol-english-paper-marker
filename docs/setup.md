# Local setup

## Prerequisites

- macOS on Apple Silicon is the primary development environment.
- Install `uv` for Python and dependency management.
- Poppler commands (`pdfinfo`, `pdftotext`, and `pdfimages`) are needed only to
  reproduce the Phase 0 structural dataset inventory.

PyMuPDF is installed by `uv sync` for PDF validation and rendering. Phase 3's
OCR architecture and Phase 4A's benchmark framework add no dependencies and
require no OCR engine. No Ollama, database, vector store, OCR engine, or UI
service is required.

## Python 3.12 environment

The repository pins `3.12` in `.python-version`, and `pyproject.toml` restricts
the supported runtime to Python 3.12. Create or update the environment with:

```bash
uv sync
uv run python --version
```

The second command must display Python 3.12.x. uv manages the local `.venv`,
which is ignored by Git.

## Environment configuration

Application settings are read from environment variables and, when present, a
local `.env` file. Start from the documented template if overrides are needed:

```bash
cp .env.example .env
```

The `.env` file is ignored and must never be committed. `DATABASE_URL` has no
default because credentials must be supplied explicitly when database support
is introduced. Other integration variables are placeholders for later phases;
their services do not exist yet.

PDF ingestion additionally supports `MAX_PDF_SIZE_MB`, `MAX_PDF_PAGES`,
`PDF_RENDER_DPI`, and `RUNTIME_DATA_DIR`. See `docs/ingestion.md` for defaults
and constraints.

## Verification

Run the complete project checks:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy app
```

Raw historical papers under `data/raw/` are intentionally ignored and must not
be moved, modified, or staged.

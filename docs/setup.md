# Local setup

## Prerequisites

- macOS on Apple Silicon is the primary development environment.
- Install `uv` for Python and dependency management.
- Poppler commands (`pdfinfo`, `pdftotext`, and `pdfimages`) are needed only to
  reproduce the Phase 0 structural dataset inventory.

PyMuPDF is installed by `uv sync` for PDF validation and rendering. The
Tesseract baseline uses `pytesseract` and requires a local executable with
English trained data:

```bash
brew install tesseract
tesseract --version
tesseract --list-langs
uv sync
```

`opencv-python-headless` is installed by `uv sync` for the Phase 4C.2
derived-image experiment. It does not require GUI frameworks. No database,
vector store, alternate Poppler rendering, or UI service is required.

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

Classical OCR supports `TESSERACT_LANGUAGE` (default `eng`), `TESSERACT_PSM`
(default `6`), and `TESSERACT_TIMEOUT_SECONDS` (default `120`).

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

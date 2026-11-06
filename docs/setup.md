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
alternate Poppler rendering, or UI service is required for the OCR baseline.
The local RAG index additionally requires Ollama and its configured embedding
model. See [rag.md](rag.md) to index the syllabus. PostgreSQL is required only
when running the compiled LangGraph with persistent checkpoints; configure
`DATABASE_URL` with a private PostgreSQL connection string for that workflow.
See [langgraph.md](langgraph.md) for checkpoint setup and privacy requirements.

## Python 3.12 environment

The repository pins `3.12` in `.python-version`, and `pyproject.toml` restricts
the supported runtime to Python 3.12. Create or update the environment with:

```bash
uv sync
uv run python --version
```

The second command must display Python 3.12.x. uv manages the local `.venv`,
which is ignored by Git.

## Windows launch when Application Control blocks `.venv`

On Windows, if uv reports that Application Control blocked
`.venv\Scripts\python.exe`, uv cannot run commands through that environment.
When the dependencies have been installed, launch Streamlit with the project
launcher instead:

```powershell
.\run_app.ps1
```

The launcher finds the installed Python 3.12 path reported by `py -0p`, loads
dependencies from `.venv\Lib\site-packages`, and starts Streamlit as a Python
module. It does not execute the blocked virtual-environment interpreter or
`streamlit.exe`. The Python Launcher and Python 3.12 must be available, and
the project dependencies must already be installed.

The application does not start Ollama. If Ollama's API is already responding
on port 11434, leave that instance running; starting another `ollama serve`
will report that the address is already in use.

## Tesseract executable

The OCR pipeline requires the local Tesseract executable and English language
data. If Tesseract is installed in its standard Windows location, the app
discovers it automatically even when its directory is not on `PATH`. For a
custom installation, set `TESSERACT_EXECUTABLE` in `.env` to the full path of
`tesseract.exe`, for example:

```dotenv
TESSERACT_EXECUTABLE='C:\Program Files\Tesseract-OCR\tesseract.exe'
```

Verify the installation and English data with:

```powershell
& 'C:\Program Files\Tesseract-OCR\tesseract.exe' --version
& 'C:\Program Files\Tesseract-OCR\tesseract.exe' --list-langs
```

## Environment configuration

Application settings are read from environment variables and, when present, a
local `.env` file. Start from the documented template if overrides are needed:

```bash
cp .env.example .env
```

The `.env` file is ignored and must never be committed. `DATABASE_URL` has no
default because credentials must be supplied explicitly. It is used by
LangGraph checkpoint persistence and must be a PostgreSQL URL. Other integration
variables should only be configured for services you intend to use.

The primary Groq vision marking path requires `GROQ_API_KEY`. Set it only in
your local `.env` file, not in `.env.example`, source code, or chat. The default
`GROQ_MODEL` is `meta-llama/llama-4-scout-17b-16e-instruct`; change it only to a
vision-capable model enabled for your Groq account. Groq mode sends every page
of both uploaded PDFs to Groq. The Streamlit page requires per-run consent.
Selecting the local marking pipeline keeps processing on this machine.
`GROQ_TIMEOUT_SECONDS` defaults to 300 seconds per request.

PDF ingestion additionally supports `MAX_PDF_SIZE_MB`, `MAX_PDF_PAGES`,
`PDF_RENDER_DPI` (default 300), and `RUNTIME_DATA_DIR`. See `docs/ingestion.md`
for defaults and constraints.

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

# O/L English Automated Paper Marking System

A local-first, AI-assisted system planned to mark scanned Sri Lankan G.C.E.
O/L English answer scripts with correctness, traceability, and objective
evaluation as primary engineering goals.

## Current status

**Phase 2 - PDF ingestion.**

The repository currently provides:

- a Python 3.12 project managed by uv;
- typed environment configuration using Pydantic Settings;
- lightweight structured logging using the Python standard library;
- a small application exception hierarchy;
- pytest, Ruff, mypy, and coverage configuration;
- a read-only Phase 0 dataset inventory tool and documented findings;
- parser-backed PDF validation and safe internal paper identifiers;
- immutable paper/page metadata and deterministic PNG page rendering.

OCR, question segmentation, marking-scheme ingestion, retrieval, grading,
LangChain, LangGraph, PostgreSQL, Chroma, Ollama integration, and Streamlit are
planned but are not implemented.

## Problem statement

The intended system will accept a scanned, unmarked student answer script and
an official marking scheme, then produce traceable question-level results with
human review when correctness cannot be guaranteed. Historical teacher-marked
papers will be used as protected evaluation ground truth, never as model input
during an accuracy evaluation.

## Target architecture

The planned dependency flow is:

```text
Streamlit presentation layer
        |
Application services
        |
Deterministic LangGraph workflow
        |
Ingestion -> OCR -> segmentation -> retrieval -> grading -> aggregation
        |
PostgreSQL system of record + Chroma vector retrieval
```

This is a target architecture, not a description of currently implemented
features. Business logic will remain independent of UI and AI frameworks where
practical.

## Technology direction

- Python 3.12 and uv
- Pydantic v2 and pydantic-settings
- PyMuPDF for PDF validation and page rendering
- pytest, Ruff, and mypy
- planned: LangChain, LangGraph, Ollama/Llama 2, ChromaDB, PostgreSQL, and
  Streamlit in their designated phases

## Local-first and privacy

The project is designed to run locally without paid APIs. Raw examination
scripts contain sensitive educational data and possible personal identifiers.
`data/raw/` is ignored by Git and must remain local, immutable, and access
controlled. Do not log or publish student names, admission numbers, handwriting,
or teacher marks.

## Dataset summary

Phase 0 identified 40 image-only PDFs containing 393 pages (approximately
509.2 MiB). Each page is a full-page JPEG scan with no extractable text layer.
See [docs/data.md](docs/data.md) for the complete inspection report.

## Developer setup

Install [uv](https://docs.astral.sh/uv/), then from the repository root run:

```bash
uv sync
uv run python --version
```

The version command must report Python 3.12.x. Copy `.env.example` to `.env`
only when local overrides are required; never commit `.env`.

## Tests and quality checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy app
```

Further setup and workflow guidance is in [docs/setup.md](docs/setup.md) and
[docs/development.md](docs/development.md). The implemented ingestion boundary
is documented in [docs/ingestion.md](docs/ingestion.md).

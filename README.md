# O/L English Automated Paper Marking System

A local-first, AI-assisted system planned to mark scanned Sri Lankan G.C.E.
O/L English answer scripts with correctness, traceability, and objective
evaluation as primary engineering goals.

## Current status

**Phase 4C.5C - teacher-focused evidence candidates prepared; human labeling required.**

The repository currently provides:

- a Python 3.12 project managed by uv;
- typed environment configuration using Pydantic Settings;
- lightweight structured logging using the Python standard library;
- a small application exception hierarchy;
- pytest, Ruff, mypy, and coverage configuration;
- a read-only Phase 0 dataset inventory tool and documented findings;
- parser-backed PDF validation and safe internal paper identifiers;
- immutable paper/page metadata and deterministic PNG page rendering;
- provider-independent OCR contracts, conservative normalization, provenance,
  and service-level error handling;
- private benchmark manifest contracts, optional page regions, deterministic
  CER/WER metrics, failure-preserving results, and aggregate reporting.
- local Tesseract OCR with typed word boxes, hierarchy metadata, confidence,
  and deterministic layout reconstruction.
- typed, geometry-preserving OpenCV preprocessing experiments with immutable
  canonical images and private derived artifacts.
- deterministic multi-signal Test-marker detection, ordered cross-page Test
  regions, explicit missing-marker evidence, private structure metrics, and
  immutable visual-debug overlays.
- immutable evidence and answer-region models, multi-signal raster/layout
  feature extraction, conservative four-class candidate attribution, and
  private labeling overlays.

Plain Tesseract has been measured as a fast layout/printed-text baseline; it is
not selected as a handwriting solution. Four fixed preprocessing variants were
measured and none beat the official baseline, so preprocessing remains disabled.
The structure layer detected 44/48 manually verified headings with zero false
positives on three private papers. Phase 4C.4 has prepared an 18-sample private
benchmark and a loopback-only visual labeling tool for human verification of
evidence and answer-region candidates. The frozen baseline achieved 1.0000
student precision but only 0.1176 recall and 0.0402 answer-region F1 at IoU 0.50.
Because the human labels contain no PRINTED or TEACHER_CANDIDATE support, the
decision is insufficient label coverage and the separator is not accepted for
downstream extraction. A later integrity check found that the complete original
`a700...` annotation file was no longer recoverable after semantic label drift.
Phase 4C.4R now preserves every changed annotation version and requires an
explicit new 18/18 visual re-verification session. The completed replacement
baseline is recoverable from a complete private snapshot with fingerprint
`41d2364c...ab35`; it does not reconstruct the lost original dataset.
Phase 4C.5B froze all 48 new human annotations with fingerprint
`b28eb7ce...2614b` and measured the unchanged separator. Overall accuracy is
0.3125, student precision/recall/F1 are 0.3750/0.3750/0.3750, and answer-region
F1 is 0.0376 at IoU 0.50. Five of nine verified-empty samples were correctly
empty. Only one human teacher sample exists, so teacher-contamination safety is
unvalidated and a teacher-focused benchmark expansion is required before
separator redesign can be accepted. These differences from Phase 4C.4R reflect
benchmark composition, not an algorithm change.
Phase 4C.5C has prepared a separate 48-crop teacher-risk pool from 12 safe
paper aliases. Its discovery categories are sampling hints only: all samples
remain pending visual review, and no separator rule or benchmark metric has
changed. See `docs/teacher-evidence-benchmark.md`.
Marking-scheme ingestion,
retrieval, grading, LangChain,
LangGraph, PostgreSQL, Chroma, and Streamlit are planned for later phases.

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
is documented in [docs/ingestion.md](docs/ingestion.md), and the OCR contracts
are documented in [docs/ocr.md](docs/ocr.md). Benchmark policy and metrics are
documented in [docs/ocr-benchmark.md](docs/ocr-benchmark.md), and the classical
baseline is documented in [docs/tesseract-ocr.md](docs/tesseract-ocr.md).
The measured preprocessing experiment is documented in
[docs/ocr-preprocessing.md](docs/ocr-preprocessing.md).
The deterministic Test-region layer and its private benchmark are documented in
[docs/exam-structure.md](docs/exam-structure.md).
The evidence-separation contracts, private labeling checkpoint, and current
limitations are documented in
[docs/evidence-separation.md](docs/evidence-separation.md).
Phase 4C.5C adds a separate 48-crop, 12-paper teacher-risk candidate pool to
address the remaining teacher-class coverage gap. Candidate categories are
sampling hints only; all samples are pending visual review, and the separator
has not been tuned or evaluated on this dataset. See
[`docs/teacher-evidence-benchmark.md`](docs/teacher-evidence-benchmark.md).

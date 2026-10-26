# O/L English Automated Paper Marking System

A local-first, AI-assisted system planned to mark scanned Sri Lankan G.C.E.
O/L English answer scripts with correctness, traceability, and objective
evaluation as primary engineering goals.

## Current status

**Phase 7 - local Streamlit FYP demonstration application implemented.**

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
- an explicit parallel Evidence Separator V2 experiment with grouped evidence,
  typed explainable scores, separate answer-space semantics, configuration
  hashes, immutable private experiment records, and visual QA overlays.
- immutable Test-owned student-answer and paper contracts, deterministic
  reading order, region-level OCR provenance, explicit empty/review/failure
  states, and private deterministic JSON serialization.
- a separate typed marking-scheme boundary with conservative Test/criterion
  parsing, explicit UNKNOWN/warning states, source provenance, and exact-number
  alignment to structured student answers.
- a deterministic Decimal-safe marking engine for exact answers, accepted
  variants, token/phrase keywords, and explicitly allocated multi-part
  criteria, with confirmed and unresolved marks kept separate.
- a replaceable local subjective-marking provider, rubric-first versioned
  prompt, strict response/ownership/mark-cap validation, and combined objective
  plus subjective paper results with mandatory review fallbacks.
- one injectable `PaperMarkingService` that coordinates both PDFs through
  extraction, scheme ingestion, exact alignment, objective and subjective
  marking, aggregation, review-queue creation, and a safe final result.
- a thin local Streamlit application with private PDF materialization, safe
  confirmed/final score semantics, Test and criterion breakdowns, feedback,
  review queue, deterministic JSON download, and an explicitly synthetic demo
  mode for presentation QA.
- provenance-tagged syllabus extraction and chunking, Ollama embeddings,
  persistent Chroma storage, and metadata-filtered retrieval; see
  [docs/rag.md](docs/rag.md).

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
Phase 4C.5D froze the 48-crop teacher-focused benchmark with fingerprint
`405f8c0d...b7d9`. Human support is 11 printed, 4 student, 13 teacher, and 20
unknown. The unchanged separator classified 3/13 teacher samples as student
and produced student-answer regions on 5/13 teacher samples, so it is not safe
for downstream extraction. The combined frozen benchmarks now provide enough
coverage for a measured Phase 4C.6 separator redesign; this is a data-readiness
decision, not acceptance of the current model. See
`docs/teacher-evidence-benchmark.md`.
Phase 4C.6 evaluated four predetermined V2 configurations across all three
frozen evidence benchmarks. None passed the hard teacher-safety gate: the best
diagnostic candidate classified 10/13 teacher samples as student, compared with
3/13 under V1, and emitted student regions on 12/13 teacher samples. V2 is not
selected; its failure motivated the deliberately smaller post-V1 safety
boundary used for the prototype. See
`docs/evidence-separator-v2.md`.
Phase 4C.6R keeps V1 and adds a thin deterministic safety filter. On teacher-v1,
teacher-to-student contamination fell from 3/13 to 2/13, teacher samples with
student regions fell from 5/13 to 4/13, and correct empty behavior improved from
16/20 to 18/20. Across all benchmarks, predicted regions fell from 539 to 264.
The filter is accepted for the FYP prototype with explicit review flags; it is
not production-grade authorship classification. See
`docs/evidence-safety-filter.md`.
Phase 4D now converts the accepted safe evidence into one ordered answer object
per detected Test. Only safe regions enter OCR; quarantined regions remain
separate, missing Tests are never invented, and failures are isolated. Because
the historical corpus contains teacher marks, its private smoke outputs are
forced to `REVIEW_REQUIRED` even when OCR succeeds. See
`docs/student-answer-extraction.md`.
Syllabus indexing and a Chroma retrieval boundary are implemented, but
retrieval is not yet connected to grading prompts. LangGraph and PostgreSQL
remain planned for later phases. The orchestration, subjective, and Streamlit
paths are implemented for an FYP prototype; real marking accuracy has not yet
been evaluated and no official marking-scheme PDF is currently available for a
real end-to-end smoke.

## FYP demo quick start

1. Install locked dependencies with `uv sync`.
2. Start local Ollama and verify the model configured by
   `OLLAMA_GRADING_MODEL` is installed.
3. Verify Tesseract is installed with `tesseract --version`.
4. Launch the UI:

   ```bash
   uv run streamlit run streamlit_app.py
   ```

5. Upload one student answer PDF and its corresponding official marking scheme.
6. Select **Mark Paper** and wait for local processing.
7. Inspect confirmed/final score semantics, Test breakdown, feedback, warnings,
   and every review-required item.

With no official scheme available, presentation-only synthetic states can be
enabled explicitly with `UI_DEMO_ENABLED=true`. They are visibly labeled and
must not be presented as real examination marking. See
[docs/streamlit-application.md](docs/streamlit-application.md).

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
  persistence integrations in their designated phases

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
The controlled Phase 4C.6 V2 architecture, results, and rejection gate are
documented in
[docs/evidence-separator-v2.md](docs/evidence-separator-v2.md).
The final prototype safety policy and its frozen evaluation are documented in
[docs/evidence-safety-filter.md](docs/evidence-safety-filter.md).
The Test-owned answer extraction contract is documented in
[docs/student-answer-extraction.md](docs/student-answer-extraction.md).
The structured marking-scheme and alignment boundary is documented in
[docs/marking-scheme-ingestion.md](docs/marking-scheme-ingestion.md).
The supported deterministic rules, score semantics, and fallbacks are in
[docs/deterministic-marking-engine.md](docs/deterministic-marking-engine.md).
The frozen 48-crop, 12-paper teacher benchmark and unchanged V1 measurements
are documented in
[`docs/teacher-evidence-benchmark.md`](docs/teacher-evidence-benchmark.md).

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
phase that needs it. The current dependency set intentionally excludes AI
frameworks, database, vector-store, and UI integrations. OpenCV headless
supports the measured OCR preprocessing experiment only.

## Code boundaries

The implemented package currently includes `app.core`, focused paper domain
models, PDF ingestion, provider-independent OCR contracts, OCR benchmarking,
deterministic exam-structure detection, and a conservative evidence-separation
boundary awaiting human benchmark labels. Future modules should be created only
when their phase requires them.
Keep configuration, logging, and shared exceptions small; introduce
domain-specific exceptions with the domain feature that needs them. The
Tesseract adapter consumes canonical `PaperPage.image_path` values; do not add
an alternative PDF rendering stack.

Validate the ignored private OCR benchmark manifest without running OCR:

```bash
uv run python -m scripts.benchmark_ocr validate
```

Run the private classical baseline only after validation:

```bash
uv run python -m scripts.benchmark_ocr run --provider tesseract --smoke
uv run python -m scripts.benchmark_ocr run --provider tesseract
uv run python -m scripts.benchmark_ocr run --provider tesseract \
  --preprocessing grayscale-denoise-threshold
```

The third command writes only ignored derived/result artifacts. The official
baseline remains `--preprocessing none`; no experimental variant is selected.

Validate or run the separate private structure benchmark with:

```bash
uv run python -m scripts.evaluate_structure validate
uv run python -m scripts.evaluate_structure run
```

This command emits safe aggregate metadata only. Private OCR evidence, source
references, structure results, and overlays remain under ignored
`data/evaluation/structure/` storage.

Prepare or safely validate the private evidence-labeling benchmark with:

```bash
uv run python -m scripts.prepare_evidence_benchmark prepare
uv run python -m scripts.prepare_evidence_benchmark validate
```

The preparation command creates ignored crops, overlays, candidate predictions,
and a private worksheet. It never supplies classifier predictions as human
labels and does not overwrite an existing worksheet.

Launch the loopback-only visual labeling tool with:

```bash
uv run python -m scripts.annotate_evidence
```

Open `http://127.0.0.1:8765/` and press `Ctrl+C` to stop the server. Labels are
saved only to ignored `data/evaluation/evidence/annotations.json`. The tool does
not infer labels, convert transcription strings into boxes, or run benchmark
metrics.

For the Phase 4C.4R human re-verification session, use:

```bash
uv run python -m scripts.annotate_evidence --reverify
```

Re-verification starts at zero even when older annotations carry the original
`human_verified` flag. The interface displays the current saved class and
rectangles as review evidence but does not preselect a class. Each sample
requires the dedicated re-verified/save action. Any semantic change first
creates a timestamp-and-fingerprint backup of the previous valid
`annotations.json` beneath ignored `data/evaluation/evidence/backups/` storage.
Do not run the evaluator until the current session reports 18/18.

After 18/18 persisted approvals validate, the one-time finalizer creates a
complete non-overwriting private snapshot, independently reloads it, and writes
the unchanged detector measurement beneath a separate `phase4c4r` result
namespace:

```bash
uv run python -m scripts.finalize_evidence_reverification
```

The completed Phase 4C.4R baseline has already been finalized. The command
refuses to overwrite its snapshot or result and is documented for auditability,
not routine reruns.

The separate Phase 4C.5A evidence-v2 candidate pool is already prepared. Safely
validate it without evaluating the separator:

```bash
uv run python -m scripts.prepare_evidence_expansion validate
```

Launch its loopback-only labeler with:

```bash
uv run python -m scripts.annotate_evidence --dataset evidence-v2
```

All v2 records began pending, no discovery category was preselected, and no
candidate rectangle became human ground truth automatically. The completed
48-sample dataset is now frozen and measured. Its one-time finalizer validates
historical assets, writes a non-overwriting complete private snapshot,
independently recovers it, and measures unchanged detector behavior:

```bash
uv run python -m scripts.finalize_evidence_expansion
```

The completed command refuses to overwrite its snapshot or result and is kept
for auditability. Do not tune the separator against these labels in Phase
4C.5B. Follow the finalized policy and results in
`docs/evidence-benchmark-v2.md`.

After all annotations validate, reproduce the frozen untuned baseline with:

```bash
uv run python -m scripts.evaluate_evidence
```

This writes only ignored provenance, predictions, metrics, and comparison
overlays. The current measured decision is insufficient human class coverage;
do not treat evidence classifications as production-ready answer extraction.

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
Teacher-focused Phase 4C.5C candidates use a separate pending namespace:

```bash
uv run python -m scripts.prepare_teacher_evidence validate
uv run python -m scripts.annotate_evidence --dataset evidence-teacher-v1
```

The latter binds only to `127.0.0.1`. Candidate discovery categories are not
ground truth; do not evaluate or tune the separator until all samples have been
visually labeled and a later phase explicitly authorizes freezing.

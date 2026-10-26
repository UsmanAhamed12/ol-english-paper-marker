# End-to-end paper marking

## Status and scope

Phase 6 connects the accepted FYP components through one application service.
It does not add a UI, database, LangChain, LangGraph, RAG, cloud service, or new
OCR/evidence logic. The existing services remain authoritative for their own
business rules.

## Architecture

```text
Student PDF
  -> parser-backed validation / PyMuPDF rendering
  -> Tesseract structured page evidence
  -> ExamStructureDetector
  -> EvidenceSeparator V1 + StudentCandidateSafetyFilter
  -> StudentExtractionService
  -> StructuredStudentPaper

Marking-scheme PDF
  -> parser-backed validation / PyMuPDF rendering
  -> Tesseract OCR
  -> MarkingSchemeIngestionService
  -> StructuredMarkingScheme

StructuredStudentPaper + StructuredMarkingScheme
  -> exact TestAlignmentService
  -> Phase 5B deterministic objective marking
  -> Phase 5C guarded local subjective marking
  -> CombinedMarkingService
  -> FinalPaperMarkingResult
```

`PaperMarkingService.mark_paper(PaperMarkingRequest(...))` is the single caller
boundary. Production construction lives in `app.orchestration.factory`; unit
tests inject fake input, student, scheme, and subjective-provider boundaries.

## Request and result

`PaperMarkingRequest` contains two PDF paths, a content-independent UUID run ID,
an optional private output path, and the orchestration version. It contains no
student name or identity metadata.

`FinalPaperMarkingResult` directly exposes document/scheme IDs, detected,
missing, and aligned Tests, objective/subjective/total confirmed marks, known
maximum, review and unsupported capacity, Test/criterion results, feedback,
review queue, stage summaries, timings, errors, warnings, and provenance.

Statuses mean:

- `COMPLETED`: all required mark-bearing criteria resolved; `final_score` is
  available.
- `COMPLETED_WITH_REVIEW`: processing completed but review or unsupported marks
  remain.
- `PARTIALLY_COMPLETED`: useful marking exists but one or more Tests are
  missing, OCR-failed, or structurally incomplete.
- `FAILED`: an unrecoverable prerequisite or core integration stage failed and
  no safe marking output was produced.

Review is not a system failure. Unresolved marks are never silently converted
to zero to manufacture a final score.

## Stage and failure policy

The result measures input validation, student PDF processing, structure
detection, answer extraction, scheme ingestion, alignment, objective marking,
subjective marking, and aggregation. Each stage has a safe status, duration,
warning codes, and optional stable error code. Structured logs contain only run
ID, stage, duration, status, and safe error code.

Invalid documents and unusable core artifacts fail fast. Missing Tests,
per-region OCR failure, review-required evidence, unknown criteria, subjective
provider failure, and malformed individual model responses remain visible and
allow safe work on other criteria to continue.

## Mark ownership and accounting

Every criterion has exactly one route:

- objective types -> Phase 5B;
- subjective types -> Phase 5C;
- `UNKNOWN` -> unsupported/review.

Pydantic validation rejects duplicate criterion ownership. Decimal invariants
require objective plus subjective confirmed marks to equal the total, confirmed
marks not to exceed the known maximum, and unresolved marks to equal review
plus unsupported capacity. A final score exists only with zero unresolved
marks.

## Feedback and review queue

Criterion feedback from Phase 5C is aggregated deterministically; no extra LLM
summary call is made. Each `ReviewItem` identifies the Test, optional criterion,
safe reason/status, possible marks affected, and a safe source reference. This
is ready for the Phase 7 presentation layer.

## CLI

```bash
uv run python -m scripts.mark_paper \
  --student /path/to/student.pdf \
  --scheme /path/to/official-scheme.pdf \
  --output data/runtime/paper-marking/result.json
```

The CLI delegates all work to `PaperMarkingService` and prints aggregate status
metadata only. Complete JSON is written only when requested and only beneath
configured ignored runtime storage.

## Streamlit caller

Phase 7 adds a thin caller at `app/ui/streamlit_app.py`. It materializes the two
uploads under UUID-named ignored runtime storage, invokes this same service, and
maps `FinalPaperMarkingResult` into presentation-specific immutable models. It
does not duplicate validation, extraction, alignment, marking, accounting, or
feedback logic. See `docs/streamlit-application.md`.

## Verification and real-data limitation

The synthetic end-to-end scenario exercises exact, accepted-variant,
multi-part, content, holistic, empty, review-required, and unknown criteria.
It confirms objective/subjective ownership, review propagation, feedback,
timing, provenance, serialization, and failure isolation without Ollama,
Tesseract, or private PDFs.

No official marking-scheme PDF is currently present in private repository
storage. Therefore a real student-plus-scheme end-to-end smoke was not
fabricated. This is a real-data availability limitation, not a core integration
failure.

## Privacy and limitations

Raw paths, OCR text, scheme text, model responses, images, and stack traces are
not included in logs or CLI summaries. Full results may contain private student
evidence and therefore remain under ignored runtime storage.

Subjective marking accuracy remains unmeasured. OCR/evidence limitations and
review fallbacks remain. Sequential local inference may be slow on the M1. The
next accepted phase is the Streamlit FYP application, not another pipeline
redesign.

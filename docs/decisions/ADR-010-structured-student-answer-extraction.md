# ADR-010: Use a typed Test-owned student-answer extraction boundary

## Status

Accepted for the FYP prototype

## Context

Phase 4C.6R accepted V1 plus a safety filter as the final evidence boundary.
Later marking needs one stable, auditable answer per detected Test, including
multi-page provenance, uncertainty, empty answers, and isolated OCR failures.
Historical evaluation scans also contain flattened teacher annotations.

## Decision

Introduce `StudentExtractionService`, immutable Test-owned answer models, and
deterministic JSON serialization. OCR only safe regions through the existing
`OCRService`; preserve review evidence separately; never invent missing Tests;
and force successful historical marked-paper extraction to review-required.

## Alternatives considered

- Pass unstructured OCR text directly to grading.
- OCR whole Test regions and discard region provenance.
- Treat uncertain or missing evidence as empty text.
- Reopen evidence separation before continuing the FYP.

## Consequences

Marking receives a stable contract with explicit provenance and failure states,
and one failed region does not stop the paper. Runtime artifacts remain private.
The approach is conservative: some usable answers require human review, and
handwriting OCR and residual teacher contamination remain limitations.

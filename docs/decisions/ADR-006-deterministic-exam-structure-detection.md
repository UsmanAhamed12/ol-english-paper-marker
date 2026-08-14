# ADR-006: Deterministic exam structure detection

## Status

Accepted

## Context

Plain Tesseract recovers useful printed words and geometry, but exact `Test NN`
matching over reconstructed page text missed headings on a real paper. Scans may
contain OCR confusions, multiple Tests per page, continuation pages, student
writing, and teacher annotations. Missing or false Test boundaries would
corrupt later answer extraction.

## Decision

Detect Test-marker candidates from typed Tesseract word evidence using bounded
text normalization, explicit OCR-confusion rules, hierarchy and geometry, and
an explainable confidence score. Select a deterministic increasing document
sequence, retain rejected/duplicate evidence, report missing Tests explicitly,
and create coarse cross-page regions between accepted markers. Keep private
manual structure ground truth and visual overlays separate from OCR
transcription ground truth.

## Alternatives considered

- Apply one regular expression to flattened page text.
- Assume one page contains one Test.
- Use an LLM or a large fuzzy/NLP framework to infer structure.
- Invent missing headings from expected numeric progression.
- Combine structure detection with handwriting or teacher-mark classification.

## Consequences

The approach is local, deterministic, testable, geometry-preserving, and
auditable. Multiple Tests per page and cross-page regions are supported without
new dependencies. The private benchmark measured perfect precision and 0.9167
recall, so uncertainty remains explicit. Rules are tied to the known Test 1-16
format, hierarchy errors may still cause misses, and coarse regions require
later refinement before answer extraction.

# ADR-007: Conservative multi-signal evidence separation

## Status

Proposed pending human benchmark verification

## Context

Historical marked papers flatten printed questions, student handwriting, and
teacher annotations into one scan. Tesseract provides useful text geometry but
its confidence cannot identify authorship. Ink color is also ambiguous: print,
students, and teachers may use overlapping colors. Incorrectly assigning
printed or teacher evidence to a student would contaminate later answer OCR.

## Decision

Introduce a deterministic evidence-separation boundary after Test segmentation.
Measure typed raster and OCR-geometry features, require multiple independent
signals for `PRINTED`, `STUDENT_CANDIDATE`, or `TEACHER_CANDIDATE`, and preserve
conflicting or insufficient evidence as `UNKNOWN`. Detect answer-space
candidates separately from repeated writing guides and conservative student
evidence clusters. Preserve all geometry and provenance, and validate the rules
against a new private, human-labeled benchmark before treating them as adopted.

## Alternatives considered

- Treat low OCR confidence as handwriting.
- Treat saturated or colored ink as student writing.
- Delete printed or suspected teacher evidence before retaining provenance.
- Use an OCR transcription or an LLM to infer authorship.
- Depend immediately on a clean examination template that is not available.
- Skip the human benchmark and evaluate against classifier-generated labels.

## Consequences

The boundary is local, deterministic, explainable, geometry-preserving, and
testable with synthetic images. `UNKNOWN` reduces false attribution, and answer
spaces can be represented even when they are blank. The approach adds explicit
feature thresholds and private derived artifacts. Heuristics may miss black or
faint handwriting and may confuse colored annotations, so human benchmark
results are required before the strategy can be accepted. Future clean-template
alignment can improve printed-content separation without replacing these domain
contracts.

# ADR-007: Conservative multi-signal evidence separation

## Status

Proposed; Phase 4C.4R baseline is trustworthy but not accepted for downstream
extraction because class coverage and localization remain insufficient

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
Collect those labels through a loopback-only visual tool that records explicit
classes and crop-relative rectangles independently from free-form worksheets;
never derive labels or geometry from transcription text or classifier output.

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

The first measured baseline produced 17 human `STUDENT_CANDIDATE` labels, one
`UNKNOWN`, and no human `PRINTED` or `TEACHER_CANDIDATE` labels. It therefore
cannot validate printed/teacher separation or teacher-contamination safety.
Student precision was 1.0000 but recall was 0.1176, and answer localization F1
was 0.0402 at IoU 0.50. The decision gate is `INSUFFICIENT HUMAN LABEL
COVERAGE`; the current separator is not accepted for downstream extraction and
no post-label threshold tuning is authorized by this ADR.

After the original `a700...` annotation fingerprint was recorded, a later
integrity check detected semantic drift and the complete original annotation
store could not be recovered. Phase 4C.4R therefore required all 18 samples to
be visually re-verified without using historical labels as anchors. It produced
the distinct replacement fingerprint `41d2364c...ab35`. Complete private frozen
snapshots, independent reload verification, and semantic backup-before-change
behavior are now required. This replacement is not represented as a
reconstruction of the original dataset. The rerun changed human support to 3
printed, 14 student candidates, 0 teacher candidates, and 1 unknown; detector
thresholds and localization metrics were unchanged.

Phase 4C.5A therefore changes the benchmark sampling unit, not the separator:
48 new smaller regions across 12 safe aliases are sampled using deterministic
local evidence signals. Discovery categories remain private sampling metadata
and never become labels. A human must assign every class and student-answer box
under a documented contiguous-region policy before any redesign measurement.

Phase 4C.5B froze those 48 labels under semantic fingerprint
`b28eb7ce...2614b` and measured the unchanged rules. Human support became 30
printed, 8 student, 1 teacher, and 9 unknown. Overall accuracy is 0.3125,
student precision/recall/F1 are all 0.3750, and answer localization F1 is 0.0376
at IoU 0.50. Five of nine verified-empty samples were correctly empty. Because
one teacher sample cannot validate contamination safety, the decision remains
proposed and requires teacher-focused benchmark expansion before separator
redesign can be accepted. No post-label threshold change was made.

Phase 4C.5C responds by creating a separate teacher-focused candidate pool,
without changing the separator. High-recall local color, component, geometry,
margin, OCR-context, and hard-negative signals select crops but never assign
ground truth. Human visual inspection remains the only source of labels. This
keeps discovery bias explicit and preserves both frozen benchmark histories.

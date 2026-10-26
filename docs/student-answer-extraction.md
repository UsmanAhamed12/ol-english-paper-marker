# Structured student-answer extraction

## Status

Phase 4D implements the deterministic Test-level extraction contract for the
FYP prototype. It consumes the accepted Phase 4C.6R safety boundary; it does
not reopen evidence detection or claim that handwriting OCR is solved.

## Architecture

```text
TestRegion
  -> EvidenceSeparator V1
  -> StudentCandidateSafetyFilter
  -> SafeTestExtraction
  -> StudentExtractionService
       -> derived region crop
       -> OCRService
       -> ordered AnswerFragment records
  -> StructuredStudentAnswer
  -> StructuredStudentPaper
```

The service produces exactly one answer for every detected Test. Missing Test
markers remain explicit paper-level metadata and never create invented answer
objects. A Test may contain regions from several pages.

## Models and ownership

`StructuredStudentAnswer` owns its Test number, ordered pages, combined text,
source regions, raw and normalized OCR fragments, quarantined review evidence,
warnings, confidence signals, duration, and structure/safety/OCR provenance.
Every child region and fragment must match the parent paper, Test, and page.

`StructuredStudentPaper` keeps numerically ordered answers plus summaries of
detected, missing, empty, review-required, OCR-failed, and structurally
incomplete Tests. The models are immutable Pydantic value objects.

## Safety and source context

Only `SAFE_STUDENT` regions are cropped and sent to OCR.
`REVIEW_REQUIRED` evidence is retained with geometry and reason codes but is
never merged into trusted answer text. `REJECTED_NON_STUDENT` evidence is
excluded. No safe or review evidence produces a valid `EMPTY` answer without
calling OCR.

Historical evaluation scans already contain flattened teacher annotations.
Private visual QA confirmed residual mixed evidence despite the accepted
safety filter. The caller must therefore identify source context explicitly.
Any otherwise successful extraction from `HISTORICAL_MARKED_EVALUATION` is
forced to `REVIEW_REQUIRED` and carries a teacher-annotation-risk warning.
Normal unmarked runtime papers retain the ordinary safe/empty policy.

## OCR and crops

The service reuses `OCRProvider` through `OCRService`; it does not introduce a
new engine. Region crops are deterministic PNG files beneath ignored runtime
storage. Before and after every crop, the canonical image SHA-256 is checked.
Canonical renders and raw PDFs are never modified.

Each fragment preserves the provider's raw output, conservative normalized
output, provider/model identity, confidence when available, warnings,
duration, page, and bounding box. A crop or OCR failure creates an explicit
failed fragment and does not stop other Tests.

## Reading order and normalization

Regions are ordered deterministically by page number, top edge, then left edge.
Normalized fragment text is joined with line breaks while the original raw
fragment text remains available. Normalization is limited to the existing OCR
normalizer: Unicode/newline normalization and harmless whitespace cleanup. It
does not correct spelling, grammar, capitalization, or missing words.

## Status policy

- `EXTRACTED`: at least one safe fragment produced text and no review trigger
  applies.
- `EMPTY`: neither safe nor review-required evidence exists.
- `REVIEW_REQUIRED`: quarantined evidence, partial failure, empty OCR output,
  or historical marked-source risk exists.
- `OCR_FAILED`: all attempted region OCR failed.
- `STRUCTURE_INCOMPLETE`: safety evidence is missing for a Test page.

## Serialization and privacy

Canonical JSON uses sorted keys and stable Pydantic JSON serialization.
Runtime writers require an output path beneath an explicitly supplied private
root and use atomic replacement. Student text, crops, and JSON stay under
ignored runtime storage and must never enter logs or documentation.

## Three-paper smoke

The private smoke runs the complete pipeline for the three existing safe
aliases. It reports only counts and durations. The historical marked-source
policy deliberately routes all observed answers to review; this is appropriate
because those scans mix student and teacher ink. Visual checks covered short,
paragraph, multi-region, cross-page, empty/review, and mixed-evidence cases.

## Limitations

- Tesseract remains weak on handwriting.
- The safety filter still permits some print and teacher annotations.
- Region localization is approximate and may combine mixed evidence.
- Missing Test markers cannot be assigned an answer automatically.
- Human review presentation is deferred; the domain state is preserved now.

Phase 5A aligns this contract to a separate typed marking-scheme boundary.
Phase 5B consumes the aligned answer without changing OCR evidence and marks
only exact, variant, keyword, and explicitly allocated multi-part criteria.
Extraction review and failure states remain unresolved rather than becoming
zero marks. Phase 5C now consumes this same status and confidence evidence for
guarded subjective marking; uncertain extraction remains review-required.
Phase 6 delegates to this service through the production student pipeline and
preserves all answer statuses in the final review queue.

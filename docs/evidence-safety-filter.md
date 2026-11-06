# Minimal teacher-safe extraction policy

## Status

Phase 4C.6R accepts `StudentCandidateSafetyFilter` as the final evidence-safety
boundary for the FYP prototype. V1 remains the production evidence separator.
The rejected V2 experiment and all its private results remain available for
audit and dissertation analysis, but V2 is not used downstream.

This acceptance is pragmatic, not a claim that evidence attribution is solved.
Uncertain regions are explicitly quarantined for review, and downstream code
must preserve that state.

## Architecture

```text
EvidenceSeparator V1
    -> StudentCandidateSafetyFilter
       -> SAFE_STUDENT
       -> REVIEW_REQUIRED
       -> REJECTED_NON_STUDENT
    -> conservative StudentAnswerRegion builder
       -> SAFE / REVIEW_REQUIRED / EMPTY Test outcome
```

`AnswerSpace` and `StudentAnswerRegion` remain different concepts. Blank dotted
lines or boxes do not create a student region unless safe student evidence is
present. No OCR, PDF rendering, structure detection, or frozen benchmark data
is modified.

## Fixed policy

The policy was written before its single benchmark run and was not retuned.

- Reject long, thin, regular guide/rule candidates and edge-spanning grid or
  border artifacts.
- Reject dense, high-confidence, regular achromatic OCR evidence that strongly
  resembles print.
- Reject tiny visual-only components outside an existing writing-guide context.
- Quarantine evidence classified as unknown or teacher-originated rather than
  treating uncertain classification as proof that no student response exists.
- Quarantine isolated chromatic candidates, red-dominant candidates, small
  isolated margin/score risks, and overlapping correction-stroke risks.
- Preserve one-character or numeric OCR candidates when they have plausible
  geometry, especially inside an existing answer-space context.
- Build regions only from safe student candidates. Merge nearby candidates on
  the same line or closely aligned paragraph lines, with a 20% Test-area cap.
- Represent quarantined evidence with one Test-page review region and retain
  all applicable safety reasons, avoiding one review item per OCR word.
- Emit `REVIEW_REQUIRED` when any candidate is quarantined; emit `EMPTY` when no
  safe candidate remains.

Color and OCR confidence are supporting signals only. Neither establishes an
author on its own.

## Frozen benchmark results

| Benchmark | Model | Student P/R/F1 | Teacher→student | Printed→student | Correct empty | Regions |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Phase 4C.4R | V1 | 1.0000/0.1429/0.2500 | n/a | 0 | 0/4 | 348 |
| Phase 4C.4R | V1 + filter | 1.0000/0.0714/0.1333 | n/a | 0 | 2/4 | 168 |
| evidence-v2 | V1 | 0.3750/0.3750/0.3750 | 0/1 | 2 | 5/9 | 157 |
| evidence-v2 | V1 + filter | 0.4286/0.3750/0.4000 | 0/1 | 1 | 5/9 | 74 |
| teacher-v1 | V1 | 0.3750/0.7500/0.5000 | 3/13 | 2 | 16/20 | 34 |
| teacher-v1 | V1 + filter | 0.4286/0.7500/0.5455 | 2/13 | 2 | 18/20 | 22 |

Across all 114 samples, V1 student precision/recall/F1 were
0.4444/0.3077/0.3636. The safety-filter values are
0.4667/0.2692/0.3415. The small recall decrease is accepted for the prototype
because precision improved, teacher contamination reached the strong target,
and region count fell from 539 to 264 rather than exploding.

Teacher samples producing student regions fell from 5/13 to 4/13; student boxes
on teacher samples fell from seven to five. Correct empty behavior improved from
21/33 to 25/33 across the three datasets. Printed-to-student dominant errors
fell from four to three.

Localization remains weak and is not the acceptance criterion for this minimal
phase. IoU-0.50 F1 values are 0.0224, 0.0462, and 0.0938; IoU-0.25 F1 values
are 0.1119, 0.1231, and 0.1250 across Phase 4C.4R, evidence-v2, and teacher-v1.

## V2 rejection record

V2 introduced richer color, component, shape, OCR, grouping, ambiguity, and
answer-space signals. Aggregate student recall and some printed metrics looked
better, but those gains were misleading: teacher-to-student contamination rose
from 3/13 to 10/13, teacher samples with student regions rose from 5/13 to
12/13, and region generation became heavily fragmented. V2 was rejected.

The post-V1 policy was chosen because it targets unsafe extraction directly,
keeps V1 reproducible, cuts region volume, improves empty handling, and retains
uncertainty as a review state without another research-scale redesign.

## Visual QA and limitations

Private overlays for printed, student, teacher-risk, review-required, and
verified-empty samples were inspected. Printed-only and empty examples emitted
no student answer region; student examples retained usable regions; teacher
marks were reduced but not eliminated. Geometry remained bounded and canonical
images were unchanged. Visual QA: **PASS for FYP prototype use with review
flags**, not production-grade automatic extraction.

Known limitations:

- two of 13 teacher samples remain classified as student;
- four teacher samples still contain five predicted student regions;
- Phase 4C.4R student recall falls from 0.1429 to 0.0714;
- printed guides can still influence V1 evidence even though they no longer
  create regions by themselves;
- mixed teacher/student ink remains difficult in flattened scans;
- `REVIEW_REQUIRED` needs a later human-review presentation boundary.

Phase 4D now consumes this boundary through typed, Test-owned answer models.
Historical teacher-marked evaluation pages are always surfaced as
`REVIEW_REQUIRED` after OCR because residual attribution risk remains. See
`student-answer-extraction.md`. The next phase is **Phase 5 — Marking Scheme
Ingestion and Test Alignment**.

# Evidence Separator V2 experiment

## Status and purpose

Phase 4C.6 implemented a parallel, deterministic `EvidenceSeparatorV2` and
`StudentAnswerRegionDetectorV2` without changing V1. The goal was to improve
printed, student, teacher, and unknown evidence separation while reducing unsafe
student-answer regions. Four configurations were fixed before evaluation and
run on all three frozen human benchmarks.

No V2 configuration passed the teacher-safety gate. V1 therefore remains the
production evidence separator and is followed by the separately measured
Phase 4C.6R safety boundary. V2 is an experiment, not a production replacement.

## Architecture

```text
canonical PaperPage + frozen V1 evidence geometry
    -> component grouping
    -> EvidenceFeatureExtractorV2
    -> explainable printed/student/teacher scoring
    -> EvidenceSeparatorV2
    -> AnswerSpaceDetectorV2
    -> StudentAnswerRegionDetectorV2
```

The implementation preserves page, Test, source-image, OCR-word, and bounding-
box provenance. It never writes canonical images. `AnswerSpace` is explicitly
separate from `StudentAnswerRegion`: printed lines or blank writing space are
layout opportunities, not proof of student writing.

## Feature and scoring policy

V2 uses typed, deterministic feature families:

- local HSV/chromatic ratios, color consistency, and foreground contrast;
- connected-component density, compactness, fragmentation, and isolation;
- diagonal, crossing-stroke, underline, circle, and correction-stroke proxies;
- OCR confidence, token density, hierarchy consistency, and regularity;
- page/Test-relative position, margin proximity, and answer-space overlap;
- grouped component size and geometry.

No single color or OCR-confidence signal establishes authorship. Each class
receives an explainable score with stable reason codes. Teacher-risk suppression
can prevent a teacher-like group from becoming student evidence. Ambiguous or
weak scores become `UNKNOWN`; the score is an engineering heuristic, not a
calibrated probability.

## Controlled variants

| Variant | Fixed change | Configuration SHA-256 |
| --- | --- | --- |
| V2-A | grouping and printed/student scoring | `aa06a1ec28f402a124736e7b7c1d08107dc7a3f9aa1aa918330745b97bfd5739` |
| V2-B | A plus teacher-risk scoring and suppression | `73940342ae2bf99a12dd01a2bb3de382c0b2184afe6ec11f9b641a3432f9c7db` |
| V2-C | B plus ambiguity margin | `d30df078701c1c965cc30e1807ce829867c47d2540c546d3b18e937b4a9df4b5` |
| V2-D | C plus paragraph merging and stricter answer-region sizing | `5408aadd740834741a2d831473a424465b7ccbe2ec2127655dfc38e7d02a3fb0` |

The full non-overwriting private run is under
`data/evaluation/evidence_v2_experiments/phase4c6/`. Predictions and overlays
remain ignored. No thresholds were changed after observing results.

An earlier private bundle (`20260821T110811.693077Z`) incorrectly recorded
V2-B as selected because the experiment orchestrator ranked candidates without
enforcing the already-defined hard V1 contamination ceiling. It remains
immutable for audit history. The gate implementation was corrected without
changing features, thresholds, or predictions; the authoritative bundle
(`20260821T111320.668571Z`) records no selected variant.

## Results

V2-B ranked highest among the controlled candidates and is used below for
diagnosis. It was **rejected** by the hard safety gate and is not selected.

| Benchmark | Model | Accuracy | Macro F1 | Student P/R/F1 | Teacher P/R/F1 | Printed P/R/F1 | Unknown rate |
| --- | --- | ---: | ---: | --- | --- | --- | ---: |
| Phase 4C.4R | V1 | 0.1667 | 0.1225 | 1.0000/0.1429/0.2500 | undefined | see frozen baseline | 0.8889 |
| Phase 4C.4R | V2-B | 0.6667 | 0.2667 | 0.7500/0.8571/0.8000 | undefined | 0/0/0 | 0.0000 |
| evidence-v2 | V1 | 0.3125 | 0.2469 | 0.3750/0.3750/0.3750 | undefined/0/0 | 1.0000/0.2000/0.3333 | 0.7083 |
| evidence-v2 | V2-B | 0.4375 | 0.2939 | 0.2727/0.7500/0.4000 | undefined/0/0 | 0.8125/0.4333/0.5652 | 0.2083 |
| teacher-v1 | V1 | 0.4792 | 0.4649 | 0.3750/0.7500/0.5000 | 0.3000/0.2308/0.2609 | 1.0000/0.3636/0.5333 | 0.5417 |
| teacher-v1 | V2-B | 0.3542 | 0.3366 | 0.1333/1.0000/0.2353 | 0/0/0 | 0.7000/0.6364/0.6667 | 0.1458 |

Across all 114 samples, V2-B had accuracy 0.4386 and macro F1 0.3410.
Student precision/recall/F1 were 0.3235/0.8462/0.4681; teacher values were
0/0/0; printed values were 0.7143/0.4545/0.5556; and UNKNOWN F1 was 0.3404.

The critical teacher-v1 result regressed from 3/13 teacher samples classified
as student under V1 to 10/13 under V2-B. Student-answer regions appeared on
12/13 teacher samples (138 boxes), versus 5/13 (7 boxes) for V1. This exceeds
the fixed V1 contamination ceiling and disqualifies every V2 variant.

### Localization and empty behavior

| Benchmark | Model | IoU .50 F1 | IoU .25 F1 | Correct empty | Predicted boxes | Mean region/crop area |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Phase 4C.4R | V1 | 0.0402 | 0.1518 | 0/4 | 348 | 0.0242 |
| Phase 4C.4R | V2-B | 0.0046 | 0.0139 | 0/4 | 764 | 0.0099 |
| evidence-v2 | V1 | 0.0376 | 0.0657 | 5/9 | 157 | 0.0338 |
| evidence-v2 | V2-B | 0.0000 | 0.0861 | 2/9 | 246 | 0.0370 |
| teacher-v1 | V1 | 0.0714 | 0.1571 | 16/20 | 34 | 0.1339 |
| teacher-v1 | V2-B | 0.0414 | 0.0769 | 4/20 | 232 | 0.0300 |

V2 produced smaller individual boxes in two datasets, but many more boxes.
Fragmentation, printed-guide detection, and teacher-mark leakage dominated the
localization metrics. Smaller average area therefore did not mean safer or more
accurate extraction.

## Visual QA and safe error analysis

Ten private overlays covered printed-only, short answer, paragraph, teacher
tick/cross, teacher score, correction, mixed, difficult, unknown, and verified-
empty cases. Geometry stayed inside the source crops and canonical images were
unchanged. Visual rendering therefore passed. Suitability failed: printed
writing guides were repeatedly promoted to student regions, teacher marks were
included as student evidence, paragraph evidence fragmented, and verified-empty
pages still produced answer regions.

Measured failures are consistent across the three benchmarks: V2 improves
student recall by aggressively promoting uncertain components, but pays for it
with poor student precision, 10 teacher-to-student errors, no correct teacher
dominant classifications, and weak IoU. The UNKNOWN rate also collapses,
showing that ambiguity is being forced into concrete classes too often.

## Decision

**C. V2 DOES NOT BEAT V1 — KEEP V1.** No V2 configuration is selected, so
there is no selected configuration hash. V2-B's hash is retained only as the
diagnostic configuration shown above. Raw V1 is not safe enough for downstream
extraction by itself.

Phase 4C.6R subsequently chose a smaller post-V1 safety filter rather than
another separator redesign. That fixed policy reduced teacher-to-student errors
to 2/13 and is accepted for the FYP prototype. See
`docs/evidence-safety-filter.md`. V2 remains rejected and preserved.

## Limitations

- Hand-crafted shape proxies are brittle on low-resolution, flattened scans.
- V2 starts from V1 evidence candidates, so missing or over-broad V1 candidates
  constrain the redesign.
- Dominant-class evaluation compresses mixed evidence into one human label.
- No clean-paper template is available for printed-content subtraction.
- Teacher and student ink may share color, stroke, and position.
- No ML classifier, handwriting OCR, grading, or downstream extraction was
  introduced.

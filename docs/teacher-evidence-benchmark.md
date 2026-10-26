# Teacher-focused evidence benchmark

## Status

Phase 4C.5D froze and evaluated the private `evidence-teacher-v1` dataset.
Its authoritative semantic annotation fingerprint is
`405f8c0de7e73cbce13859457d2be28c3d615dc3d0ee2dd734a348b29ebbb7d9`.
All 48 annotations validated and the complete snapshot was independently
reloaded with every rectangle intact. The existing separator and answer-region
detector were measured without changes.

The dataset contains 11 `PRINTED`, 4 `STUDENT_CANDIDATE`, 13
`TEACHER_CANDIDATE`, and 20 `UNKNOWN` samples. This closes the previous absence
of meaningful teacher support and is sufficient to begin a measured separator
redesign. It does not show that teacher isolation is safe: teacher recall is
0.2308, 3/13 teacher samples were classified as student, and 5/13 generated at
least one predicted student-answer region.

## Purpose and labeling policy

Evidence-v2 contained only one human teacher sample. The teacher-focused set
therefore oversampled visual teacher-risk candidates and hard negatives using
local discovery signals. Candidate categories were never ground truth. A human
visually assigned exactly one dominant class and either drew crop-relative
student-answer rectangles or explicitly verified the answer region empty.

Color, shape, position, OCR confidence, and the candidate overlay remained
hints only. Red did not imply teacher, and blue or black did not imply student.
Mixed or unclear authorship remained `UNKNOWN`. Student boxes use the smallest
practical contiguous region intended for downstream OCR; teacher-only,
printed-only, and blank samples normally have no student box.

## Discovery and private storage

The deterministic `teacher-risk-discovery-v1` scanner used canonical PyMuPDF
renders, Tesseract evidence, OpenCV color and connected-component features,
geometry, margin activity, and nearby OCR context. It selected 48 crops from 32
pages across 12 safe aliases. It did not alter OCR, canonical images, the
separator, or the answer detector.

All source mappings, images, annotations, predictions, overlays, snapshots,
and detailed results remain under ignored `data/evaluation/evidence_teacher_v1/`
storage. The frozen directory contains a non-overwriting complete annotation
snapshot and provenance; the result namespace contains the 48 private
predictions and visual overlays. No external service was used.

## Frozen baseline methodology

The finalizer verified all historical fingerprints, validated exactly 48
human annotations, computed the semantic fingerprint, atomically froze the
complete store, reloaded and validated it independently, and then ran the
unchanged `EvidenceSeparator` and `AnswerRegionDetector`. Metrics use the
existing dominant-class, confusion-matrix, empty-region, and deterministic
one-to-one IoU contracts. Undefined metrics remain undefined.

## Classification results

| Class | Human | Predicted | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PRINTED | 11 | 4 | 1.0000 | 0.3636 | 0.5333 |
| STUDENT_CANDIDATE | 4 | 8 | 0.3750 | 0.7500 | 0.5000 |
| TEACHER_CANDIDATE | 13 | 10 | 0.3000 | 0.2308 | 0.2609 |
| UNKNOWN | 20 | 26 | 0.5000 | 0.6500 | 0.5652 |

- overall accuracy: 0.4792;
- macro F1: 0.4649;
- weighted F1: 0.4700;
- predicted UNKNOWN rate: 0.5417;
- human UNKNOWN rate: 0.4167.

Confusion matrix (rows are human, columns are predicted):

| Human / predicted | PRINTED | STUDENT | TEACHER | UNKNOWN |
| --- | ---: | ---: | ---: | ---: |
| PRINTED | 4 | 2 | 0 | 5 |
| STUDENT | 0 | 3 | 0 | 1 |
| TEACHER | 0 | 3 | 3 | 7 |
| UNKNOWN | 0 | 0 | 7 | 13 |

Teacher-to-student contamination is 3/13 (0.2308). Teacher evidence was also
sent to `UNKNOWN` seven times and correctly labeled teacher three times. No
student or printed sample was predicted teacher, while seven unknown samples
were predicted teacher. Teacher precision/recall/F1 are 0.3000/0.2308/0.2609.
These values establish a redesign baseline, not production safety.

## Answer-region results

There are 106 human boxes across 28 samples, 20 verified-empty samples, and 34
predicted boxes.

| Metric | IoU >= 0.50 | IoU >= 0.25 |
| --- | ---: | ---: |
| Matches | 5 | 11 |
| Precision | 0.1471 | 0.3235 |
| Recall | 0.0472 | 0.1038 |
| F1 | 0.0714 | 0.1571 |
| Missed human boxes | 101 | 95 |
| Extra predicted boxes | 29 | 23 |

Across all positive one-to-one overlaps there are 16 matches, with mean IoU
0.3810 and median IoU 0.4294. Of 20 verified-empty samples, 16 were correctly
predicted empty and four produced false-positive regions. Empty precision,
recall, and F1 are 0.5517, 0.8000, and 0.6531 respectively.

Teacher-only localization is unsafe: 8/13 teacher samples produced no student
region, but 5/13 produced seven student-answer boxes in total (mean 0.5385 per
teacher sample).

## Stratified observations

Discovery strata are analysis context, not labels. Teacher-to-student errors
occurred in chromatic-ink, margin/score, detected-Test, and hard-negative
contexts. The rate was 2/6 for teacher samples with margin activity, 2/6 for
score-like geometry, and 2/3 in the margin/score-risk stratum. No such error
occurred among teacher samples in the correction-stroke, tick/cross-geometry,
mixed-teacher-context, OCR-context, or short-answer-teacher-risk strata, but
their supports are small. Five teacher samples with nearby high-confidence
print produced five answer boxes, illustrating that classification and
localization safety must be evaluated separately.

## Comparison and decision

| Benchmark | Samples | P/S/T/U support | Accuracy | Macro F1 | Student F1 | Teacher F1 | IoU .50 F1 | IoU .25 F1 | Correct empty |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 4C.4R | 18 | 3/14/0/1 | 0.1667 | 0.1225 | 0.2500 | undefined | 0.0402 | 0.1518 | 0/4 |
| Evidence-v2 | 48 | 30/8/1/9 | 0.3125 | 0.2469 | 0.3750 | 0.0000 | 0.0376 | 0.0657 | 5/9 |
| Teacher-v1 | 48 | 11/4/13/20 | 0.4792 | 0.4649 | 0.5000 | 0.2609 | 0.0714 | 0.1571 | 16/20 |

The algorithm is identical across these measurements; differences reflect
benchmark composition, not improvement or regression.

Decision: **B. DATA COVERAGE SUFFICIENT FOR PHASE 4C.6 EVIDENCE SEPARATOR V2
REDESIGN.** Teacher-v1 adds 13 human teacher samples across diverse local risk
contexts and 12 papers, while the two prior frozen datasets retain stronger
printed and student support. The combined evidence is adequate to define and
measure a redesign. Teacher safety is emphatically not accepted: future work
must target teacher-to-student contamination, unknown/teacher confusion,
student precision, box recall, and empty-region false positives.

## Limitations

- Teacher-v1 deliberately changes the class distribution and cannot be used to
  infer corpus prevalence.
- Four human student samples are too few to assess student behavior from this
  dataset alone; evidence-v2 and Phase 4C.4R remain necessary complements.
- Discovery signals bias the pool toward chromatic and component-based risks.
- Teacher authorship remains a human visual judgment in flattened scans.
- Box recall remains low, and teacher samples still emit student regions.
- No detector tuning, OCR change, ML model, grading, or downstream extraction
  was performed in Phase 4C.5D.

## Phase 4C.6 V2 safety result

Four fixed V2 variants were evaluated without modifying this frozen benchmark.
The best diagnostic candidate, V2-B, regressed teacher-to-student contamination
from V1's 3/13 to 10/13, classified no teacher sample correctly, produced
student regions on 12/13 teacher samples, and emitted 138 student boxes on those
samples. Teacher-v1 accuracy/macro F1 fell from 0.4792/0.4649 under V1 to
0.3542/0.3366. The candidate was rejected; these results do not alter or replace
the Phase 4C.5D baseline. See `docs/evidence-separator-v2.md`.

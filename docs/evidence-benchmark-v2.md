# Evidence-v2 benchmark expansion

## Status

Phase 4C.5B has frozen and measured 48 private crops from 36 pages across 12
safely aliased papers. All 48 records are explicitly human verified. Candidate
discovery categories remain sampling hints, **not ground truth**. The complete
recoverable annotation snapshot has semantic fingerprint
`b28eb7ce4daa69bdaa89687cc905366e92d4ed351205c706a77bb16ffea2614b`.

The frozen 18-sample Phase 4C.4R benchmark remains immutable and separate. Its
authoritative annotation fingerprint is
`41d2364cbc0ac56269c30ef41473ccb67e9c08d7109e748f119f274f0671ab35`.

Because evidence-v2 contains only one human teacher sample, Phase 4C.5C has
prepared a separate pending teacher-risk dataset. It does not alter this frozen
snapshot or its metrics. See `docs/teacher-evidence-benchmark.md`.

## Labeling unit and class definitions

An evidence-v2 sample is a smaller page-local region with enough surrounding
context for a human to assess its dominant visible evidence. The human selects
exactly one class:

- `PRINTED`: examination questions, instructions, passages, choices, guides,
  table labels, or other mechanically printed content;
- `STUDENT_CANDIDATE`: visually attributable student handwriting, including
  short, fill-in, sentence, paragraph, faint, corrected, or crossed-out work;
- `TEACHER_CANDIDATE`: visually attributable ticks, crosses, scores,
  corrections, comments, circles, or other teacher-added marks;
- `UNKNOWN`: genuinely ambiguous authorship or inseparable mixed evidence.

`UNKNOWN` is not a replacement for an oversized crop. Samples are deliberately
bounded so the human can prefer a homogeneous class where the visual evidence
supports one. Ink color, Tesseract confidence, and discovery category never
determine the human label automatically.

## Student answer-box policy

A human answer rectangle represents the smallest practical contiguous student
answer region that a downstream handwriting OCR system should attempt to
transcribe:

- one handwritten word in a blank: one tight word box;
- one continuous handwritten sentence: a sentence- or line-level box where
  practical;
- paragraph handwriting: line-level or coherent contiguous text-region boxes,
  not one whole-Test rectangle;
- spatially separated answers: separate boxes;
- printed guides without visible student writing: explicit verified-empty;
- teacher-only marks: explicit verified-empty;
- mixed student/teacher evidence: box only the student portion when visually
  separable; otherwise label the crop `UNKNOWN`.

Do not create one box per character. Do not include surrounding printed prompts
or separable teacher ticks/scores merely to enlarge a box. The task is geometry
annotation only; no transcription is requested.

## Candidate discovery and sampling

The local `evidence-candidate-discovery-v1` sampler uses existing canonical
PyMuPDF page renders, Tesseract word hierarchy and confidence, local HSV/color
evidence, connected components, sparse writing-guide bands, and detected Test
marker proximity. It proposes five discovery groups: printed, student-risk,
teacher-mark-risk, mixed/uncertain, and blank-answer. These signals select
regions for human inspection only.

The fixed selected pool is:

| Discovery group (not a label) | Candidates |
| --- | ---: |
| Printed | 10 |
| Student/irregular-writing risk | 14 |
| Teacher-mark risk | 12 |
| Mixed/uncertain | 6 |
| Blank-answer risk | 6 |
| **Total** | **48** |

Discovery retained 233 proposals after page-local overlap filtering.
Deterministic category quotas, per-paper caps, and cross-category overlap
checks selected 48; 185 retained proposals were not selected during balancing.
Crop bytes are checked for exact duplication during
materialization. The selected pool contains no exact duplicate crop.

## Provenance, privacy, and immutability

All version-2 artifacts are private and ignored:

```text
data/evaluation/evidence_v2/
  benchmark_manifest.json
  discovery_provenance.json
  labeling_metadata.json
  samples/
  overlays/
  annotations.json       # created only by human saves
  frozen/                # complete non-overwriting annotation snapshot
  results/phase4c5b/     # private predictions and aggregate evaluation
  evaluation-overlays/   # private geometry comparisons
```

Private provenance records a safe alias, page number, optional detected Test
number, page-local crop geometry, canonical page SHA-256, source PDF SHA-256,
and sampling reason. Original filenames never appear in committed code or
terminal summaries. Canonical source hashes are checked before and after crop
creation; crops and overlays never overwrite their sources. The interface binds
only to `127.0.0.1` and has no external dependencies or image upload path.

## Human workflow

Launch from the repository root:

```bash
uv run python -m scripts.annotate_evidence --dataset evidence-v2
```

Open `http://127.0.0.1:8765/`. Every sample begins pending with no preselected
class and no automatically accepted rectangles. Inspect the crop and neutral
candidate overlay, choose one class, draw zero or more student-answer boxes or
explicitly verify empty, then save. The header reports completed progress and
the current human class distribution for information only.

Safe candidate validation:

```bash
uv run python -m scripts.prepare_evidence_expansion validate
```

## Frozen human benchmark

Persisted human annotations—not UI counters—contain 30 `PRINTED`, 8
`STUDENT_CANDIDATE`, 1 `TEACHER_CANDIDATE`, and 9 `UNKNOWN` samples. Thirty-nine
samples contain 56 total student-answer boxes; nine are explicitly verified
empty. The snapshot and provenance were independently reloaded through the
production models. Their annotation and manifest fingerprints, complete record
count, and answer-box count match the live validated store.

The fingerprint uses the existing annotation contract: records are sorted by
sample ID, rectangle order is preserved, and Pydantic's semantic JSON data is
serialized with Unicode preserved, compact separators, and sorted object keys.
It covers schema version, sample IDs, human classes, explicit answer status,
ordered rectangle geometry, and the `human_verified` state. File whitespace and
key order do not affect it.

## Unchanged separator baseline

The evaluation used the existing Tesseract settings, `EvidenceSeparator`,
`AnswerRegionDetector`, thresholds, and geometry rules without tuning.

| Class | Support | Predicted | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PRINTED | 30 | 6 | 1.0000 | 0.2000 | 0.3333 |
| STUDENT_CANDIDATE | 8 | 8 | 0.3750 | 0.3750 | 0.3750 |
| TEACHER_CANDIDATE | 1 | 0 | undefined | 0.0000 | 0.0000 |
| UNKNOWN | 9 | 34 | 0.1765 | 0.6667 | 0.2791 |

Overall accuracy is 0.3125, macro F1 is 0.2469, weighted F1 is 0.3232,
predicted UNKNOWN rate is 0.7083, and human UNKNOWN rate is 0.1875. The
confusion matrix (rows human, columns predicted) is:

| Human / predicted | PRINTED | STUDENT | TEACHER | UNKNOWN |
| --- | ---: | ---: | ---: | ---: |
| PRINTED | 6 | 2 | 0 | 22 |
| STUDENT | 0 | 3 | 0 | 5 |
| TEACHER | 0 | 0 | 0 | 1 |
| UNKNOWN | 0 | 3 | 0 | 6 |

The student class has five false positives and five false negatives. The only
human teacher sample became UNKNOWN; teacher precision is undefined because the
detector predicted no teacher sample. One example cannot establish
teacher-contamination safety.

## Answer localization and empty behavior

The 56 human boxes were compared with 157 predictions using deterministic
sample-local one-to-one matching:

| Metric | IoU >= 0.50 | IoU >= 0.25 |
| --- | ---: | ---: |
| Matches | 4 | 7 |
| Precision | 0.0255 | 0.0446 |
| Recall | 0.0714 | 0.1250 |
| F1 | 0.0376 | 0.0657 |
| Missed human boxes | 52 | 49 |
| Extra predictions | 153 | 150 |

There are 31 positive-overlap one-to-one matches, with mean IoU 0.1712 and
median IoU 0.0762. Eighteen samples received no predicted answer region. Of nine
human verified-empty samples, five were correctly empty and four received false
positive regions. Empty-prediction precision is 0.2778 and empty recall is
0.5556.

## Safe stratified observations

Discovery groups are not labels. Human review showed why: all 12 teacher-risk
candidates became either PRINTED (5) or UNKNOWN (7), while the sole human
teacher example came from the blank-answer discovery group. Classification
errors occurred in 6/10 printed candidates, 11/14 student-risk candidates, 7/12
teacher-risk candidates, 5/6 mixed candidates, and 4/6 blank candidates.
Errors also occurred in 11/18 paragraph contexts, 6/6 short-answer contexts,
12/18 colored-ink contexts, 14/19 detected-Test samples, and 19/29 samples with
no detected Test. These are composition observations, not ground truth derived
from discovery hints.

## Comparison with Phase 4C.4R

Phase 4C.4R used 18 large Test crops; evidence-v2 uses 48 smaller evidence
units. The unchanged algorithm therefore has different metrics because the
benchmark distribution and box policy changed. Overall accuracy is 0.3125
versus 0.1667, macro F1 0.2469 versus 0.1225, and student F1 0.3750 versus
0.2500. Conversely, IoU-0.50 F1 is 0.0376 versus 0.0402 and IoU-0.25 F1 is
0.0657 versus 0.1518. These deltas are **not** algorithm improvement or
regression.

## Readiness decision

**C. TEACHER-FOCUSED BENCHMARK EXPANSION REQUIRED FIRST.** Printed, student,
unknown, and blank coverage is sufficient to expose redesign problems, but one
human teacher sample cannot evaluate teacher attribution or contamination
safety. Candidate discovery produced teacher-risk cases, not verified teacher
ground truth. The current separator is not accepted for downstream extraction,
and no thresholds were tuned in Phase 4C.5B.

## Limitations

- Discovery categories can be wrong and must not be used as labels.
- A teacher-risk crop may contain student or printed evidence instead.
- Only 19 selected samples have a nearby OCR-detected Test number; Test metadata
  is optional and may itself be noisy.
- Candidate selection did not produce balanced human labels; teacher support is
  one despite 12 teacher-risk discovery candidates.
- No detector thresholds, answer-region rules, OCR settings, or models changed
  in this phase.
- Metrics describe 48 private regions and should not be generalized to teacher
  contamination safety or the full 40-paper corpus.

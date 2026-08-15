# Evidence separation and answer-region isolation

## Status

Phase 4C.4 implements the deterministic evidence-separation boundary, a
loopback-only visual labeling tool, frozen human annotation provenance, and a
measured 18-sample private baseline. The current separator is **not sufficiently
validated for downstream student-answer extraction**. Human class coverage is
17 `STUDENT_CANDIDATE`, 1 `UNKNOWN`, and zero `PRINTED` or
`TEACHER_CANDIDATE`; consequently printed and teacher metrics are undefined.
Answer localization also has high over-expansion and low geometric recall.

Phase 4C.4R completed a full human re-verification and immutable snapshot. The original
`a7007ef2...` fingerprint remains historical baseline metadata, but the complete
annotation file that produced it is not recoverable after a later semantic
label drift. It was not reconstructed from aggregate results. All 18 images were
visually re-verified in a fresh session before the replacement baseline was
frozen.

Phase 4C.5B froze the separate schema-v2 benchmark: 48 smaller regions from 36
pages across 12 safe aliases, with fingerprint
`b28eb7ce4daa69bdaa89687cc905366e92d4ed351205c706a77bb16ffea2614b`.
The unchanged separator measured 0.3125 overall accuracy and 0.0376
answer-region F1 at IoU 0.50. Human support is 30 printed, 8 student, 1 teacher,
and 9 unknown. The single teacher sample is insufficient for safety validation,
so the decision is teacher-focused benchmark expansion before redesign.
See `docs/evidence-benchmark-v2.md` for the complete metrics and policy.

Phase 4C.5C now prepares a separate 48-crop teacher-risk pool across 12 safe
paper aliases. It remains entirely pending: discovery categories are not human
labels, no separator metrics have been run, and no detector rule changed. See
`docs/teacher-evidence-benchmark.md`.

## Purpose and architecture

This layer asks where likely answer evidence exists inside an already detected
`TestRegion`; it does not attempt to transcribe or grade handwriting.

```text
canonical PaperPage + Tesseract word evidence + TestRegion
                         |
                  EvidenceSeparator
                         |
        typed image and geometry measurements
                         |
                 EvidenceRegion[]
          printed / student candidate /
          teacher candidate / unknown
                         |
                AnswerRegionDetector
                         |
               StudentAnswerRegion[]
```

`EvidenceSeparationService` composes these boundaries without changing
`PaperPage`, `OCRService`, `OCRProvider`, or `TestRegion`. All domain results are
immutable and retain paper, page, Test, bounding-box, source-image, OCR-word,
and strategy-version provenance.

## Evidence classes

- `PRINTED` means multiple signals consistently support regular printed text.
- `STUDENT_CANDIDATE` means multiple signals support handwriting-like evidence;
  it is not a verified claim of authorship.
- `TEACHER_CANDIDATE` means multiple signals support an isolated annotation-like
  mark; it is not a verified claim that a teacher made it.
- `UNKNOWN` preserves conflicting, mixed, or insufficient evidence. It is an
  intentional production result, not an error.

No class is ground truth. Low OCR confidence does not imply student writing,
and color or saturation alone does not imply authorship.

## Image and geometry features

The OpenCV feature extractor measures a bounded local raster region and emits
typed, normalized values:

- HSV saturation mean and variation;
- red- and blue-dominant foreground ratios;
- grayscale foreground and dark-pixel ratios;
- local contrast and edge density;
- connected-component count.

Tesseract hierarchy and boxes provide:

- line baseline, height, and spacing irregularity;
- line density and regularity;
- fragmentation and isolation;
- margin position and words per line;
- normalized Tesseract confidence when available.

The deterministic `evidence-separation-v1` classifier accepts a class only when
independent signals agree. Competing or weak scores produce `UNKNOWN`.
Classification scores are explainable rule scores in `[0, 1]`; they are not
calibrated probabilities.

## Answer-space detection

`answer-region-v1` detects conservative candidates from two sources:

1. repeated horizontal writing guides with sufficient width and regular
   vertical spacing, including blank answer spaces; and
2. spatial clusters of `STUDENT_CANDIDATE` evidence in areas with limited
   printed overlap.

Candidates preserve page geometry and list the signals and evidence indices
that produced them. The detector does not crop, resize, erase printed material,
or assert that every candidate contains an answer. Tables, boxes, irregular
scans, and unconstrained paragraph areas remain difficult.

## Immutable sources and visual debugging

The separator, answer detector, and overlay renderer hash the source before and
after processing. They never write to a canonical path. Private overlays show
safe class labels and answer-candidate boxes on derived copies only.

Generated artifacts live beneath ignored
`data/evaluation/evidence/` storage:

```text
samples/                 private crops
overlays/                classifier and answer-region overlays
results/                 private candidate predictions
labeling_worksheet.md    human labeling worksheet
benchmark_manifest.json private manifest
annotations.json         visual labels created by the local tool
backups/                 preserved pre-tool worksheet backup
```

## Private benchmark and human verification

The prepared benchmark contains 18 representative Test-region crops covering
mostly printed text, clear and difficult handwriting, teacher marks/scores,
mixed evidence, paragraph answers, short answers, and blank answer areas.
Categories describe sampling coverage only; they are not labels.

All 18 visual annotations passed strict validation: the expected IDs appear
exactly once, every record is explicitly human verified, every class is valid,
answer status is explicit, all geometry is integer and crop-relative, all boxes
remain inside their crops, and every private sample image exists. The earlier
private worksheet contains human transcription work entered into fields intended
for rectangle coordinates. It remains preserved byte-for-byte under ignored
`backups/`; its strings were never converted into geometry or labels.

The canonical human annotation fingerprint is:

`a7007ef2e4887dd7c9b298de0dcb6809df7a222291a4a13104e3576f3c330f2a`

This SHA-256 covers sorted canonical JSON for every class, explicit answer
state, and rectangle. It detects accidental label changes but is not a privacy
or security mechanism. Private provenance records 18 samples, 14 with answer
boxes, 4 explicitly verified empty, and 100 total human boxes.

### Phase 4C.4R integrity incident and re-verification

The original baseline fingerprint remains associated with the preserved
historical provenance and result bundle. A read-only investigation found that
the current annotation file had fingerprint
`bcb7dd2f85042b03c896d8dd49f1acd1c61cadc45b1a3e1a607378865ae6465e`
and one safe class-metadata difference. Localization aggregates still matched,
but the semantic fingerprint did not. No complete authoritative copy of the
original annotations was found, so the old fingerprint is not being reassigned
or silently replaced.

Before re-verification, the current annotation file, original provenance, and
historical evaluation result were copied into ignored forensic backup storage.
The repository now compares canonical semantic fingerprints before each
replacement. A changed store causes the previous valid file to be copied
byte-for-byte to `backups/` using a UTC timestamp and its full semantic
fingerprint. Unchanged saves create no redundant backup, and existing backups
are never deleted or overwritten automatically.

Start the separate re-verification workflow with:

```bash
uv run python -m scripts.annotate_evidence --reverify
```

The loopback-only interface shows the current saved annotation and rectangles,
but deliberately leaves all class choices unselected. Old `human_verified`
values do not count. A separate private session ledger records the exact
fingerprint of each annotation approved by the dedicated re-verified/save
action and reports `Re-verified: X/18`. If an approved annotation later changes,
that sample no longer counts. All 18 exact current records subsequently
validated, with no missing or stale approvals.

### Phase 4C.4R frozen replacement baseline

The authoritative replacement fingerprint is:

`41d2364cbc0ac56269c30ef41473ccb67e9c08d7109e748f119f274f0671ab35`

The complete private annotation store and matching provenance are retained
beneath ignored `data/evaluation/evidence/frozen/` storage. The snapshot was
independently reloaded through production models, validated against all samples,
and fingerprinted again. Live, snapshot, and provenance fingerprints match.
The original `a700...` fingerprint, later `bcb7...` drifted file, and new
`41d2...` replacement remain distinct historical states.

Human distribution is 3 `PRINTED`, 14 `STUDENT_CANDIDATE`, 0
`TEACHER_CANDIDATE`, and 1 `UNKNOWN`; there are 100 answer boxes across 14
samples and 4 explicitly verified-empty samples. The unchanged separator
baseline measured 0.1667 overall accuracy, 0.1225 macro F1, 0.2010 weighted F1,
and 0.8889 predicted UNKNOWN rate. Student precision/recall/F1 are
1.0000/0.1429/0.2500. Teacher metrics remain undefined because human teacher
support is zero.

Answer localization remains unchanged: 348 predicted boxes, 9 matches and
0.0402 F1 at IoU 0.50, and 34 matches and 0.1518 F1 at IoU 0.25. None of the
four verified-empty samples was predicted empty. These results remain unsuitable
for downstream extraction; the purpose of re-verification was benchmark
integrity, not detector improvement.

Launch the replacement local interface from the repository root:

```bash
uv run python -m scripts.annotate_evidence
```

Then open `http://127.0.0.1:8765/`. The server binds only to IPv4 loopback,
serves all HTML, JavaScript, crops, and overlays locally, suppresses request
logging, and has no external dependencies or network integrations. Press
`Ctrl+C` in the terminal to stop it.

For each sample, a human must:

1. inspect the original crop and optional existing prediction overlay;
2. choose exactly one evidence label (`PRINTED`, `STUDENT_CANDIDATE`,
   `TEACHER_CANDIDATE`, or `UNKNOWN`);
3. drag zero or more rectangles directly around student-answer regions; or
4. explicitly select verified empty when no student-answer region exists;
5. save or save and advance.

Canvas rectangles are converted from displayed coordinates to original
crop-relative `x,y,width,height` values. Saved labels are validated against the
sample dimensions and written atomically to ignored `annotations.json`. Mixed
or uncertain evidence should be `UNKNOWN`; no transcription is requested.

Run the private preparation and safe validation commands with:

```bash
uv run python -m scripts.prepare_evidence_benchmark prepare
uv run python -m scripts.prepare_evidence_benchmark validate
```

Preparation is deterministic and idempotent: it does not overwrite an existing
worksheet. The CLI reports safe counts only and never prints private image paths
or content.

The visual tool provides add-by-drag, per-rectangle delete, clear, previous,
next, save, save-and-next, verified-empty, and completed/total progress controls.
It does not run evaluation. The measured baseline can be reproduced locally
only after annotations validate:

```bash
uv run python -m scripts.evaluate_evidence
```

## Baseline methodology

The evaluation reran the unchanged `EvidenceSeparator` and
`AnswerRegionDetector` on exactly the frozen 18 samples. No thresholds, labels,
OCR transcription ground truth, or OCR provider were changed after labels were
seen. Dominant predicted class is the unique class with the largest summed
evidence-box area; an empty or tied result becomes `UNKNOWN`.

Classification metrics compare one dominant prediction with one human dominant
class per sample. Zero-support classes retain `None` rather than manufactured
zeros. Macro F1 averages only defined class F1 values; weighted F1 weights
defined supported classes.

Answer boxes use crop-relative geometry. Candidate pairs are ordered by highest
IoU, then stable human/prediction index, and matched one-to-one. Metrics are
reported independently at IoU 0.50 and 0.25. Mean and median use all positive
one-to-one overlaps.

## Evidence-class results

| Class | Support | Predicted | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PRINTED | 0 | 0 | undefined | undefined | undefined |
| STUDENT_CANDIDATE | 17 | 2 | 1.0000 | 0.1176 | 0.2105 |
| TEACHER_CANDIDATE | 0 | 0 | undefined | undefined | undefined |
| UNKNOWN | 1 | 16 | 0.0625 | 1.0000 | 0.1176 |

- overall accuracy: 0.1667 (3/18);
- macro F1 over defined classes: 0.1641;
- weighted F1: 0.2054;
- predicted UNKNOWN rate: 0.8889;
- human UNKNOWN rate: 0.0556;
- student false positives: 0;
- student false negatives: 15.

Confusion matrix (rows human, columns predicted):

| Human / predicted | PRINTED | STUDENT | TEACHER | UNKNOWN |
| --- | ---: | ---: | ---: | ---: |
| PRINTED | 0 | 0 | 0 | 0 |
| STUDENT | 0 | 2 | 0 | 15 |
| TEACHER | 0 | 0 | 0 | 0 |
| UNKNOWN | 0 | 0 | 0 | 1 |

Teacher-to-student and student-to-teacher counts are both zero, but this does
**not** demonstrate contamination safety: no sample has a human
`TEACHER_CANDIDATE` label, and the separator predicted no teacher-dominant
sample. Teacher precision, recall, and F1 are therefore undefined.

## Answer-localization results

| Metric | IoU >= 0.50 | IoU >= 0.25 |
| --- | ---: | ---: |
| Matched boxes | 9 | 34 |
| Precision | 0.0259 | 0.0977 |
| Recall | 0.0900 | 0.3400 |
| F1 | 0.0402 | 0.1518 |

- human boxes: 100;
- predicted boxes: 348;
- positive-overlap one-to-one matches: 51;
- mean positive matched IoU: 0.3286;
- median positive matched IoU: 0.3122;
- missed human boxes at IoU 0.50: 91;
- extra predicted boxes at IoU 0.50: 339;
- explicitly verified-empty samples: 4;
- correctly predicted empty: 0;
- empty samples with false-positive regions: 4.

## Safe error analysis and visual QA

Measured errors occur across every sampling category. Classification errors
include 8/9 mixed-evidence, 6/7 short-answer, 5/7 paragraph, 3/4 difficult-
handwriting, 6/7 teacher-mark-risk, and 3/3 teacher-score-risk samples. All four
difficult-handwriting samples miss at least one human box at IoU 0.50. Every
category contains extra answer predictions; all three samples selected for blank
answer-space coverage do so as well.

Private comparison overlays were inspected for short answers, paragraphs,
difficult handwriting, teacher-mark risk, and verified-empty cases. Human and
predicted geometry aligns with the source crop, and canonical images remain
unchanged: visual rendering integrity **PASS**. Quality suitability **FAIL**:
writing guides frequently create broad predicted regions, while human labels
often use tighter evidence boxes. Fragmented OCR evidence also produces many
small candidates. This granularity mismatch contributes to over-expansion and
under-coverage and must be resolved explicitly in any replacement benchmark.

## Decision gate

**D. INSUFFICIENT HUMAN LABEL COVERAGE.** The benchmark cannot measure PRINTED
or TEACHER_CANDIDATE behavior because both have zero human support, despite
selection categories that were intended to include those risks. It therefore
cannot establish teacher-contamination safety. Independently, the measured
answer-localization baseline is too weak for downstream extraction and indicates
that answer-region detection will likely require redesign after label coverage
and box-granularity policy are corrected. No thresholds were tuned in this
phase.

## Privacy and teacher contamination

All evaluation images, labels, predictions, and worksheets are private and
ignored by Git. Safe aliases and deterministic sample IDs replace original
filenames. Historical pages contain flattened printed content, student writing,
and teacher marks, so color and spatial heuristics remain uncertain. Nothing is
removed destructively and no private transcription is used for tuning.

## Limitations and future template alignment

- Human dominant-class coverage is severely imbalanced and excludes PRINTED and
  TEACHER_CANDIDATE.
- Human boxes and predicted answer-space envelopes use inconsistent granularity
  on some samples.
- Black handwriting and faded colored ink can resemble print.
- Red/blue ink does not prove teacher/student authorship.
- Tesseract boxes omit unrecognized marks and can fragment handwriting.
- Answer-space detection over-expands writing guides, misses tight human boxes,
  and produces regions on all verified-empty samples.
- No handwriting model, OCR correction, grading, or downstream answer extraction
  is implemented.

A preferred future improvement is an optional clean-exam template boundary:
align a clean page with a student page, compare stable printed content, and
retain residual ink as candidates. This phase does not depend on a template and
does not implement alignment or subtraction.

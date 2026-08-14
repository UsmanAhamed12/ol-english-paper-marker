# Evidence separation and answer-region isolation

## Status

Phase 4C.4 has implemented the deterministic evidence-separation boundary and
prepared a private 18-sample labeling benchmark. The benchmark is **not yet
evaluated**: all samples remain pending human verification. Precision, recall,
F1, confusion-matrix, UNKNOWN-rate, and answer-region metrics must not be
reported until those labels exist.

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
```

## Private benchmark and human verification

The prepared benchmark contains 18 representative Test-region crops covering
mostly printed text, clear and difficult handwriting, teacher marks/scores,
mixed evidence, paragraph answers, short answers, and blank answer areas.
Categories describe sampling coverage only; they are not labels.

Every sample is currently `pending`. A human must inspect each crop and overlay,
select exactly one evidence label (`PRINTED`, `STUDENT_CANDIDATE`,
`TEACHER_CANDIDATE`, or `UNKNOWN`), and explicitly verify zero or more
crop-relative student-answer rectangles. Mixed or uncertain evidence should be
`UNKNOWN`; no transcription is requested.

Run the private preparation and safe validation commands with:

```bash
uv run python -m scripts.prepare_evidence_benchmark prepare
uv run python -m scripts.prepare_evidence_benchmark validate
```

Preparation is deterministic and idempotent: it does not overwrite an existing
worksheet. The CLI reports safe counts only and never prints private image paths
or content.

After human verification, a later continuation of this same phase can measure:

- per-class precision, recall, and F1;
- UNKNOWN rate and confusion matrix;
- student-answer region precision, recall, and IoU.

Conservative student precision is prioritized over aggressive recall. Metrics
must not be generated from the classifier's own predictions.

## Privacy and teacher contamination

All evaluation images, labels, predictions, and worksheets are private and
ignored by Git. Safe aliases and deterministic sample IDs replace original
filenames. Historical pages contain flattened printed content, student writing,
and teacher marks, so color and spatial heuristics remain uncertain. Nothing is
removed destructively and no private transcription is used for tuning.

## Limitations and future template alignment

- Candidate labels are heuristic and await human measurement.
- Black handwriting and faded colored ink can resemble print.
- Red/blue ink does not prove teacher/student authorship.
- Tesseract boxes omit unrecognized marks and can fragment handwriting.
- Answer-space detection cannot yet reliably model every table or free-form
  writing area.
- No handwriting model, OCR correction, grading, or downstream answer extraction
  is implemented.

A preferred future improvement is an optional clean-exam template boundary:
align a clean page with a student page, compare stable printed content, and
retain residual ink as candidates. This phase does not depend on a template and
does not implement alignment or subtraction.

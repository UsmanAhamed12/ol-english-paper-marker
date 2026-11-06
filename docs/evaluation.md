# Historical marking evaluation

## Status and scope

Phase 19 provides a typed, local evaluation boundary for comparing completed
system predictions with human-marked historical references. It does not
expose teacher marks to the model or claim accuracy. A verified reference set
and a controlled evaluation run are prerequisites for measured results.

## Data separation

`HistoricalEvaluationDataset` stores only anonymized paper aliases, Test
numbers, human-awarded marks, and known maxima. `EvaluationPredictionRun`
stores model, prompt, and pipeline versions plus prediction outcomes; its
schema has no teacher-score field and rejects unknown fields. Create and freeze
the prediction run before loading the reference dataset into evaluation.
Evaluation happens only after grading has completed.

Unresolved or failed question outcomes contain no predicted mark. A missing
prediction or a question needing review is counted in coverage, not converted
to zero. A paper total is scored only when every question in that paper has a
scored prediction. Predictions outside the reference set and predictions
above the official maximum fail explicitly.

## Dataset format and loading

The versioned JSON format is validated by Pydantic and can be loaded with
`load_dataset(Path(...))`:

```json
{
  "schema_version": "1.0",
  "dataset_id": "verified-set-2025",
  "dataset_version": "v1",
  "references": [
    {
      "paper_alias": "paper-001",
      "test_number": 1,
      "awarded_marks": "4",
      "maximum_marks": "5"
    }
  ]
}
```

Aliases must be non-identifying safe identifiers. Duplicate paper/Test
references, invalid marks, and unknown fields are rejected. The SHA-256
fingerprint covers the canonical full dataset, including reference marks, and
is included in every report for auditability. Keep real reference files in
private, access-controlled storage; do not commit them to the repository.

## Metrics and incomplete runs

`evaluate_run(dataset, prediction_run)` produces:

- MAE;
- RMSE;
- mean signed error (`prediction - teacher`, so positive values indicate
  over-marking);
- percentage of scored marks within ±1, ±2, and ±5 marks;
- question-level and paper-total metrics;
- question and paper coverage;
- missing and unscored prediction counts; and
- unavailable outcomes grouped by pipeline stage when that information exists.

Metrics are calculated only for scored pairs. Undefined metrics are `null`
when there are no eligible pairs. Paper totals use complete papers only;
question metrics can still use the independently scored questions in an
incomplete paper. Reports contain aggregate metrics, safe aliases, and
component provenance, not student response text.

```python
report = evaluate_run(dataset, frozen_prediction_run)
report_json = serialize_report(report)
```

The framework intentionally does not choose, clean, or calibrate historical
marks. Reference preparation, leakage controls for a real evaluation run, and
the 40-paper evaluation remain separate follow-up work. Correlation is not
used as a substitute for mark-error metrics.

## Phase 20 real-input smoke

The private `data/test/` directory contains one unmarked student paper and one
marking-scheme PDF. Both are ignored by Git. A local CLI smoke completed using
the installed Tesseract English model and local `llama2:latest`. It aligned 12
Tests, reported 0 confirmed marks, 69 unresolved marks, 88 review items, and
did not provide a final score. This is a conservative incomplete pipeline run,
not a 40-paper evaluation and not an accuracy measurement.

There is no separately verified teacher-score reference for that paper. The
run therefore cannot be compared against ground truth, and its prediction must
not be converted into evaluation metrics. The 40 PDFs in `data/raw/marked_papers/`
are teacher-annotated and remain prohibited as model inputs for evaluation.
Phase 20 remains incomplete until 40 safe student-paper/scheme pairs and
separately held, verified teacher marks are available.

## Phase 21 one-paper error analysis

A follow-up diagnostic reran the same private input and summarized the
serialized result without displaying student text, rubric text, or raw model
responses. It observed:

| Boundary | Observed result count |
|---|---:|
| Objective criteria routed to `review_required` (`student_review_required`) | 63 |
| Objective results marked `structure_incomplete` (`missing_student_test`) | 6 |
| Subjective criteria marked `unsupported_criterion` (`unknown_criterion`) | 16 |
| Final confirmed marks | 0 of 69 known maximum |
| Review queue items | 88 |

Stage warnings recorded two missing Tests, 14 non-extracted Tests, 23 marking
scheme warnings, one student-missing Test, two scheme-missing Tests, and 12
review-required aligned Tests. The aggregate evidence points to three
investigation areas: conservative student-evidence extraction/review,
incomplete structure detection, and marking-scheme parsing that leaves
criteria unknown. These counts describe one paper and are not estimates of
general error rates.

All 63 objective results routed to review carried student evidence with
`review_required` extraction status. The six `structure_incomplete` results
had no student evidence reference. This indicates that the immediate blockers
occur before objective score comparison; changing the subjective grading
confidence threshold would not resolve them.

No confidence threshold, model setting, or grading behavior was changed.
There are no verified teacher scores for this paper, so changing thresholds
based on this run would optimize for workflow completion rather than marking
correctness. Calibration remains unperformed until predictions can be compared
against independently verified reference marks.

The local formative-feedback service was also exercised with synthetic
answer/rubric evidence using the configured `llama2` model. It returned a
validated response with feedback and an improvement tip in about 119 seconds.
This tests local provider wiring and response validation only; it is not a
student-paper feedback result or a quality evaluation. A subsequent full
`data/test/` CLI rerun failed before alignment after about 593 seconds and
produced no usable score or real-paper feedback. The earlier smoke counts
above remain the recorded successful real-input run; the later failure is
reported separately rather than treated as a successful retry.

## Validation

Unit tests use synthetic references and predictions only. They validate score
bounds, prediction/reference separation, versioned dataset loading,
fingerprints, missing-result coverage, complete-paper aggregation, failure
stage accounting, and deterministic report serialization. These tests are
contract checks, not evidence of grading accuracy.

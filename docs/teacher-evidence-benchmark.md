# Teacher-focused evidence benchmark

## Status

Phase 4C.5C has prepared a separate private `evidence-teacher-v1` candidate
set for human labeling. It contains 48 pending crops from 32 pages across 12
safely aliased papers. No human ground truth exists yet, no separator metrics
have been calculated, and the existing separator and answer detector are
unchanged.

The prior frozen baselines remain separate and immutable: Phase 4C.4R has
fingerprint `41d2364cbc0ac56269c30ef41473ccb67e9c08d7109e748f119f274f0671ab35`
and evidence-v2 has fingerprint
`b28eb7ce4daa69bdaa89687cc905366e92d4ed351205c706a77bb16ffea2614b`.

## Purpose and labeling policy

Evidence-v2 contains only one human `TEACHER_CANDIDATE`, so it cannot validate
teacher-contamination safety. This dataset deliberately oversamples visual
regions that may contain teacher marks plus hard negatives. Candidate discovery
is high recall and is not an authorship decision.

For each crop, a human must choose exactly one dominant class: `PRINTED`,
`STUDENT_CANDIDATE`, `TEACHER_CANDIDATE`, or `UNKNOWN`. Color, shape, position,
OCR confidence, and the displayed candidate box are hints only. A red mark is
not automatically a teacher mark, and a blue or black stroke is not
automatically student writing. Mixed or unclear authorship should remain
`UNKNOWN`.

Student-answer rectangles follow the existing v2 policy: draw the smallest
practical contiguous student-answer region intended for downstream OCR. Do not
include separable printed prompts or teacher marks. Teacher-only, printed-only,
and genuinely blank crops should be explicitly verified empty. No text is
transcribed.

## Discovery architecture

The local deterministic `teacher-risk-discovery-v1` scanner reuses existing
canonical PyMuPDF renders and Tesseract structured evidence. OpenCV derives
candidate signals from chromatic foreground, compact connected components,
angled tick/cross-like geometry, score-like shapes, correction strokes, margin
location, nearby OCR hierarchy, printed controls, and mixed context. Signals
select crops only; none is persisted as a human label.

Overlap and containment suppression remove redundant views of the same local
evidence. Selection applies fixed discovery-stratum targets and caps each paper
at four samples and each page at two. Crop context is bounded, preserves native
geometry, and is neither resized nor written back to the canonical page.

The selected discovery distribution (not ground truth) is:

| Discovery stratum | Candidates |
| --- | ---: |
| Chromatic-ink risk | 12 |
| Margin/score risk | 10 |
| Tick/cross/correction risk | 10 |
| Mixed teacher-context risk | 9 |
| Ambiguous-mark risk | 2 |
| Hard-negative control | 5 |
| **Total** | **48** |

Discovery produced 72,807 raw component proposals and retained 10,406 after
overlap suppression. The final pool has no exact duplicate crop. Large proposal
counts are expected from a deliberately high-recall component scan; only the
fixed 48 derived crops are presented for annotation.

## Private storage and human workflow

All artifacts remain beneath ignored local storage:

```text
data/evaluation/evidence_teacher_v1/
  benchmark_manifest.json
  candidate_discovery_provenance.json
  labeling_metadata.json
  samples/
  overlays/
  annotations.json  # created by human saves
```

The neutral overlay says `CANDIDATE REGION - HUMAN REVIEW`; it never says that
a teacher mark was detected. The labeler binds only to `127.0.0.1`, uses no
external service, and begins with all 48 samples pending and no preselected
class or accepted answer rectangle.

Launch it from the repository root:

```bash
uv run python -m scripts.annotate_evidence --dataset evidence-teacher-v1
```

Open `http://127.0.0.1:8765/`, inspect every crop and overlay, choose the class,
draw student-answer rectangles or explicitly verify empty, then save. Candidate
statistics must not be interpreted as label targets.

Validate the private preparation safely with:

```bash
uv run python -m scripts.prepare_teacher_evidence validate
```

## Privacy, integrity, and limitations

Private provenance contains source paths and geometry; it is Git-ignored and is
never printed by the labeler. Public documentation records only safe aliases
and aggregates. Canonical SHA-256 hashes are checked before and after discovery
and materialization. Raw PDFs, canonical renders, existing annotations,
snapshots, OCR ground truth, and prior benchmark results are not modified.

The pool is drawn from 12 papers already available through the evidence-v2
canonical rendering set rather than creating new canonical renders. Discovery
signals can include student strokes, printed glyphs, scan artifacts, and
ambiguous marks. Human labeling may therefore yield fewer teacher examples than
the candidate strata suggest. Phase 4C.5C stops before freezing or evaluation.

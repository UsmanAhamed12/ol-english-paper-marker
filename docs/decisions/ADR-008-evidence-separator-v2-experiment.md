# ADR-008: Preserve V1 after the deterministic V2 experiment

## Status

Accepted

## Context

V1 has low answer-region recall and allows teacher evidence into student-answer
regions. Phase 4C.6 evaluated whether grouped components, richer local features,
explainable class scores, explicit answer spaces, and teacher-risk suppression
could improve this deterministically across three frozen human benchmarks.

## Decision

Keep V1 available and implement V2 only as an explicit parallel experiment.
Require a V2 candidate to be no worse than V1's 3/13 teacher-to-student errors
before selection. None of four predetermined variants passed that gate, so do
not promote V2 for downstream extraction. Retain immutable experiment records
and redesign candidate generation/grouping in a separately authorized phase.

## Alternatives considered

- Replace V1 based on higher student recall or aggregate accuracy.
- Tune V2 thresholds after inspecting each benchmark.
- Move directly to a learned classifier or vision model.

## Consequences

V1 behavior remains reproducible and downstream code is not silently changed.
The rejected V2 still provides typed features, explanations, overlays, and
controlled evidence for the next redesign. No current separator is approved for
student-only extraction; additional deterministic redesign is required before
considering a learned approach.

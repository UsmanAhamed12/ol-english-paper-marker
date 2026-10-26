# ADR-009: Use a minimal post-V1 extraction safety filter

## Status

Accepted for the FYP prototype

## Context

V1 is imperfect but reproducible. The research-oriented V2 redesign increased
teacher-to-student contamination from 3/13 to 10/13 and fragmented answer
regions. Project time is limited, while downstream extraction needs explicit
safe, uncertain, and empty outcomes.

## Decision

Keep V1 and add a thin deterministic `StudentCandidateSafetyFilter`. Reject
obvious guides, borders, print, and noise; quarantine strong teacher-risk
candidates; build answer regions only from safe evidence; and preserve
`REVIEW_REQUIRED` and `EMPTY` as valid outcomes. Accept the measured filter for
the academic prototype because teacher contamination falls to 2/13 without a
catastrophic combined student-recall collapse.

## Alternatives considered

- Promote the rejected V2 experiment.
- Continue deterministic separator redesign and benchmark expansion.
- Train or introduce a learned segmentation model.
- Use raw V1 without a safety boundary.

## Consequences

The project can proceed to structured extraction with fewer unsafe and empty
regions, bounded output volume, provenance, and explicit review flags. Some
student evidence is lost and teacher leakage remains possible, so the result is
appropriate for an FYP prototype rather than unattended production grading.

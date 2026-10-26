# ADR-014: Single paper-marking orchestrator

## Status

Accepted for the FYP prototype.

## Context

The repository has independently tested ingestion, OCR, structure, safe answer
extraction, marking-scheme ingestion, alignment, and objective/subjective
marking components. A UI or CLI caller should not coordinate these manually,
and unresolved evidence must survive the complete workflow.

## Decision

Introduce an injectable `PaperMarkingService` as the single application-level
boundary. Production pipeline adapters compose existing services. The service
adds stage timing, safe error mapping, review-queue projection, provenance, and
one `FinalPaperMarkingResult`; it does not duplicate domain marking logic.
`CombinedMarkingService` exposes objective, subjective, and combination steps
so each is executed exactly once and measured independently.

## Alternatives considered

- Put coordination in the CLI or future Streamlit pages: rejected because it
  would duplicate business workflow and make testing difficult.
- Introduce LangGraph now: outside Phase 6 and unnecessary for the FYP demo.
- Rewrite completed services into one pipeline class: rejected because it would
  weaken established boundaries and provenance.

## Consequences

The complete workflow is callable, injectable, deterministic outside local OCR
and model adapters, and ready for a presentation layer. The current execution
is sequential and in-process. Persistent workflow state, resumability, and a
real official-scheme smoke remain future concerns.

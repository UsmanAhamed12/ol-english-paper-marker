# ADR-013: Guarded local subjective marking

## Status

Accepted for the FYP prototype with mandatory review fallbacks.

## Context

Phase 5B can safely score explicit deterministic criteria but cannot assess
content, language, format, or holistic rubrics. The project must remain local,
auditable, and practical on an M1 laptop. OCR and model output can be uncertain,
and a generative model must not control final scores directly.

## Decision

Use a replaceable `SubjectiveMarkingProvider` with a loopback-only Ollama
adapter. One versioned rubric-first prompt requests strict JSON. Pydantic plus
deterministic ownership, confidence, and mark-cap checks gate every result.
Phase 5B remains unchanged, and `CombinedMarkingService` merges confirmed marks
while preserving unresolved capacity. Unsafe states route to human review.

## Alternatives considered

- Deterministic-only grading: cannot cover subjective criteria.
- Direct unvalidated model prose: unsafe and unauditable.
- Cloud APIs: violate the local-first privacy requirement.
- LangChain, LangGraph, RAG, fine-tuning, or a new model study: unnecessary for
  the current FYP milestone.

## Consequences

The prototype gains a complete typed subjective path and explainable feedback
without weakening objective marking. It adds local inference latency and model
variability. Strict validation and review fallbacks may leave marks unresolved,
and real marking accuracy still requires a later controlled evaluation.

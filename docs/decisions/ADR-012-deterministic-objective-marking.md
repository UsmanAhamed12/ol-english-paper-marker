# ADR-012: Deterministic objective and short-answer marking

Status: Accepted

## Context

Phase 5A provides structured criteria and exact Test-number alignment. The FYP
needs an auditable numerical path before subjective model grading is added.
Exact answers, explicit variants, keywords, and allocated components can be
evaluated without generative inference, while essays and language quality
cannot.

## Decision

Use a framework-independent `DeterministicMarkingEngine` with explicit
criterion-type dispatch, conservative comparison-only normalization, Decimal
arithmetic, and immutable results. Automatically score only exact answers,
accepted variants, token/phrase keywords, and explicitly allocated multi-part
components. Keep review, OCR failure, missing structure/scheme data, and
subjective criteria unresolved instead of assigning zero.

## Alternatives considered

- Fuzzy or semantic matching: rejected because it can silently infer answers
  and is difficult to audit.
- One LLM grader for every criterion: deferred because objective criteria do
  not need model uncertainty or runtime cost.
- Treat every unresolved criterion as zero: rejected because extraction or
  scheme failure is not evidence of an incorrect answer.

## Consequences

Objective marking is fast, local, deterministic, explainable, and testable.
Confirmed marks are cleanly separated from review-required and unsupported
capacity. The tradeoff is conservative recall: OCR spelling errors, synonyms,
and all subjective criteria remain unresolved for later controlled handling.

# ADR-003: Use a replaceable OCR provider abstraction

## Status

Accepted

## Context

The corpus mixes printed prompts, difficult handwriting, corrections, and
teacher annotations. The quality of local OCR and vision approaches is unknown,
and Phase 4 may need to compare several strategies without changing downstream
application code.

## Decision

Application OCR orchestration depends on a typed `OCRProvider` protocol. A
provider returns validated raw extraction evidence; `OCRService` adds
normalization and complete provider/page provenance. Production providers will
be introduced only when evaluated.

## Alternatives considered

- Couple the service directly to one OCR engine or model: simpler initially,
  but makes experiments, replacement, and deterministic unit testing costly.
- Let providers return arbitrary dictionaries: flexible, but weakens validation
  and provenance guarantees.

## Consequences

- The architecture has an additional small contract and result type.
- Providers are replaceable and testable without real OCR in unit tests.
- Downstream consumers receive one stable, validated result model.
- Provider-specific capabilities must be adapted to shared confidence, warning,
  and provenance semantics.

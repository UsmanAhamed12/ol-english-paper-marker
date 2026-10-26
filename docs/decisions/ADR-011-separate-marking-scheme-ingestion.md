# ADR-011: Use a separate typed marking-scheme ingestion boundary

## Status

Accepted for the FYP prototype

## Context

Official marking schemes are predominantly printed rubric documents, while the
student pipeline handles mixed handwriting, printed questions, and teacher ink.
Reusing student evidence separation would apply the wrong assumptions. Future
marking also needs source-traceable criteria and exact Test ownership.

## Decision

Create a separate `MarkingSchemeIngestionService` and immutable marking-scheme
models while reusing PDF and OCR infrastructure. Parse only explicit structures,
preserve ambiguity as `UNKNOWN`, and align student and scheme records solely by
Test number.

## Alternatives considered

- Treat the official scheme as another student paper.
- Store one unstructured OCR string and defer all interpretation to an LLM.
- Infer missing Tests, criteria, or marks heuristically.
- Introduce RAG or LangChain during ingestion.

## Consequences

The marking engine receives deterministic, auditable Test-owned criteria and
explicit missing/review states. Parsing stays local and testable. Complex or
implicit rubric content may remain unknown and require later review; no real
scheme PDF was available for this phase's smoke test.

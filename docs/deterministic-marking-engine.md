# Deterministic marking engine

## Status and scope

Phase 5B implements the first numerical marking path for the FYP prototype.
It consumes exact-number `AlignedTest` records and automatically scores only
criteria whose comparison rules are explicit and reproducible. It does not use
an LLM, embeddings, semantic similarity, fuzzy matching, or network services.

No private official marking-scheme PDF is currently available, so this phase is
validated with clearly synthetic fixtures. That verifies engine behavior, not
real examination marking accuracy.

## Architecture

```text
AlignedPaper / AlignedTest
  -> DeterministicMarkingEngine
  -> criterion-type matcher
  -> CriterionMarkResult[]
  -> TestMarkResult
  -> PaperMarkingResult
```

The engine reuses the Phase 5A criterion and alignment contracts and the Phase
4D student-answer status. It never reparses OCR text or invents Test ownership.

## Automatically supported criteria

- `EXACT_ANSWER`: full marks only when the comparison copy equals the one
  explicit expected answer.
- `ACCEPTED_VARIANTS`: full marks when the comparison copy equals one of the
  explicit variants; the matched variant is retained.
- `KEYWORD`: whole-token or contiguous-phrase matching against explicit
  keywords. A keyword such as `in` does not match `interesting`.
- `MULTI_PART`: each component is evaluated only when its expected evidence and
  marks are explicitly allocated.

Partial credit is never inferred. It is available only from explicit component
allocations, including explicit per-keyword components. Otherwise a criterion
is full-or-zero, or unresolved if its required allocation is missing.

## Comparison normalization

`marking-normalization-v1` creates a comparison copy using NFC Unicode,
normalized line endings, collapsed whitespace, case folding, and removal of
limited punctuation at the outer boundary. Internal punctuation is retained.
The original student OCR text is preserved in `StudentEvidenceReference` and
is never rewritten, spell-corrected, grammar-corrected, or replaced.

## Status and fallback policy

Automatic outcomes use `MARKED` or `NOT_ATTEMPTED`. The engine preserves
uncertainty with `REVIEW_REQUIRED`, `OCR_FAILED`, `SCHEME_INCOMPLETE`, and
`STRUCTURE_INCOMPLETE`. Subjective or unknown criteria use
`UNSUPPORTED_CRITERION`; they are not assigned zero.

An explicitly `EMPTY` answer receives zero only for objectively supported
criteria. An OCR failure is never interpreted as a wrong answer. For a
`REVIEW_REQUIRED` objective answer, comparison evidence may be recorded, but
`awarded_marks` remains unset and the result is not final.

Phase 5B deliberately leaves `CONTENT_POINT`, `LANGUAGE_QUALITY`,
`FORMAT_REQUIREMENT`, `HOLISTIC`, and `UNKNOWN` unresolved. Phase 5C now fills
the first four through a separate guarded local provider; `UNKNOWN` remains
unsupported. The deterministic engine itself is unchanged.

## Score safety and semantics

All marks use `Decimal`. Pydantic validation and aggregation enforce criterion,
Test, and known paper maxima. Missing allocation is preserved as unresolved
capacity rather than silently discarded.

`PaperMarkingResult.confirmed_marks` means only marks established by current
deterministic rules. `review_required_marks` and `unsupported_marks` remain
separate. Therefore a partially automated result such as 42 confirmed with 18
unresolved is not presented as a definitive 42/100 score.

Every criterion retains a decision, stable reason codes, scheme criterion ID,
source page, engine/normalization/matcher versions, and original student
evidence reference. Canonical JSON is deterministic and may be written only
beneath explicitly approved ignored private storage.

## Synthetic smoke

Run:

```bash
uv run python -m scripts.evaluate_marking_engine
```

The command evaluates six synthetic criteria and prints aggregate counts only.
It contains no private paper, transcription, identity, or official scheme.

## Limitations

- Real-scheme parsing and marking accuracy remain unmeasured because no
  official local marking-scheme PDF is available.
- Exact/variant matching deliberately provides no fuzzy tolerance for OCR
  mistakes.
- Keyword rules cannot infer synonyms or omitted concepts.
- Component matching depends on explicit component allocations.
- Subjective accuracy against official human marks remains unmeasured.

See `docs/subjective-marking.md`. The accepted next phase is **Phase 6 —
End-to-End Marking Orchestration**. Phase 6 now invokes this engine exactly once
and retains its awards unchanged inside the combined final result.

# Guarded subjective marking

## Status and scope

Phase 5C adds a local, rubric-guided path for `CONTENT_POINT`,
`LANGUAGE_QUALITY`, `FORMAT_REQUIREMENT`, and `HOLISTIC` criteria. It is an FYP
prototype boundary with mandatory validation and review fallbacks, not evidence
of research-grade marking accuracy. Phase 5B remains the sole owner of exact,
variant, keyword, and multi-part criteria. `UNKNOWN` is never sent to a model.

## Architecture

```text
AlignedTest + StructuredStudentAnswer + MarkingCriterion
  -> SubjectiveMarkingService
  -> SubjectiveMarkingProvider
  -> local Ollama adapter
  -> exact raw JSON response
  -> strict Pydantic and ownership validation
  -> SubjectiveCriterionMarkResult

DeterministicMarkingEngine + SubjectiveMarkingService
  -> CombinedMarkingService
  -> CombinedPaperMarkingResult
```

The provider Protocol keeps unit tests offline and allows later replacement.
`app.llm.factory` constructs the maintained `langchain_ollama.ChatOllama`
integration with the configured loopback endpoint, JSON mode, temperature 0,
seed 0, a bounded output length, and the configured request timeout.
`OllamaSubjectiveMarkingProvider` preserves the exact raw response and maps
local transport failures to sanitized domain errors. Shared Ollama validation
rejects remote endpoints and cloud model tags.

## Rubric-first prompt

The controlled prompt is `prompts/grading/subjective-v2.txt`, versioned as
`subjective-marking-v2`. It separates an authoritative criterion JSON object
from untrusted student evidence JSON. It instructs the model to apply only that
criterion, respect its maximum, distinguish OCR uncertainty from student error,
and return one specified JSON object. Student instructions are explicitly data,
not commands; the model must not follow prompt injection embedded in an answer.

LangChain is used only as the local chat integration. This path does not correct
or rewrite student answers, retrieve rubric material, or run an autonomous
chain or LangGraph workflow.

## Validation and score safety

The response is parsed into `SubjectiveModelAssessment` with additional fields
forbidden. Validation requires matching Test, criterion ID/type, and maximum;
known status; bounded confidence; required reason/feedback fields; and marks in
`[0, maximum]`. A model award above the rubric maximum is rejected, never
clamped. Malformed or mismatched responses become `INVALID_MODEL_RESPONSE` for
that criterion while other Tests continue.

Only a validated `MARKED` result at or above the configured grading-confidence
threshold becomes a confirmed subjective mark. Lower confidence, an explicit
model review outcome, low OCR confidence, or uncertain extraction remains
`REVIEW_REQUIRED`. OCR failure is `OCR_FAILED`, not zero. A verified empty
answer receives a deterministic zero for supported subjective criteria without
calling a model. Missing allocation is `SCHEME_INCOMPLETE`.

## Criterion behavior

- `CONTENT_POINT`: the supplied description defines the required ideas; matched
  and missing requirements are retained.
- `LANGUAGE_QUALITY`: only language dimensions stated by the criterion may be
  assessed. OCR uncertainty prevents a definitive mark.
- `FORMAT_REQUIREMENT`: only format evidence present in the extracted answer may
  be assessed. Missing layout evidence should cause review.
- `HOLISTIC`: the complete Test answer is assessed only against the supplied
  rubric and maximum.

Concise criterion-level feedback, confidence, reason codes, rubric/student
references, prompt/provider/model versions, duration, and raw model response
are retained for audit and later examiner review.

## Combined score semantics

`CombinedMarkingService` runs Phase 5B unchanged, sends only eligible subjective
criteria to Phase 5C, and aggregates with `Decimal`. The result separates:

- confirmed objective marks;
- confirmed subjective marks;
- review-required marks;
- unsupported marks;
- known maximum marks.

A definitive `final_score` exists only when every mark-bearing criterion is
resolved. Otherwise the confirmed total is explicitly partial and the final
score is `None`.

## Development smoke

The default command uses a deterministic fake and four synthetic criteria:

```bash
uv run python -m scripts.evaluate_subjective_marking
```

An already-installed local model can be checked without private data:

```bash
uv run python -m scripts.evaluate_subjective_marking \
  --ollama-model llama3.2:latest
```

The command prints safe aggregate statuses and reason counts only. It is an
integration smoke, not a marking-accuracy evaluation.

The Phase 5C local smoke used installed `llama3.2:latest` with four synthetic
criteria. All four responses passed structural validation and were safely
routed to `REVIEW_REQUIRED` for low model confidence; none became a confirmed
mark, none violated a mark cap, and total sequential provider time was about
31 seconds. This demonstrates the local integration and guardrail behavior,
not subjective marking accuracy.

Phase 11 validated the LangChain `ChatOllama` path with `llama2:latest` using
the same four synthetic criteria. All four responses passed structural
validation and were accepted as marks, with zero provider failures or mark-cap
violations; total sequential inference time was about 608 seconds. This is an
integration smoke only and is not evidence of marking accuracy.

## Privacy and limitations

The Ollama endpoint must be loopback-only. No student data is sent to a cloud
service. Private raw responses and result JSON belong only under ignored
runtime/private storage.

The system has not been evaluated against a real official scheme and human
subjective marks. Model confidence is an uncalibrated workflow signal. OCR may
lose wording or layout needed for language/format criteria. The guarded review
fallback therefore remains part of the accepted FYP design.

Phase 6 now invokes this service behind `PaperMarkingService`, aggregates its
existing feedback, and maps all unresolved criteria into an application-level
review queue without another model call.

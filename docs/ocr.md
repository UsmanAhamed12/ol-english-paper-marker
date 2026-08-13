# OCR architecture

## Phase 3 status

Phase 3 implements provider-independent OCR contracts and orchestration. It does
not implement or select an OCR engine, recognize handwriting, or make any claim
about OCR accuracy.

The implemented flow is:

```text
PaperPage
    -> OCRService
    -> OCRProvider
    -> OCRExtraction (raw evidence)
    -> OCRNormalizer
    -> OCRPageResult (raw + normalized + provenance)
```

## Provider abstraction

`OCRProvider` is a Python `Protocol`. A provider exposes a stable `name`, an
optional `model_version`, and `extract_page(PaperPage)`. Application code
depends on this contract rather than an engine-specific SDK.

No production provider module exists yet. Deterministic fake providers live in
tests and prove that providers can be replaced without changing `OCRService` or
downstream result consumers.

## Result models and provenance

`OCRExtraction` is the raw provider response. It contains:

- the provider's exact `raw_text`;
- optional confidence;
- typed non-fatal warning codes;
- provider-measured processing duration.

`OCRPageResult` adds normalized text and durable provenance:

- internal paper UUID and one-based page number;
- absolute canonical source-image reference;
- provider name and optional model/version identifier;
- raw and normalized text side by side.

Results are immutable Pydantic models. Provenance is stored in the result, not
left only in logs.

## Raw and normalized text

Raw provider text is never overwritten. `OCRNormalizer` creates a separate
`normalized_text` value using only deterministic representation cleanup:

- Unicode NFC normalization;
- CRLF and CR conversion to LF;
- removal of spaces and tabs at line ends;
- removal of surrounding blank lines.

It does not correct grammar or spelling, infer missing words, rewrite sentences,
or replace uncertain tokens. For example, `He go to scool yesterday` remains
unchanged because student language is grading evidence.

## Confidence semantics

Confidence is either `None` or a float from 0.0 through 1.0. `None` explicitly
means the provider does not supply a meaningful confidence signal. The service
does not fabricate a value.

Even when present, confidence is provider-reported workflow metadata, not a
calibrated probability unless later evaluation demonstrates calibration.
Providers may use different confidence definitions, so provider/model
provenance must accompany comparisons.

## Warnings

Providers can return typed, non-fatal warning codes for:

- low confidence;
- partial extraction;
- handwriting ambiguity;
- image-resolution concerns.

Warnings are preserved by `OCRService`. They are not exceptions and do not hide
provider failures.

## Service and error behavior

`OCRService.process_page()` verifies that the canonical rendered image exists,
invokes the provider, preserves raw evidence, normalizes a copy, and constructs
the final result. Provider exceptions and invalid provider output become
`OCRProviderError`. A missing rendered page or an unrendered document becomes
`OCRProcessingError`.

A successful extraction may legitimately contain an empty string. Provider
failure is always an exception and is never converted into an empty successful
result.

`process_document()` processes pages synchronously in `PaperDocument.pages`
order. It does not skip failed pages, retry, or introduce concurrency.

## Canonical and derived images

Phase 2 PNG files under
`data/runtime/<paper_id>/pages/page_XXXX.png` remain canonical and immutable to
OCR. Phase 3 performs no resizing, grayscale conversion, contrast adjustment,
deskewing, cropping, or denoising.

The representative Phase 2 page was approximately 3409 x 4932 pixels. Some
future providers may need provider-specific resizing, but larger input does not
automatically mean better OCR. Any future preprocessing must use an in-memory
copy or a separate derived artifact and must never overwrite the canonical PNG.

## Historical teacher-mark leakage

Historical evaluation scans flatten student writing, printed prompts, and
teacher marks into the same image. OCR may therefore extract teacher
annotations as if they were source text. Phase 3 does not remove or classify
teacher markings.

Paper/page identity, canonical image reference, provider/model identity, raw
text, warnings, and normalized text are retained so later evaluation and
filtering can trace exactly what was processed. This provenance prepares for,
but does not solve, the leakage problem. Normal runtime papers are expected to
be unmarked.

## Phase 4A benchmark foundation

Phase 4A now provides typed private benchmark manifests, optional pixel-region
metadata, deterministic CER/WER metrics, failure-preserving results, aggregation,
and an `OCRService`-based benchmark runner. The framework is documented in
[ocr-benchmark.md](ocr-benchmark.md). It adds no real OCR provider and declares
no winning model.

Phase 4B may implement and compare local provider adapters against
representative printed, handwritten, corrected, and low-quality pages.
Experiments must preserve raw outputs, record provider/model and prompt versions,
and select a provider from dataset evidence rather than popularity.

## Not implemented

- no real OCR or vision provider;
- no Ollama, LangChain, or LangGraph integration;
- no image preprocessing pipeline;
- no handwriting recognition claim;
- no teacher-mark removal;
- no retries, concurrency, persistence, segmentation, or grading.

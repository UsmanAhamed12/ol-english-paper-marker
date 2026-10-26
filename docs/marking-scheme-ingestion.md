# Marking-scheme ingestion and Test alignment

## Status

Phase 5A implements a deterministic marking-scheme domain boundary for the FYP
prototype. It structures official printed marking material and aligns it to
Phase 4D student answers by explicit Test number. Phase 5B now consumes that
alignment to award only safely deterministic marks.

No official marking-scheme PDF currently exists in the repository's private
data paths. The implementation is therefore verified with synthetic fixtures;
real-scheme ingestion and private student/scheme alignment smoke tests remain
pending until an official local PDF is supplied.

## Architecture

```text
Marking Scheme PDF
  -> PDFValidator / PDFLoader / PDFRenderer
  -> OCRService (configured Tesseract provider)
  -> MarkingSchemeIngestionService
  -> MarkingSchemeParser
  -> StructuredMarkingScheme
       -> MarkingSchemeTest[]
       -> MarkingCriterion[]

StructuredStudentPaper + StructuredMarkingScheme
  -> TestAlignmentService
  -> AlignedPaper / AlignedTest[]
```

The marking scheme is not passed through the student evidence separator or
teacher-risk safety filter. It has a separate application boundary while
reusing shared PDF, rendering, OCR, geometry, exception, and configuration
infrastructure.

## Domain models

`StructuredMarkingScheme` preserves a content-derived scheme UUID, source
document UUID, page count, ordered Tests, total marks only when all Test maxima
are known, complete page OCR evidence, warnings, duration, and versioned
provenance.

Each `MarkingSchemeTest` owns an explicit Test number, optional title, optional
maximum, ordered criteria, accepted answers when explicitly parsed, source
pages/regions, and warnings. Each `MarkingCriterion` has a deterministic ID,
structural type, source description, explicit accepted variants/keywords,
optional maximum, optional partial-credit indication, and parser provenance.

The initial structural criterion enum contains exact answers, accepted
variants, keywords, multi-part material, content points, language quality,
format requirements, holistic criteria, and `UNKNOWN`. These values do not yet
implement scoring behavior.

## OCR and provenance

The ingestion service runs each rendered page through the existing
`OCRService`. It retains raw and normalized page text, OCR words and bounding
boxes when available, provider/model identity, warnings, and durations. A page
OCR failure is recorded without fake text, and remaining pages continue.

Every Test and criterion retains its page, optional source rectangle, OCR word
indices, raw source line, and deterministic parser rule. Complete structured
JSON may contain private marking material and must remain in ignored runtime
storage.

## Conservative parsing rules

Test headings accept only anchored, explicit forms such as `Test 1`, `Test
01`, `TEST 01`, and `Test01`. Duplicate or decreasing Test markers fail rather
than being silently rearranged. No Test number is invented.

The parser recognizes only explicit structures:

- numbered or lettered answers;
- slash-separated accepted variants;
- labeled accepted answers or keywords;
- labeled content, language, format, or holistic criteria;
- explicit `(1 mark)`, `[2 marks]`, or `Total: N marks` allocations.

Unrecognized non-empty material is preserved as `UNKNOWN`. A Test maximum is
the explicit total, or the sum only when every parsed criterion has an explicit
allocation. Otherwise it remains `None` with a warning. The paper total exists
only when every Test maximum is known.

## Alignment policy

`TestAlignmentService` uses Test number as the sole key. It does not use fuzzy
or semantic matching. The union of student and scheme Test numbers is ordered
numerically and represented with one of:

- `ALIGNED`;
- `STUDENT_TEST_MISSING`;
- `SCHEME_TEST_MISSING`;
- `STUDENT_REVIEW_REQUIRED`;
- `STUDENT_OCR_FAILED`;
- `STRUCTURE_INCOMPLETE`.

A review-required student answer still retains its structural relationship to
the matching scheme Test. Missing Tests and criteria are never manufactured.

## Serialization, CLI, and privacy

Scheme and alignment JSON use stable key ordering and compact canonical JSON.
The writer requires an explicit private root and writes atomically.

When an official local marking scheme becomes available, run:

```bash
uv run python -m scripts.ingest_marking_scheme /private/path/to/scheme.pdf
```

The command prints only page/Test/criterion/maximum/warning counts and duration;
complete OCR and parsed content remains under ignored
`data/runtime/marking-schemes/` storage. No external service is used.

## Limitations and decision

- Real marking-scheme layout has not yet been smoke-tested locally.
- Tesseract errors may cause UNKNOWN criteria or missing markers.
- Complex tables, multi-column rubrics, implicit marks, and prose alternatives
  may require manual review.
- `UNKNOWN` preserves source evidence but cannot be scored automatically yet.
- Alignment proves structural ownership, not answer correctness.

Phase 5B also permits explicit `CriterionComponent` allocations for multi-part
or per-keyword partial credit. Neither ingestion nor marking invents component
marks when the source does not provide them.

Phase 5A remains accepted with UNKNOWN/warning fallback. Phase 5B consumes the
alignment contract for deterministic criteria, and Phase 5C consumes the same
exact-number contract for guarded subjective criteria. `UNKNOWN` remains
unsupported rather than being guessed.
Phase 6 now constructs this ingestion service inside the production-local
orchestration factory and preserves scheme warnings through final results.

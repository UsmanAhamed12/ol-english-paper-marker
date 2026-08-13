# OCR benchmark foundation

## Phase 4A status

Phase 4A provides a reproducible evaluation boundary for comparing future local
OCR providers. It does not implement OCR, run a vision model, create ground
truth automatically, or select a winning provider.

The benchmark flow is:

```text
private manifest + rendered input
              -> OCRBenchmarkRunner
              -> OCRService
              -> OCRProvider
              -> student-text prediction
              -> CER and WER against human transcription
              -> result + aggregate summary
```

## Why benchmark our dataset

O/L English scans contain printed questions, varied handwriting, crossed-out
work, and flattened teacher marks. General model rankings do not establish how
well a provider transcribes the student-answer evidence needed for this system.
Phase 4B must compare candidates on representative local samples using the same
manifest, prompt version, scoring policy, and hardware.

The intended initial candidates are DeepSeek-OCR, Qwen3-VL 4B, and Gemma 3 4B
through Ollama. They are candidates only; Phase 4A makes no quality claim and
does not install Ollama or any model.

## Sample strategy

The ignored local manifest at
`data/evaluation/ocr/benchmark_manifest.json` contains eight candidate slots
covering:

- clear, average, and difficult handwriting;
- short answers and paragraph writing;
- student corrections and cross-outs;
- teacher-annotation contamination risk;
- dense and sparse pages;
- full-page and region-level inputs.

Safe paper aliases and sample IDs are used instead of original filenames or
student names. A human curator must confirm that every selected page or region
actually matches its category before marking it ready. The current candidate
entries are pending manual transcription and cannot be scored by the runner.

## Manifest and ground truth

`BenchmarkManifest` is versioned and requires unique, safe sample identifiers.
Each `OCRBenchmarkSample` records:

- a safe paper alias and one-based page number;
- an image reference and page dimensions;
- optional pixel region;
- difficulty and representative categories;
- whether printed content and teacher annotations are present;
- the fixed target `student_answer_text`;
- manual-ground-truth status and student transcription;
- non-identifying curator notes.

Ground truth must be manually transcribed by a human from the student answer.
An LLM or OCR model must never create or repair benchmark ground truth. Printed
question text and teacher marks are excluded. Student spelling and grammar
errors remain exactly as written. A verified transcription may legitimately be
an empty string only with the explicit `human_verified_empty` state; `null`
means transcription is pending. A blank field never implies a verified-empty
target.

Phase 4A.2 freezes the authoritative human worksheet into the ignored private
manifest. `human_verified` requires non-empty manually entered text, while
`human_verified_empty` requires an explicit human decision and an empty string.
The transfer preserves spelling, grammar, capitalization, punctuation, and
internal newlines. Non-text visual marks can be retained in notes without being
added to CER/WER ground truth.

The frozen manifest has a deterministic SHA-256 fingerprint over a versioned,
canonical representation of sample ID, verification state, and student text.
This detects accidental changes between experiments; it is not encryption or a
privacy control. The validation command reports only safe counts, readiness,
and the fingerprint, never student text.

The committed JSON fixture under `tests/fixtures/` contains synthetic text only
and demonstrates the schema without disclosing student data.

## Region support

Regions use page-pixel coordinates: `x`, `y`, `width`, and `height`. Coordinates
must be non-negative, dimensions positive, and the rectangle contained within
the recorded page dimensions. A missing region means full-page evaluation.

Phase 4A.1 materializes each region as a derived private PNG for manual
transcription. It uses the existing Phase 2 ingestion and rendering boundary
when a canonical page is absent, validates coordinates against actual rendered
dimensions, and verifies that canonical images remain unchanged. Full-page and
region results retain distinct sample IDs.

## Metric policy

CER and WER use Levenshtein edit distance. Metric normalization is deliberately
separate from `OCRNormalizer` and never changes stored OCR evidence or manual
ground truth. For comparison only, it:

1. normalizes Unicode to NFC;
2. collapses all whitespace runs to a single ASCII space;
3. removes surrounding whitespace as a consequence of that collapse.

Case and punctuation remain significant. CER treats every character in the
normalized string, including interior spaces, as a unit. WER splits the
normalized string on spaces and treats each resulting word token as a unit.

Each `ErrorRate` stores edit errors, reference units, and the rate. Edge cases
are explicit:

| Reference | Prediction | Rate |
| --- | --- | --- |
| non-empty | empty | `1.0` |
| empty | empty | `0.0` |
| empty | non-empty | undefined (`null`) |

An empty-reference/non-empty-prediction still records insertion errors, but no
rate is invented because the denominator is zero. Undefined rates are excluded
from mean and median calculations and remain visible in individual results.

## Results and aggregation

Every result records sample ID, provider and model version, OCR prompt version,
status, the raw provider prediction, CER, WER, duration, OCR warnings, optional
manual teacher-annotation contamination assessment, and a safe error message.
Metric normalization operates on a comparison copy; the stored prediction is
not rewritten to improve a score.

A provider failure becomes a failed result rather than terminating the entire
experiment. It has no prediction or metrics and remains counted. Summaries
report:

- total, successful, and failed samples;
- mean and median CER;
- mean and median WER;
- mean successful processing duration.

## Teacher-annotation contamination

Teacher marks are not student ground truth. A reviewer can record
`teacher_annotation_contamination` on a result as `true`, `false`, or `null`
when not yet assessed. Phase 4A does not automatically detect ink color or
separate annotations. Phase 4B reports should include contamination alongside
accuracy because a low CER obtained by transcribing the wrong content is not a
valid success.

## Privacy

`data/evaluation/` is ignored by Git. Real manifests, transcriptions, derived
crops, predictions, and reports remain local. Do not store original filenames,
student names, admission numbers, or other identifiers in aliases or notes.
Never commit benchmark screenshots or crops from historical papers.

Validate the private manifest without running OCR:

```bash
uv run python -m scripts.benchmark_ocr validate
```

The command prints only schema version, human-verified/verified-empty/pending
counts, readiness, and the ground-truth fingerprint. It does not print
transcription content, paths, aliases, or notes, and reports
`ocr_executed: false`.

Prepare or refresh the ignored sample images and create the blank private
transcription worksheet:

```bash
uv run python -m scripts.prepare_ocr_benchmark
```

The helper reads the ignored safe-alias source mapping, renders missing pages
through `PDFValidator`, `PDFLoader`, and `PDFRenderer`, and writes deterministic
`sample_001.png` style filenames under `data/evaluation/ocr/samples/`. It is
idempotent: images are reproducibly refreshed, while an existing worksheet is
never overwritten so later human transcription is preserved. It performs no
OCR and prints no original source filenames.

## Phase 4B selection criteria

Provider selection will prioritize evidence in this order:

1. student-handwriting CER and WER;
2. provider failure rate;
3. teacher-annotation contamination;
4. latency;
5. memory practicality on the MacBook Air M1;
6. output stability across repeated deterministic runs.

Accuracy dominates. Speed, model size, or popularity alone cannot select a
provider. Comparisons must record exact provider/model and OCR prompt versions.

## Current limitations

- no real provider has been implemented or benchmarked;
- all private candidate samples still require human selection confirmation and
  transcription;
- teacher contamination is manually assessed;
- region coordinates and candidate categories still require human visual
  confirmation before benchmarking;
- CER and WER do not measure semantic equivalence or layout quality;
- the small benchmark will estimate comparative performance, not prove broad
  population-level accuracy.

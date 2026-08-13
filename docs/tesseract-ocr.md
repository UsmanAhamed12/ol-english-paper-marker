# Tesseract OCR baseline

## Phase 4C.1 status

Phase 4C.1 integrates plain Tesseract 5.5.2 as a local `OCRProvider` and
evaluates it on the frozen eight-sample benchmark. The baseline uses canonical
PyMuPDF PNG images without preprocessing. It does not classify authorship,
segment questions, correct text, or grade answers.

## Why evaluate classical OCR

The Phase 4B Qwen3-VL 4B run was slow on the MacBook Air M1 and inaccurate for
student-only transcription. Tesseract provides a lightweight deterministic
comparison with word boxes, hierarchy, and engine confidence.

## Architecture and dependency

```text
PDFValidator -> PDFLoader -> PDFRenderer (PyMuPDF) -> PaperPage
    -> OCRService -> TesseractOCRProvider -> OCRExtraction -> OCRPageResult
```

The Python adapter is `pytesseract`; it requires the local Tesseract executable:

```bash
brew install tesseract
uv sync
```

No `pdf2image`, Poppler rendering, OpenCV, or cloud service is used.

## Fixed baseline configuration

| Variable | Baseline | Meaning |
| --- | ---: | --- |
| `TESSERACT_LANGUAGE` | `eng` | English trained data |
| `TESSERACT_PSM` | `6` | one uniform text block assumption |
| `TESSERACT_TIMEOUT_SECONDS` | `120` | per-image deadline |

This configuration was selected before results and applied to every sample. A
timeout is an explicit provider failure, never empty success.

## Structured evidence and layout

Every non-empty word becomes an immutable `OCRWord` containing exact recognized
text, a `BoundingBox`, optional normalized confidence, and available
block/paragraph/line/word numbers. Output-column lengths are validated first.
Tesseract's structural `-1` confidence is treated as unavailable.

Valid confidence from 0 through 100 is divided by 100 for the generic unit
interval. It is an engine recognition signal, not a calibrated probability and
not evidence that a word is printed, student-written, or teacher-written.

Layout reconstruction orders hierarchy keys and words deterministically,
preserving lines and blank lines between paragraphs. It is approximate layout,
not pixel-perfect reconstruction. No spelling, grammar, capitalization, or
student mistake is corrected.

## Frozen benchmark results

Ground-truth fingerprint:
`33a5dc8e46a1cf0631d46da41a8490c4ec10a18194591144425422c61ff73f9a`.
Private results remain under
`data/evaluation/ocr/results/tesseract-baseline/`.

| Metric | Tesseract 5.5.2 | Qwen3-VL 4B |
| --- | ---: | ---: |
| Successful samples | 8/8 | 7/8 |
| Mean CER | 6.4043 | 1.0000 |
| Median CER | 6.0670 | 1.0000 |
| Mean WER | 8.2470 | 1.0000 |
| Median WER | 8.9847 | 1.0000 |
| Mean successful duration | 1.961 s | about 393.447 s |
| Median successful duration | 1.906 s | not recorded in Phase 4B |
| Empty successful predictions | 0 | 5 on non-empty references |
| Verified-empty behavior | non-empty on 2/2 | non-empty on 2/2 |

Lower error is better. Tesseract was about 200 times faster, but student-only
error was worse because plain OCR also recovered printed content and flattened
marks. It is useful as fast layout evidence, not a selected handwriting OCR
solution.

## Real-paper structural smoke

One representative private PDF was loaded and rendered through Phase 2. Its
first canonical page yielded 246 words in about 2.30 seconds and two strings
matching a conservative `Test NN` pattern. Only safe counts were reported.

## Limitations and privacy

Historical pages combine print, student handwriting, and teacher marks. This
phase deliberately makes no authorship decision. Low confidence can mean poor
print, noise, handwriting, tables, or scan degradation. Color identifies
neither author nor semantic role reliably. Teacher contamination and
handwriting quality therefore remain unresolved.

Private images, predictions, evidence, and ground truth stay under ignored
evaluation/runtime storage. Nothing is sent off-device.

Phase 4C.2 separately measured four fixed OpenCV variants. All regressed against
this official baseline, which remains unchanged and selected. See
[ocr-preprocessing.md](ocr-preprocessing.md). Deskewing and sample-specific
tuning remain out of scope.

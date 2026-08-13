# OCR preprocessing experiment

## Phase 4C.2 outcome

Phase 4C.2 evaluated four fixed, conservative OpenCV preprocessing variants
against the unchanged eight-sample OCR benchmark. None improved the official
plain-Tesseract baseline across the aggregate CER/WER measures, so no
preprocessing variant was adopted. The selected production baseline remains
`none`.

## Purpose and immutable evidence

The experiment asked whether simple image cleanup could improve Tesseract while
preserving handwriting, print, teacher marks, page geometry, and local M1
practicality. Canonical `PaperPage.image_path` files are immutable evidence.
Preprocessing writes derived PNGs beneath ignored evaluation/runtime storage and
checks the source SHA-256 before and after every operation.

```text
canonical PaperPage
      -> ImagePreprocessor
      -> separate derived PNG + PreprocessingResult
      -> existing TesseractOCRProvider
      -> existing OCRService
```

`PreprocessingResult` records absolute source/derived paths, source and output
dimensions, ordered operations, and duration. Output must differ from the
source path and have identical dimensions. There is no crop, resize, rotation,
deskew, or coordinate transform.

## Dependency and algorithms

Phase 4C.2 adds `opencv-python-headless`; NumPy is its transitive dependency.
The predetermined variants were fixed before measurement:

| Variant | Operations |
| --- | --- |
| baseline | none |
| grayscale | BGR to grayscale |
| grayscale-denoise | grayscale, then 3 x 3 Gaussian blur |
| grayscale-threshold | grayscale, then global Otsu binary threshold |
| grayscale-denoise-threshold | grayscale, blur, then Otsu threshold |

Every sample used the same operation sequence. PSM 6, English trained data,
120-second timeout, regions, ground truth, CER/WER code, and metric normalization
were unchanged. Deskew was deferred because it adds a geometric transform and
the inspected samples did not justify that complexity.

## Frozen benchmark results

Ground-truth fingerprint:
`33a5dc8e46a1cf0631d46da41a8490c4ec10a18194591144425422c61ff73f9a`.
Negative deltas are improvements; every measured delta below is positive.

| Variant | Success | Mean CER | Median CER | Mean WER | Median WER | Mean OCR | Median OCR | Mean preprocess | Empty target output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 8/8 | 6.4043 | 6.0670 | 8.2470 | 8.9847 | 1.961 s | 1.906 s | 0 | non-empty 2/2 |
| grayscale | 8/8 | 6.6971 | 6.5624 | 8.9653 | 9.4717 | 1.240 s | 1.148 s | 0.230 s | non-empty 2/2 |
| grayscale-denoise | 8/8 | 6.7124 | 6.5904 | 8.7715 | 9.5606 | 1.339 s | 1.212 s | 0.240 s | non-empty 2/2 |
| grayscale-threshold | 8/8 | 6.4943 | 6.2290 | 8.6905 | 9.6051 | 0.881 s | 0.829 s | 0.141 s | non-empty 2/2 |
| grayscale-denoise-threshold | 8/8 | 6.4296 | 6.2946 | 8.5162 | 9.3895 | 0.888 s | 0.857 s | 0.141 s | non-empty 2/2 |

| Variant | Mean CER delta | Median CER delta | Mean WER delta | Median WER delta |
| --- | ---: | ---: | ---: | ---: |
| grayscale | +0.2929 | +0.4954 | +0.7183 | +0.4870 |
| grayscale-denoise | +0.3082 | +0.5233 | +0.5245 | +0.5760 |
| grayscale-threshold | +0.0900 | +0.1620 | +0.4434 | +0.6205 |
| grayscale-denoise-threshold | +0.0254 | +0.2276 | +0.2692 | +0.4048 |

The official Phase 4C.1 baseline was not rerun or replaced. Private predictions,
ground truth, and derived images remain under `data/evaluation/ocr/`.

## Visual QA and selection

A clear crop, a difficult-handwriting crop, and a dense full page were inspected
for each variant. Dimensions and boundaries were preserved; handwriting,
printed text, ruling, and table/line structures remained visible. Thresholding
strengthened scan-edge/background artifacts and removed color distinctions,
which may damage future evidence interpretation even where dark strokes remain.

Selected variant: **none**. All variants preserved success count but worsened
aggregate error and did not improve verified-empty contamination. Lower OCR
latency in thresholded variants does not outweigh worse accuracy and added
visual risk.

## Real-paper smoke

Because no variant won, one existing 3409 x 4932 canonical page was processed
with the unchanged baseline. It produced 216 detected words in 2.270 seconds,
zero conservative `Test NN` matches, and no preprocessing time. Full text and
identifying metadata were not printed. This smoke verifies safe execution only;
marker count depends on page content and is not a segmentation result.

## Limitations and next phase

CER/WER compares student-only human transcription against Tesseract output that
may also contain printed questions and teacher marks. Poor scores therefore
combine recognition error with unresolved content-selection error. Preprocessing
does not classify print, student writing, or teacher annotations, and confidence
or color must not be treated as authorship.

The small benchmark cannot prove population-level behavior. The next recommended
phase is Phase 4C.3, exam structure and Test segmentation, using preserved word
geometry rather than changing OCR evidence.

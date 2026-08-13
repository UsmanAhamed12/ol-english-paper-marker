# Dataset

## Phase 0 status

Phase 0 inspected the raw historical marked-paper corpus on 2026-08-13. No raw
PDF was modified, renamed, annotated, or deleted. This document describes
observations only; PDF ingestion, OCR, segmentation, grading, and evaluation are
not implemented yet.

## Location and handling rules

Raw papers are stored in `data/raw/marked_papers/`. They are immutable source
data and may contain student identifiers, handwriting, and teacher markings.
Some filenames also contain student names. Treat the directory as sensitive:

- do not edit or overwrite source PDFs;
- do not commit additional student data without explicit approval;
- do not expose names or admission numbers in logs, reports, fixtures, or UI;
- write derived artifacts only to the future processed, evaluation, or runtime
  locations;
- keep teacher markings isolated as evaluation ground truth and never provide
  them to the grading model during an accuracy evaluation.

## Corpus inventory

The inventory covered every PDF directly under `data/raw/marked_papers/`.

| Measure | Finding |
| --- | ---: |
| PDF files | 40 |
| Numeric filenames | 37 |
| Name-based filenames | 3 |
| Total pages | 393 |
| Total size | 533,891,866 bytes (about 509.2 MiB) |
| File-size range | 9,189,551-22,595,489 bytes |
| Median file size | 13,305,083 bytes |
| Encrypted PDFs | 0 |
| PDFs with any extractable text | 0 |

Page-count distribution:

| Pages per PDF | PDF count |
| ---: | ---: |
| 8 | 4 |
| 9 | 10 |
| 10 | 15 |
| 11 | 11 |

The minimum is 8 pages, the median is 10 pages, the maximum is 11 pages, and
the mean is approximately 9.82 pages.

## PDF and image characteristics

All 393 pages are image-only scans. Each page contains exactly one embedded,
full-page JPEG image; no page has an extractable text layer. Consequently,
normal PDF text extraction returns no usable exam or answer text, and OCR will
be mandatory in a later phase.

Embedded page-image measurements across the corpus are:

| Measure | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| Width | 1,477 px | 1,678 px | 3,154 px |
| Height | 1,993 px | 2,413 px | 4,083 px |
| Pixel count | 2.97 MP | 4.05 MP | 12.29 MP |

Poppler reports 72 x 72 PPI because the PDF page boxes are sized to the image
pixel dimensions. That is a PDF construction artifact, not a trustworthy
physical scan-resolution measurement. Pixel dimensions are therefore the
reliable Phase 0 measurement. Page aspect ratios and crop extents vary, even
within a single PDF, and every page has rotation metadata of zero despite small
visual skew in some scans.

## Representative visual inspection

Three numeric-ID papers were selected to avoid repeating name-based identifiers
in documentation and to cover the observed page-count and image-size range.
All pages in each selected paper were rendered and reviewed.

| Paper | Selection reason | Observed structure |
| --- | --- | --- |
| `4440.pdf` | 8-page minimum | 2022 Grade 10 English Language I and II printed question pages; answers written directly in the booklet; several longer-response areas left blank; red teacher corrections and subtotal arithmetic. |
| `4430.pdf` | 10-page median | 2022 Grade 10 English Language I and II; printed booklet pages followed by ruled-paper continuations for Tests 14 and 16; red ticks, corrections, criterion-like marks, and totals. |
| `4552.pdf` | 11-page maximum and largest file | Grade 11 English Language I and II papers with a different printed form/layout; direct booklet answers plus long-form writing on ruled response space; red per-item and per-section marks. |

The sample demonstrates more than one exam template and year/grade layout. A
paper generally combines two English Language parts. Printed sections use
labels such as `Test 01` through `Test 16`, while student-added continuation
pages may use handwritten labels such as `Test 14` or `Test 16`. One PDF page
cannot be assumed to equal one question: multiple short tests share pages, and
long answers can continue onto later or additional ruled pages.

### Handwriting characteristics

- Student writing ranges from compact answers in blanks to multi-paragraph
  responses on ruled or dotted lines.
- Dark blue, black, and pencil-like low-contrast strokes occur; character size,
  slant, spacing, and baseline alignment vary.
- Corrections include overwriting, crossings-out, insertions, arrows, and text
  squeezed around printed prompts.
- Some responses are blank or incomplete, so absence of writing must not be
  mistaken for an OCR failure.

### Teacher-annotation characteristics

- Teacher annotations are predominantly red and include ticks, crosses,
  underlines, corrections, marginal marks, circled scores, and arithmetic at
  page tops.
- Marks sometimes overlap student answers or printed lines and may use small
  numerals, fractions, and abbreviated notes.
- Teacher annotations and student responses are flattened into the same page
  image. There is no removable annotation layer.

This flattening creates a critical evaluation-leakage risk. Historical papers
cannot be supplied directly to a grading model while their teacher marks remain
visible. A later evaluation design must obtain unmarked source copies or create
and validate a masking/separation process before predictions are made; teacher
scores may be revealed only after each prediction is frozen.

## Observable OCR and segmentation challenges

- image-only PDFs with no searchable text;
- mixed machine print, handwriting, and red teacher markup in one raster;
- low-contrast pencil or faint pen strokes and occasional show-through;
- skew, perspective variation, uneven crops, page curvature, folds, shadows,
  and variable exposure;
- answers written on dotted lines, inside tables, next to pictures, and across
  dense printed passages;
- overwritten text, strike-throughs, symbols, fractions, arrows, and cramped
  insertions;
- multiple questions per page and answers that span or move to extra pages;
- inconsistent printed numbering formats (`Test 1`, `Test 01`, and handwritten
  continuation labels);
- blank response areas and omitted questions;
- different examination templates across the corpus.

OCR quality and grading quality must therefore be evaluated separately. Future
pipeline diagnostics should preserve the raw image and raw OCR output and
attribute errors to OCR, segmentation, retrieval, grading, or aggregation.

## Reproducing the structural inventory

Run the read-only inspection script from the repository root:

```bash
python3 scripts/inspect_dataset.py
```

The script requires Poppler's `pdfinfo`, `pdftotext`, and `pdfimages` commands.
It prints JSON containing aggregate measurements and per-file structural
metadata. It does not perform OCR, render derived data into the repository, or
modify the PDFs. Visual observations in this document still require human
review of rendered pages and are not inferred by the script.

## Phase 0 limitations

- The visual review is representative, not a page-by-page semantic audit of all
  393 pages.
- No OCR accuracy claim can be made because OCR has not been implemented or
  evaluated.
- No official marking scheme was identified or inspected in this corpus; the
  directory appears to contain marked student answer scripts.
- Physical scan DPI cannot be established reliably from current PDF metadata.
- Student identity, consent, retention, and access-control requirements require
  explicit privacy review in a later phase.

# Exam structure and Test segmentation

## Phase 4C.3 outcome

Phase 4C.3 adds a deterministic structure layer that converts Tesseract word
evidence into ordered, spatial `TestRegion` values. On a private, manually
verified benchmark of 3 papers, 29 pages, and 48 Test headings, it detected 44
headings with 1.0000 precision, 0.9167 recall, and 0.9565 F1. Four headings
remained explicitly missing; the detector did not manufacture replacements.

This is structure detection, not handwriting recognition, question extraction,
authorship classification, or grading.

## Architecture

```text
PaperPage
  -> OCRService
  -> TesseractOCRProvider
  -> OCRStructuredEvidence (words + hierarchy + boxes)
  -> ExamStructureDetector
  -> ExamPageStructure[] + TestRegion[]
  -> private benchmark metrics and visual overlays
```

The detector reuses the Phase 2 canonical render and Phase 3 OCR service. It
does not parse PDFs itself, rerun a different OCR model, or flatten evidence to
one regular-expression search over page text.

Immutable Pydantic models retain each candidate's raw marker text, page-local
bounding box, source word indexes, OCR confidence, text/numeric/geometry
signals, and detection strategy. Accepted markers add sequence support and a
safe normalized label such as `Test 06`. A `TestRegion` contains one or more
page-local boxes so a Test may share a page or continue across page boundaries.

## Marker detection

Tesseract hierarchy values group words into lines and Tesseract word ordering
is preferred over inferred pixel order. The detector accepts conservative
variants including:

- case differences and compact forms (`TEST 01`, `Test01`);
- explicit digit confusions in a one- or two-character number (`O`/`o` to
  zero and `I`/`l` to one);
- a small allowlist of observed keyword confusions and a one-edit fuzzy
  keyword match;
- isolated punctuation between the keyword and nearby number;
- a punctuation/noise token before a line-leading heading.

Candidate scoring combines keyword similarity (35%), numeric confidence (20%),
provider OCR confidence or a neutral missing value (20%), and geometry (25%).
Long sentence-like lines and footer positions receive explicit penalties. A
candidate must score at least 0.60 and have an expected number from 1 through
16. These scores are explainable ranking signals, not calibrated probabilities.

No large fuzzy-search dependency was needed; bounded edit distance is
implemented with the Python standard library. The rules never classify print,
student writing, or teacher ink.

## Document ordering and segmentation

Candidates are sorted by page, vertical coordinate, horizontal coordinate, and
number. A deterministic weighted increasing-subsequence selection favors
strong candidates and adjacent Test-number progression. Duplicates and rejected
candidates remain observable. Sequence support can prefer `7 -> 8 -> 9` over a
random `88`, but it never invents a missing Test.

Each accepted marker starts a full-width page-local region. The region ends at
the next accepted marker on that page or continues through intervening pages.
This supports multiple Tests on one page and Tests spanning pages. It is a
coarse structural envelope: headers, footers, questions, answers, and teacher
marks inside it are not yet separated.

## Private structure benchmark

The structure benchmark is separate from the frozen OCR transcription
benchmark. Its ignored manifest lives under `data/evaluation/structure/` and
contains safe aliases, expected page counts, Test numbers, pages, and manually
verified approximate marker boxes. It contains no student transcription. The
OCR ground-truth fingerprint remains
`33a5dc8e46a1cf0631d46da41a8490c4ec10a18194591144425422c61ff73f9a`.

Validation and local execution are:

```bash
uv run python -m scripts.evaluate_structure validate
uv run python -m scripts.evaluate_structure run
```

Normal terminal output exposes only safe aliases and aggregate counts. Complete
structure evidence, source references, and overlay images stay ignored beneath
`data/evaluation/structure/`.

### Results

| Metric | Result |
| --- | ---: |
| Papers | 3 |
| Pages | 29 |
| Expected markers | 48 |
| Detected markers | 44 |
| True positives | 44 |
| False positives | 0 |
| False negatives | 4 |
| Duplicate candidates | 0 |
| Precision | 1.0000 |
| Recall | 0.9167 |
| F1 | 0.9565 |
| Test-number accuracy | 1.0000 |
| Mean ordering accuracy | 0.9167 |

Ordering accuracy is the longest common subsequence divided by the longer
expected/detected sequence. It is below 1.0 when a marker is missing even if all
detected markers remain correctly ordered. The three safe paper aliases missed
1, 2, and 1 headings respectively. No false positive was accepted.

## Visual debugging and privacy

The overlay utility reads a canonical image, draws safe normalized marker
labels and region outlines on a copy, and verifies the source SHA-256 before
and after writing. It refuses source/output path equality. Private overlays for
all 29 evaluated pages were inspected: accepted boxes align with visible Test
headings, coarse regions preserve dimensions and page boundaries, and the four
known misses remain visible for human review. Visual QA: **PASS**.

Raw PDFs, canonical renders, private manifests, predictions, and overlays are
ignored. Documentation and tests contain no private page content or student
identity. Unit tests use synthetic OCR words and generated images only.

## Limitations and next work

- Recall is not yet complete; poor or fragmented Tesseract hierarchy can hide
  an otherwise visible heading.
- Sequence selection expects Tests 1 through 16 for this paper format.
- Regions are coarse full-width envelopes and include any page furniture or
  annotations between markers.
- Confidence is heuristic and uncalibrated.
- Approximate marker boxes are evaluated primarily by page and vertical
  proximity, not intersection-over-union.
- This small private benchmark does not establish population-level accuracy.

Future answer-area extraction may refine regions and handle missing-marker
review. Authorship separation remains a distinct later experiment; geometry,
color, and OCR confidence are not authorship labels.

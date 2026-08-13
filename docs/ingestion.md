# PDF ingestion

## Status and responsibility

Phase 2 implements the boundary from an untrusted PDF path to validated paper
metadata and stable PNG page references:

```text
PDF -> validation -> PaperDocument -> page rendering -> PaperDocument + PaperPage[]
```

The ingestion boundary does not perform OCR, text extraction, question
segmentation, image cleanup, grading, or persistence.

## Validation

`PDFValidator` requires all of the following before a document can be loaded:

- a `.pdf` extension;
- an existing, non-empty regular file;
- a file size no greater than `MAX_PDF_SIZE_MB`;
- parser-confirmed readable PDF content, not merely a trusted extension;
- no password requirement;
- at least one page and no more than `MAX_PDF_PAGES` pages.

Current local defaults are 50 MiB and 100 pages. Failures use safe application
exceptions without including a potentially identifying original filename in
the message.

## Domain models

`PaperDocument` is an immutable Pydantic value object containing:

- a generated UUID paper ID unrelated to the source filename;
- the resolved source reference and original filename;
- page count, byte size, and SHA-256 digest;
- an empty or complete ordered tuple of rendered pages.

`PaperPage` contains the internal paper ID, one-based page number, absolute PNG
path, and positive pixel dimensions. It deliberately has no OCR or grading
fields.

The original filename remains metadata for controlled internal use but never
determines the application ID or runtime output path. It may contain personal
information and must not be logged.

## Rendering behavior

`PDFRenderer` uses PyMuPDF to render each PDF page as an opaque PNG at the
configured `PDF_RENDER_DPI`, currently 150 DPI. This is a moderate baseline for
future handwriting OCR without choosing an extreme resolution. The Phase 0
corpus already contains full-page raster scans; rendering performs no
thresholding, deskewing, denoising, sharpening, or other OCR preprocessing.

Domain page numbers start at 1. PyMuPDF's zero-based iteration is adapted to
deterministic filenames:

```text
data/runtime/<paper-uuid-hex>/pages/page_0001.png
data/runtime/<paper-uuid-hex>/pages/page_0002.png
...
```

Rendering occurs in a temporary directory beneath the configured runtime root.
The completed directory is moved into place only after every page renders and
reopens with valid dimensions. Existing final output is not silently
overwritten. The loader's SHA-256 is checked before rendering to detect source
changes between stages.

## Security assumptions

- PDFs are parsed as untrusted data; embedded content and links are not
  executed or followed.
- Parser validation confirms content instead of trusting the extension.
- Generated UUIDs and fixed page filenames prevent raw filenames and traversal
  components from influencing output paths.
- A resolved-path containment check protects the generated paper directory.
- Runtime output is ignored by Git.
- Raw PDFs remain immutable and ignored under `data/raw/`.

PyMuPDF and MuPDF remain part of the trusted local parsing boundary. Dependency
updates should be reviewed because malformed document parsing is
security-sensitive.

## Configuration

| Variable | Default | Constraint |
| --- | ---: | --- |
| `MAX_PDF_SIZE_MB` | 50 | 1-1024 MiB |
| `MAX_PDF_PAGES` | 100 | 1-1000 pages |
| `PDF_RENDER_DPI` | 150 | 72-600 DPI |
| `RUNTIME_DATA_DIR` | `data/runtime` | local generated output root |

## Current limitations

- Rendering all pages is synchronous.
- Rendered PNGs can require substantially more disk space than source PDFs.
- Password-protected PDFs are rejected rather than decrypted.
- The source file must remain unchanged and available between loading and
  rendering.
- No malware sandbox is provided around MuPDF; deployment hardening belongs to
  a later security phase.
- OCR quality cannot be inferred from successful rendering. OCR is introduced
  in later phases and will consume `PaperPage.image_path`.

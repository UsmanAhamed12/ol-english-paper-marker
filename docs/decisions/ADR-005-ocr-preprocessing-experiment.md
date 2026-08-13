# ADR-005: Evaluate conservative OCR preprocessing

## Status

Accepted; no preprocessing variant adopted

## Context

Plain Tesseract is fast and preserves layout evidence, but its student-only
CER/WER is poor. Conservative preprocessing might improve recognition, but it
can also erase faint handwriting or amplify scan artifacts.

## Decision

Evaluate a small, fixed set of geometry-preserving OpenCV variants against the
frozen benchmark. Canonical images remain immutable, all outputs are derived,
and benchmark evidence determines adoption. Grayscale, mild Gaussian denoise,
Otsu thresholding, and their fixed combinations were measured. All regressed
on aggregate CER/WER, so the adopted variant is `none`.

## Alternatives considered

- Adopt preprocessing based on visual appearance without measurement.
- Tune operations separately per sample.
- Add resizing, cropping, deskew, adaptive thresholding, or authorship logic in
  the same experiment.
- Keep OpenCV out of the project and leave the research question untested.

## Consequences

OpenCV headless and NumPy are added, and typed derived-image infrastructure is
available for reproducible experiments. Canonical geometry and source hashes
remain protected. The experiment adds local computation and private artifacts,
and confirms that these simple variants do not justify replacing the baseline.
Thresholding can strengthen edge/noise artifacts and remove potentially useful
color evidence. More preprocessing should not be adopted without a new frozen,
controlled experiment.

# ADR-004: Evaluate a classical OCR baseline

## Status

Accepted

## Context

Qwen3-VL 4B was very slow on the local M1 and achieved CER/WER of 1.0 on the
frozen student-text benchmark. A fast layout-preserving baseline was needed
before increasing vision-model complexity.

## Decision

Evaluate plain local Tesseract through the existing `OCRProvider` and
`OCRService` boundaries. Preserve typed words, bounding boxes, hierarchy, and
confidence, and benchmark canonical PyMuPDF images without preprocessing.

## Alternatives considered

- Continue tuning or downloading VLMs before a classical baseline.
- Copy the experimental OpenCV/Tesseract notebook into production.
- Flatten OCR immediately and discard spatial evidence.

## Consequences

Positive consequences are local/free execution, deterministic behavior, low
latency, word boxes, confidence, and useful printed-text/layout evidence.

Negative consequences are weak handwriting transcription, confidence that
cannot identify authorship, unresolved teacher-mark contamination, and likely
need for separately evaluated preprocessing. Typed evidence also expands
private result storage.

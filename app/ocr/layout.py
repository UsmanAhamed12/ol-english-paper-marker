"""Deterministic reconstruction of readable text from spatial OCR words."""

from __future__ import annotations

from collections import defaultdict

from app.ocr.models import OCRWord


def reconstruct_layout_text(words: tuple[OCRWord, ...]) -> str:
    """Preserve Tesseract hierarchy as lines and paragraph-separated blocks."""

    if not words:
        return ""

    grouped: dict[tuple[int, int, int], list[OCRWord]] = defaultdict(list)
    for word in words:
        grouped[_line_key(word)].append(word)

    paragraphs: list[list[str]] = []
    previous_paragraph: tuple[int, int] | None = None
    for key in sorted(grouped):
        paragraph = key[:2]
        if paragraph != previous_paragraph:
            paragraphs.append([])
            previous_paragraph = paragraph
        line_words = sorted(
            grouped[key],
            key=lambda word: (
                word.word_number if word.word_number is not None else 2**31,
                word.bbox.x,
                word.bbox.y,
                word.text,
            ),
        )
        paragraphs[-1].append(" ".join(word.text for word in line_words))

    return "\n\n".join("\n".join(lines) for lines in paragraphs)


def _line_key(word: OCRWord) -> tuple[int, int, int]:
    """Prefer hierarchy, falling back deterministically to pixel position."""

    return (
        word.block_number if word.block_number is not None else word.bbox.y + 1,
        word.paragraph_number if word.paragraph_number is not None else 1,
        word.line_number if word.line_number is not None else word.bbox.y + 1,
    )

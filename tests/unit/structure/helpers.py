"""Synthetic OCR evidence builders for structure tests."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

from app.domain.models.paper import PaperPage
from app.ocr.models import (
    BoundingBox,
    OCRPageResult,
    OCRStructuredEvidence,
    OCRWord,
)


def page(
    tmp_path: Path, page_number: int = 1, paper_id: UUID | None = None
) -> PaperPage:
    image_path = (tmp_path / f"page_{page_number:04d}.png").resolve()
    image_path.write_bytes(b"synthetic page")
    return PaperPage(
        paper_id=paper_id or uuid4(),
        page_number=page_number,
        image_path=image_path,
        width=1000,
        height=1400,
    )


def word(
    text: str,
    *,
    x: int,
    y: int,
    line: int,
    word_number: int,
    confidence: float = 0.9,
    width: int = 80,
) -> OCRWord:
    return OCRWord(
        text=text,
        confidence=confidence,
        bbox=BoundingBox(x=x, y=y, width=width, height=40),
        block_number=1,
        paragraph_number=1,
        line_number=line,
        word_number=word_number,
    )


def result(page_value: PaperPage, words: tuple[OCRWord, ...]) -> OCRPageResult:
    return OCRPageResult(
        paper_id=page_value.paper_id,
        page_number=page_value.page_number,
        source_image_path=page_value.image_path,
        raw_text="synthetic",
        normalized_text="synthetic",
        provider="synthetic",
        model_version="v1",
        processing_duration_ms=1,
        evidence=OCRStructuredEvidence(
            words=words,
            layout_text="synthetic" if words else "",
        ),
    )

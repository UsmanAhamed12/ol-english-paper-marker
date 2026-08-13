"""Tests for deterministic layout reconstruction."""

from app.ocr.layout import reconstruct_layout_text
from app.ocr.models import BoundingBox, OCRWord


def _word(
    text: str,
    *,
    block: int,
    paragraph: int,
    line: int,
    word: int,
    x: int,
) -> OCRWord:
    return OCRWord(
        text=text,
        bbox=BoundingBox(x=x, y=line * 10, width=8, height=5),
        block_number=block,
        paragraph_number=paragraph,
        line_number=line,
        word_number=word,
    )


def test_layout_reconstruction_preserves_lines_paragraphs_and_word_order() -> None:
    words = (
        _word("line", block=1, paragraph=1, line=2, word=2, x=20),
        _word("First", block=1, paragraph=1, line=1, word=1, x=1),
        _word("Second", block=1, paragraph=1, line=2, word=1, x=1),
        _word("paragraph", block=1, paragraph=2, line=1, word=1, x=1),
    )

    assert reconstruct_layout_text(words) == ("First\nSecond line\n\nparagraph")


def test_layout_reconstruction_handles_no_words() -> None:
    assert reconstruct_layout_text(()) == ""

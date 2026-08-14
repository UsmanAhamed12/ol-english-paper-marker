"""Synthetic tests for conservative multi-signal Test-marker detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ocr.models import BoundingBox
from app.structure import models as structure_models
from app.structure.marker_detector import detect_marker_candidates, select_markers
from app.structure.models import MarkerDetectionStrategy
from tests.unit.structure.helpers import page, result, word


@pytest.mark.parametrize(
    ("tokens", "number", "strategy"),
    [
        (("Test", "01"), 1, MarkerDetectionStrategy.EXACT_TOKENS),
        (("TEST", "02"), 2, MarkerDetectionStrategy.EXACT_TOKENS),
        (("Test03",), 3, MarkerDetectionStrategy.COMPACT_TOKEN),
        (("Test", "O4"), 4, MarkerDetectionStrategy.OCR_CONFUSION),
        (("Test", "Ol"), 1, MarkerDetectionStrategy.OCR_CONFUSION),
        (("lest", "05"), 5, MarkerDetectionStrategy.OCR_CONFUSION),
        (("TesT", "06"), 6, MarkerDetectionStrategy.EXACT_TOKENS),
    ],
)
def test_marker_variants_preserve_geometry_and_strategy(
    tmp_path: Path,
    tokens: tuple[str, ...],
    number: int,
    strategy: MarkerDetectionStrategy,
) -> None:
    paper_page = page(tmp_path)
    words = tuple(
        word(token, x=100 + index * 100, y=100, line=1, word_number=index + 1)
        for index, token in enumerate(tokens)
    )

    candidates = detect_marker_candidates(paper_page, result(paper_page, words))

    assert len(candidates) == 1
    assert candidates[0].test_number == number
    assert candidates[0].strategy is strategy
    assert candidates[0].bbox.x == 100
    assert candidates[0].source_word_indices == tuple(range(len(tokens)))


def test_nearby_punctuation_noise_is_skipped_but_retained_in_source_indexes(
    tmp_path: Path,
) -> None:
    paper_page = page(tmp_path)
    words = (
        word("§", x=20, y=100, line=1, word_number=1, width=20),
        word("Test", x=100, y=100, line=1, word_number=2),
        word("07", x=200, y=100, line=1, word_number=3),
    )

    candidate = detect_marker_candidates(paper_page, result(paper_page, words))[0]

    assert candidate.test_number == 7
    assert candidate.source_word_indices == (1, 2)


def test_isolated_dash_between_keyword_and_number_is_allowed(tmp_path: Path) -> None:
    paper_page = page(tmp_path)
    words = (
        word("Test", x=100, y=100, line=1, word_number=1),
        word("-", x=180, y=100, line=1, word_number=2),
        word("04", x=205, y=100, line=1, word_number=3),
    )

    candidates = detect_marker_candidates(paper_page, result(paper_page, words))

    assert len(candidates) == 1
    assert candidates[0].test_number == 4
    assert candidates[0].source_word_indices == (0, 2)


def test_normal_sentence_and_footer_noise_are_not_accepted(tmp_path: Path) -> None:
    paper_page = page(tmp_path)
    sentence = (
        word("This", x=50, y=300, line=1, word_number=1),
        word("test", x=150, y=300, line=1, word_number=2),
        word("01", x=250, y=300, line=1, word_number=3),
    )
    long_heading_like_sentence = tuple(
        word(token, x=50 + index * 100, y=500, line=2, word_number=index + 1)
        for index, token in enumerate(("Test", "02", "is", "in", "this", "sentence"))
    )
    footer = (
        word("Test", x=100, y=1320, line=3, word_number=1),
        word("03", x=200, y=1320, line=3, word_number=2),
    )

    candidates = detect_marker_candidates(
        paper_page,
        result(paper_page, sentence + long_heading_like_sentence + footer),
    )
    markers, rejected, _ = select_markers(
        candidates,
        expected_test_numbers=tuple(range(1, 17)),
    )

    assert markers == ()
    assert len(rejected) == 2


def _candidate(
    number: int, page_number: int, y: int, confidence: float
) -> structure_models.TestMarkerCandidate:
    return structure_models.TestMarkerCandidate(
        test_number=number,
        raw_text=f"Test {number:02d}",
        page_number=page_number,
        bbox=BoundingBox(x=10, y=y, width=150, height=40),
        confidence=confidence,
        text_similarity=1,
        numeric_confidence=1,
        ocr_confidence=confidence,
        geometry_confidence=1,
        strategy=MarkerDetectionStrategy.EXACT_TOKENS,
        source_word_indices=(number,),
    )


def test_sequence_selection_rejects_duplicate_and_out_of_order_candidates() -> None:
    candidates = (
        _candidate(1, 1, 100, 0.9),
        _candidate(3, 1, 300, 0.8),
        _candidate(2, 1, 400, 0.95),
        _candidate(3, 1, 500, 0.95),
        _candidate(4, 2, 100, 0.9),
    )

    markers, rejected, duplicates = select_markers(
        candidates,
        expected_test_numbers=(1, 2, 3, 4),
    )

    assert [marker.test_number for marker in markers] == [1, 2, 3, 4]
    assert len(rejected) == 1
    assert duplicates == (3,)
    assert all(0 <= marker.sequence_confidence <= 1 for marker in markers)


def test_unexpected_number_is_rejected_without_invention() -> None:
    markers, rejected, _ = select_markers(
        (_candidate(1, 1, 10, 0.9), _candidate(88, 1, 100, 0.99)),
        expected_test_numbers=(1, 2, 3),
    )

    assert [marker.test_number for marker in markers] == [1]
    assert [candidate.test_number for candidate in rejected] == [88]

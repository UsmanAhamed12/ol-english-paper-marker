"""Tests for page and cross-page Test segmentation."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.ocr.models import BoundingBox
from app.structure import models as structure_models
from app.structure.models import (
    ExamPageStructure,
    ExamStructure,
    MarkerDetectionStrategy,
)
from app.structure.service import ExamStructureDetector
from tests.unit.structure.helpers import page, result, word


def test_multiple_tests_on_one_page_produce_distinct_regions(tmp_path: Path) -> None:
    paper_page = page(tmp_path)
    words = (
        word("Test", x=100, y=100, line=1, word_number=1),
        word("01", x=200, y=100, line=1, word_number=2),
        word("Test", x=100, y=700, line=2, word_number=1),
        word("02", x=200, y=700, line=2, word_number=2),
    )

    structure = ExamStructureDetector(expected_test_numbers=(1, 2)).detect(
        (paper_page,),
        (result(paper_page, words),),
    )

    assert [test.test_number for test in structure.tests] == [1, 2]
    assert structure.tests[0].page_regions[0].bbox.y == 100
    assert structure.tests[0].page_regions[0].bbox.height == 600
    assert structure.tests[1].page_regions[0].bbox.height == 700
    assert len(structure.pages[0].markers) == 2


def test_test_region_spans_page_boundary_and_missing_is_explicit(
    tmp_path: Path,
) -> None:
    paper_id = uuid4()
    first = page(tmp_path, 1, paper_id)
    second = page(tmp_path, 2, paper_id)
    first_words = (
        word("Test", x=100, y=1000, line=1, word_number=1),
        word("01", x=200, y=1000, line=1, word_number=2),
    )
    second_words = (
        word("Test", x=100, y=500, line=1, word_number=1),
        word("03", x=200, y=500, line=1, word_number=2),
    )

    structure = ExamStructureDetector(expected_test_numbers=(1, 2, 3)).detect(
        (first, second),
        (result(first, first_words), result(second, second_words)),
    )

    assert structure.missing_test_numbers == (2,)
    assert structure.tests[0].start_page == 1
    assert structure.tests[0].end_page == 2
    assert [span.page_number for span in structure.tests[0].page_regions] == [1, 2]
    assert structure.tests[0].page_regions[1].bbox.height == 500


def test_detection_is_deterministic_and_models_are_immutable(tmp_path: Path) -> None:
    paper_page = page(tmp_path)
    words = (
        word("Test", x=100, y=100, line=1, word_number=1),
        word("01", x=200, y=100, line=1, word_number=2),
    )
    detector = ExamStructureDetector(expected_test_numbers=(1, 2))

    first = detector.detect((paper_page,), (result(paper_page, words),))
    second = detector.detect((paper_page,), (result(paper_page, words),))

    assert first == second
    assert first.missing_test_numbers == (2,)
    with pytest.raises(ValidationError):
        first.page_count = 2


def test_invalid_page_order_is_rejected(tmp_path: Path) -> None:
    paper_id = uuid4()
    second = page(tmp_path, 2, paper_id)

    with pytest.raises(ValueError, match="ordered"):
        ExamStructureDetector().detect((second,), (result(second, ()),))


def test_exam_structure_rejects_invalid_geometry_sequence(tmp_path: Path) -> None:
    paper_page = page(tmp_path)
    structure = ExamStructureDetector(expected_test_numbers=(1,)).detect(
        (paper_page,),
        (result(paper_page, ()),),
    )
    payload = structure.model_dump()
    payload["page_count"] = 2

    with pytest.raises(ValidationError, match="page count"):
        ExamStructure.model_validate(payload)


def test_page_structure_rejects_evidence_outside_page(tmp_path: Path) -> None:
    paper_page = page(tmp_path)
    candidate = structure_models.TestMarkerCandidate(
        test_number=1,
        raw_text="Test 01",
        page_number=1,
        bbox=BoundingBox(x=paper_page.width - 5, y=10, width=20, height=20),
        confidence=0.9,
        text_similarity=1.0,
        numeric_confidence=1.0,
        ocr_confidence=0.9,
        geometry_confidence=1.0,
        strategy=MarkerDetectionStrategy.EXACT_TOKENS,
        source_word_indices=(0, 1),
    )

    with pytest.raises(ValidationError, match="fit within page geometry"):
        ExamPageStructure(
            page_number=1,
            width=paper_page.width,
            height=paper_page.height,
            candidates=(candidate,),
        )

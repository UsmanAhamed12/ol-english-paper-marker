"""Synthetic metric tests for the private structure benchmark."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from app.evaluation.structure_benchmark.models import (
    StructureBenchmarkPaper,
    StructureGroundTruthMarker,
)
from app.evaluation.structure_benchmark.runner import (
    evaluate_structure,
    summarize_structure_results,
)
from app.ocr.models import BoundingBox
from app.structure.service import ExamStructureDetector
from tests.unit.structure.helpers import page, result, word


def test_structure_metrics_count_false_positive_missing_duplicate_and_order(
    tmp_path: Path,
) -> None:
    paper_id = uuid4()
    paper_page = page(tmp_path, paper_id=paper_id)
    words = (
        word("Test", x=100, y=100, line=1, word_number=1),
        word("01", x=200, y=100, line=1, word_number=2),
        word("Test", x=100, y=300, line=2, word_number=1),
        word("02", x=200, y=300, line=2, word_number=2),
        word("Test", x=100, y=500, line=3, word_number=1),
        word("02", x=200, y=500, line=3, word_number=2),
        word("Test", x=100, y=700, line=4, word_number=1),
        word("04", x=200, y=700, line=4, word_number=2),
    )
    structure = ExamStructureDetector(expected_test_numbers=(1, 2, 3, 4)).detect(
        (paper_page,),
        (result(paper_page, words),),
    )
    ground_truth = StructureBenchmarkPaper(
        paper_alias="synthetic-paper",
        source_path=Path("private.pdf"),
        expected_page_count=1,
        expected_markers=tuple(
            StructureGroundTruthMarker(
                test_number=number,
                page_number=1,
                bbox=BoundingBox(x=100, y=y, width=180, height=40),
            )
            for number, y in ((1, 100), (2, 300), (3, 600))
        ),
    )

    measured = evaluate_structure(ground_truth, structure)
    summary = summarize_structure_results((measured,))

    assert measured.true_positives == 2
    assert measured.false_positives == 1
    assert measured.false_negatives == 1
    assert measured.duplicate_markers == 1
    assert measured.missing_test_numbers == (3,)
    assert measured.precision == pytest.approx(2 / 3)
    assert measured.recall == pytest.approx(2 / 3)
    assert measured.test_number_accuracy == pytest.approx(2 / 3)
    assert measured.ordering_accuracy == pytest.approx(2 / 3)
    assert summary.f1 == pytest.approx(2 / 3)

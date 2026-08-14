"""Synthetic immutability tests for private structure overlays."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.exceptions import StructureDetectionError
from app.structure.overlay import render_structure_overlay
from app.structure.service import ExamStructureDetector
from tests.unit.structure.helpers import page, result, word


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_overlay_is_separate_and_canonical_image_is_unchanged(tmp_path: Path) -> None:
    paper_page = page(tmp_path)
    image = np.full((1400, 1000, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(paper_page.image_path), image)
    before = _hash(paper_page.image_path)
    words = (
        word("Test", x=100, y=100, line=1, word_number=1),
        word("01", x=200, y=100, line=1, word_number=2),
    )
    structure = ExamStructureDetector(expected_test_numbers=(1,)).detect(
        (paper_page,),
        (result(paper_page, words),),
    )
    output = (tmp_path / "private" / "overlay.png").resolve()

    rendered = render_structure_overlay(paper_page, structure.pages[0], output)

    assert rendered == output
    assert output.is_file()
    assert output != paper_page.image_path
    assert _hash(paper_page.image_path) == before


def test_overlay_rejects_source_overwrite(tmp_path: Path) -> None:
    paper_page = page(tmp_path)
    image = np.full((1400, 1000, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(paper_page.image_path), image)
    structure = ExamStructureDetector(expected_test_numbers=(1,)).detect(
        (paper_page,),
        (result(paper_page, ()),),
    )

    with pytest.raises(StructureDetectionError, match="overwrite"):
        render_structure_overlay(
            paper_page,
            structure.pages[0],
            paper_page.image_path,
        )

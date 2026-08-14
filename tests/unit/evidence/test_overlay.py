"""Synthetic private evidence-overlay safety tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.core.exceptions import EvidenceSeparationError
from app.evidence.answer_regions import AnswerRegionDetector
from app.evidence.overlay import render_evidence_overlay
from app.evidence.separator import EvidenceSeparator
from app.evidence.service import EvidenceSeparationService
from app.ocr.models import BoundingBox, OCRPageResult, OCRStructuredEvidence
from tests.unit.evidence.helpers import page


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_overlay_is_derived_and_canonical_hash_is_unchanged(tmp_path: Path) -> None:
    paper_page = page(tmp_path)
    result = OCRPageResult(
        paper_id=paper_page.paper_id,
        page_number=1,
        source_image_path=paper_page.image_path,
        raw_text="",
        normalized_text="",
        confidence=None,
        provider="synthetic",
        processing_duration_ms=1,
        evidence=OCRStructuredEvidence(),
    )
    test_evidence = EvidenceSeparationService(
        EvidenceSeparator(), AnswerRegionDetector()
    ).analyze_region(
        paper_page,
        result,
        test_number=8,
        region_bbox=BoundingBox(x=100, y=100, width=500, height=400),
    )
    before = _hash(paper_page.image_path)
    output = (tmp_path / "private" / "overlay.png").resolve()

    rendered = render_evidence_overlay(
        paper_page,
        test_evidence,
        output,
        crop_bbox=BoundingBox(x=100, y=100, width=500, height=400),
    )

    assert rendered == output
    assert output.is_file()
    assert output != paper_page.image_path
    assert _hash(paper_page.image_path) == before


def test_overlay_rejects_source_overwrite(tmp_path: Path) -> None:
    paper_page = page(tmp_path)
    result = OCRPageResult(
        paper_id=paper_page.paper_id,
        page_number=1,
        source_image_path=paper_page.image_path,
        raw_text="",
        normalized_text="",
        confidence=None,
        provider="synthetic",
        processing_duration_ms=1,
        evidence=OCRStructuredEvidence(),
    )
    evidence = EvidenceSeparationService(
        EvidenceSeparator(), AnswerRegionDetector()
    ).analyze_region(
        paper_page,
        result,
        test_number=1,
        region_bbox=BoundingBox(x=100, y=100, width=400, height=300),
    )

    with pytest.raises(EvidenceSeparationError, match="overwrite"):
        render_evidence_overlay(paper_page, evidence, paper_page.image_path)

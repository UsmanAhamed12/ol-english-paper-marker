"""Synthetic evidence service, answer-region, and immutability tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import pytest
from pydantic import ValidationError

from app.evidence import models as evidence_models
from app.evidence.answer_regions import AnswerRegionDetector
from app.evidence.models import (
    AnswerRegionSignal,
    EvidenceType,
)
from app.evidence.separator import EvidenceSeparator
from app.evidence.service import EvidenceSeparationService
from app.ocr.models import (
    BoundingBox,
    OCRPageResult,
    OCRStructuredEvidence,
    OCRWord,
)
from tests.unit.evidence.helpers import page


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result(page_image: object, words: tuple[OCRWord, ...]) -> OCRPageResult:
    page_value = page_image
    raw = " ".join(word.text for word in words)
    return OCRPageResult(
        paper_id=page_value.paper_id,  # type: ignore[attr-defined]
        page_number=page_value.page_number,  # type: ignore[attr-defined]
        source_image_path=page_value.image_path,  # type: ignore[attr-defined]
        raw_text=raw,
        normalized_text=raw,
        confidence=0.9,
        provider="synthetic",
        model_version="1",
        processing_duration_ms=1,
        evidence=OCRStructuredEvidence(words=words, layout_text=raw),
    )


def test_blank_writing_guides_create_answer_candidate(tmp_path: Path) -> None:
    paper_page = page(tmp_path)
    image = cv2.imread(str(paper_page.image_path), cv2.IMREAD_COLOR)
    assert image is not None
    for y in (300, 390, 480, 570):
        cv2.line(image, (150, y), (850, y), (80, 80, 80), 2)
    assert cv2.imwrite(str(paper_page.image_path), image)
    region = BoundingBox(x=100, y=200, width=800, height=450)

    detected = AnswerRegionDetector().detect(
        paper_page,
        test_number=6,
        region_bbox=region,
        evidence_regions=(),
    )

    assert detected
    assert AnswerRegionSignal.WRITING_GUIDES in detected[0].signals
    assert AnswerRegionSignal.BLANK_WRITING_SPACE in detected[0].signals
    assert detected[0].bbox.width <= region.width


def test_service_preserves_source_and_classifies_printed_word(tmp_path: Path) -> None:
    paper_page = page(tmp_path)
    image = cv2.imread(str(paper_page.image_path), cv2.IMREAD_COLOR)
    assert image is not None
    cv2.putText(
        image,
        "PRINT",
        (100, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    assert cv2.imwrite(str(paper_page.image_path), image)
    before = _hash(paper_page.image_path)
    word = OCRWord(
        text="PRINT",
        confidence=0.95,
        bbox=BoundingBox(x=95, y=120, width=150, height=50),
        block_number=1,
        paragraph_number=1,
        line_number=1,
        word_number=1,
    )

    analyzed = EvidenceSeparationService(
        EvidenceSeparator(), AnswerRegionDetector()
    ).analyze_region(
        paper_page,
        _result(paper_page, (word,)),
        test_number=1,
        region_bbox=BoundingBox(x=50, y=80, width=400, height=200),
    )

    assert analyzed.evidence_regions
    assert analyzed.evidence_regions[0].evidence_type is EvidenceType.PRINTED
    assert _hash(paper_page.image_path) == before


def test_visual_red_cross_is_preserved_as_teacher_candidate(tmp_path: Path) -> None:
    paper_page = page(tmp_path)
    image = cv2.imread(str(paper_page.image_path), cv2.IMREAD_COLOR)
    assert image is not None
    cv2.line(image, (500, 300), (560, 360), (0, 0, 200), 7)
    cv2.line(image, (560, 300), (500, 360), (0, 0, 200), 7)
    assert cv2.imwrite(str(paper_page.image_path), image)

    evidence = EvidenceSeparator().separate(
        paper_page,
        _result(paper_page, ()),
        test_number=4,
        region_bbox=BoundingBox(x=400, y=200, width=300, height=300),
    )

    assert any(
        item.evidence_type is EvidenceType.TEACHER_CANDIDATE for item in evidence
    )


def test_models_are_immutable_and_reject_child_outside_region(tmp_path: Path) -> None:
    paper_page = page(tmp_path)
    evidence = EvidenceSeparationService(
        EvidenceSeparator(), AnswerRegionDetector()
    ).analyze_region(
        paper_page,
        _result(paper_page, ()),
        test_number=1,
        region_bbox=BoundingBox(x=100, y=100, width=300, height=300),
    )

    with pytest.raises(ValidationError):
        evidence.test_number = 2

    payload = evidence.model_dump()
    payload["answer_regions"] = [
        {
            "paper_id": str(paper_page.paper_id),
            "page_number": 1,
            "test_number": 1,
            "bbox": {"x": 0, "y": 0, "width": 50, "height": 50},
            "confidence": 0.5,
            "signals": ["blank_writing_space"],
            "source_evidence_indices": [],
            "source_image_path": str(paper_page.image_path),
            "detection_strategy": "synthetic",
        }
    ]
    with pytest.raises(ValidationError, match="fit within"):
        evidence_models.TestEvidence.model_validate(payload)

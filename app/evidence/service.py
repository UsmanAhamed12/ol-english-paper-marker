"""Application service composing evidence and answer-region detection."""

from __future__ import annotations

from app.domain.models.paper import PaperPage
from app.evidence.answer_regions import AnswerRegionDetector
from app.evidence.models import TestEvidence
from app.evidence.separator import EvidenceSeparator
from app.ocr.models import BoundingBox, OCRPageResult


class EvidenceSeparationService:
    """Analyze one detected Test page region without changing OCR evidence."""

    def __init__(
        self,
        separator: EvidenceSeparator,
        answer_region_detector: AnswerRegionDetector,
    ) -> None:
        self._separator = separator
        self._answer_region_detector = answer_region_detector

    def analyze_region(
        self,
        page: PaperPage,
        ocr_result: OCRPageResult,
        *,
        test_number: int,
        region_bbox: BoundingBox,
    ) -> TestEvidence:
        """Return classified evidence and answer-space candidates."""

        evidence = self._separator.separate(
            page,
            ocr_result,
            test_number=test_number,
            region_bbox=region_bbox,
        )
        answers = self._answer_region_detector.detect(
            page,
            test_number=test_number,
            region_bbox=region_bbox,
            evidence_regions=evidence,
        )
        return TestEvidence(
            paper_id=page.paper_id,
            page_number=page.page_number,
            test_number=test_number,
            region_bbox=region_bbox,
            evidence_regions=evidence,
            answer_regions=answers,
        )

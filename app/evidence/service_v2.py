"""Application boundary for explicit evidence separation V2."""

from __future__ import annotations

from app.domain.models.paper import PaperPage
from app.evidence.separator_v2 import EvidenceSeparatorV2
from app.evidence.v2_models import TestEvidenceV2
from app.ocr.models import BoundingBox, OCRPageResult


class EvidenceSeparationServiceV2:
    """Compose V2 without changing the existing production V1 service."""

    def __init__(self, separator: EvidenceSeparatorV2) -> None:
        self._separator = separator

    def analyze_region(
        self,
        page: PaperPage,
        ocr_result: OCRPageResult,
        *,
        test_number: int,
        region_bbox: BoundingBox,
    ) -> TestEvidenceV2:
        return self._separator.separate(
            page,
            ocr_result,
            test_number=test_number,
            region_bbox=region_bbox,
        )

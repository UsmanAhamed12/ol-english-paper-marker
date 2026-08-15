"""Run the unchanged evidence separator on evidence-v2 samples."""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import cv2

from app.core.config import Settings
from app.core.exceptions import EvidenceSeparationError
from app.domain.models.paper import PaperPage
from app.evaluation.evidence_expansion.models import (
    EvidenceExpansionManifest,
    EvidenceExpansionSample,
)
from app.evidence.answer_regions import AnswerRegionDetector
from app.evidence.models import TestEvidence
from app.evidence.separator import EvidenceSeparator
from app.evidence.service import EvidenceSeparationService
from app.ocr.models import OCRPageResult
from app.ocr.normalizer import OCRNormalizer
from app.ocr.providers.tesseract import TesseractOCRProvider
from app.ocr.service import OCRService


def run_current_evidence_baseline(
    manifest: EvidenceExpansionManifest,
    *,
    settings: Settings | None = None,
) -> dict[str, TestEvidence]:
    """Measure current production rules without changing settings or thresholds."""

    settings = settings or Settings()
    ocr = OCRService(
        TesseractOCRProvider.from_system(
            language=settings.tesseract_language,
            psm=settings.tesseract_psm,
            timeout_seconds=settings.tesseract_timeout_seconds,
        ),
        OCRNormalizer(),
    )
    service = EvidenceSeparationService(EvidenceSeparator(), AnswerRegionDetector())
    page_cache: dict[tuple[str, int], tuple[PaperPage, OCRPageResult]] = {}
    predictions: dict[str, TestEvidence] = {}
    for sample in manifest.samples:
        key = (sample.paper_alias, sample.page_number)
        cached = page_cache.get(key)
        if cached is None:
            page = _sample_page(sample)
            source_hash = _sha256(page.image_path)
            if source_hash != sample.source_image_sha256:
                raise EvidenceSeparationError(
                    "Evidence-v2 canonical source hash changed"
                )
            result = ocr.process_page(page)
            if _sha256(page.image_path) != source_hash:
                raise EvidenceSeparationError(
                    "Evidence-v2 canonical source changed during OCR"
                )
            page_cache[key] = (page, result)
        else:
            page, result = cached
        predictions[sample.sample_id] = service.analyze_region(
            page,
            result,
            test_number=sample.test_number or 1,
            region_bbox=sample.region,
        )
    return predictions


def _sample_page(sample: EvidenceExpansionSample) -> PaperPage:
    source = sample.source_image_path.resolve(strict=True)
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim < 2:
        raise EvidenceSeparationError("Private evidence-v2 source page is invalid")
    height, width = image.shape[:2]
    if (width, height) != (
        sample.page_width,
        sample.page_height,
    ):
        raise EvidenceSeparationError("Private evidence-v2 source dimensions changed")
    return PaperPage(
        paper_id=uuid5(
            NAMESPACE_URL,
            f"ol-english-evidence-v2-evaluation/{sample.paper_alias}",
        ),
        page_number=sample.page_number,
        image_path=source,
        width=width,
        height=height,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

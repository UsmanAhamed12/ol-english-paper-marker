"""Separate typed evidence candidates inside one detected Test region."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2

from app.core.exceptions import EvidenceSeparationError
from app.domain.models.paper import PaperPage
from app.evidence.classifier import EvidenceClassifier
from app.evidence.features import (
    component_geometry_features,
    detect_chromatic_components,
    extract_geometry_features,
    extract_ink_features,
    intersection_ratio,
)
from app.evidence.models import (
    EvidenceClassification,
    EvidenceFeatures,
    EvidenceRegion,
)
from app.ocr.models import BoundingBox, OCRPageResult


class EvidenceSeparator:
    """Create conservative word and visual-component evidence candidates."""

    def __init__(self, classifier: EvidenceClassifier | None = None) -> None:
        self._classifier = classifier or EvidenceClassifier()

    def separate(
        self,
        page: PaperPage,
        ocr_result: OCRPageResult,
        *,
        test_number: int,
        region_bbox: BoundingBox,
    ) -> tuple[EvidenceRegion, ...]:
        """Classify OCR words and non-overlapping chromatic components."""

        _validate_inputs(page, ocr_result, region_bbox)
        source_hash = _sha256(page.image_path)
        image = cv2.imread(str(page.image_path), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3:
            raise EvidenceSeparationError("Evidence source image is invalid")
        words = ocr_result.evidence.words if ocr_result.evidence is not None else ()
        regions: list[EvidenceRegion] = []
        word_boxes: list[BoundingBox] = []
        for word_index, word in enumerate(words):
            clipped = _intersection_box(word.bbox, region_bbox)
            if clipped is None or intersection_ratio(word.bbox, region_bbox) < 0.60:
                continue
            ink = extract_ink_features(image, clipped)
            geometry = extract_geometry_features(
                words,
                word_index,
                page_width=page.width,
                component_count=ink.connected_component_count,
            )
            features = EvidenceFeatures(
                ink=ink,
                geometry=geometry,
                ocr_confidence=word.confidence,
            )
            classification = self._classifier.classify(features)
            regions.append(
                _region(
                    page,
                    test_number,
                    clipped,
                    features,
                    classification,
                    source_word_indices=(word_index,),
                )
            )
            word_boxes.append(clipped)

        for component in detect_chromatic_components(image, region_bbox):
            if any(
                intersection_ratio(component, word_box) >= 0.50
                for word_box in word_boxes
            ):
                continue
            ink = extract_ink_features(image, component)
            geometry = component_geometry_features(
                component,
                page_width=page.width,
                ink=ink,
            )
            features = EvidenceFeatures(ink=ink, geometry=geometry)
            classification = self._classifier.classify(features)
            regions.append(
                _region(
                    page,
                    test_number,
                    component,
                    features,
                    classification,
                    source_word_indices=(),
                )
            )
        if _sha256(page.image_path) != source_hash:
            raise EvidenceSeparationError(
                "Canonical page changed during evidence separation"
            )
        return tuple(
            sorted(
                regions,
                key=lambda region: (
                    region.bbox.y,
                    region.bbox.x,
                    region.evidence_type.value,
                    region.source_word_indices,
                ),
            )
        )


def _region(
    page: PaperPage,
    test_number: int,
    bbox: BoundingBox,
    features: EvidenceFeatures,
    classification: EvidenceClassification,
    *,
    source_word_indices: tuple[int, ...],
) -> EvidenceRegion:
    return EvidenceRegion(
        paper_id=page.paper_id,
        page_number=page.page_number,
        test_number=test_number,
        bbox=bbox,
        evidence_type=classification.evidence_type,
        confidence=classification.score,
        signals=classification.signals,
        features=features,
        source_word_indices=source_word_indices,
        source_image_path=page.image_path,
        classification_strategy=classification.strategy_version,
    )


def _validate_inputs(
    page: PaperPage,
    ocr_result: OCRPageResult,
    region_bbox: BoundingBox,
) -> None:
    if not page.image_path.is_file():
        raise EvidenceSeparationError("Evidence source image is unavailable")
    if (
        ocr_result.paper_id != page.paper_id
        or ocr_result.page_number != page.page_number
    ):
        raise EvidenceSeparationError("OCR evidence does not match its canonical page")
    if ocr_result.source_image_path.resolve() != page.image_path.resolve():
        raise EvidenceSeparationError(
            "OCR evidence source does not match canonical page"
        )
    page_box = BoundingBox(x=0, y=0, width=page.width, height=page.height)
    if _intersection_box(region_bbox, page_box) != region_bbox:
        raise EvidenceSeparationError("Test region exceeds canonical page geometry")


def _intersection_box(left: BoundingBox, right: BoundingBox) -> BoundingBox | None:
    x = max(left.x, right.x)
    y = max(left.y, right.y)
    right_edge = min(left.x + left.width, right.x + right.width)
    bottom = min(left.y + left.height, right.y + right.height)
    if right_edge <= x or bottom <= y:
        return None
    return BoundingBox(x=x, y=y, width=right_edge - x, height=bottom - y)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

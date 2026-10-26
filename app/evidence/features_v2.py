"""Deterministic aggregate feature extraction for evidence groups."""

from __future__ import annotations

from statistics import fmean, pstdev

import cv2
import numpy as np

from app.core.exceptions import EvidenceSeparationError
from app.evidence.features import extract_ink_features
from app.evidence.v2_models import EvidenceFeaturesV2, EvidenceGroup
from app.ocr.models import BoundingBox


class EvidenceFeatureExtractorV2:
    """Measure color, component, stroke, OCR, layout, and context features."""

    def extract(
        self,
        image: np.ndarray,
        group: EvidenceGroup,
        *,
        test_region: BoundingBox,
        answer_spaces: tuple[BoundingBox, ...] = (),
    ) -> EvidenceFeaturesV2:
        """Return typed aggregate evidence without making an authorship claim."""

        ink = extract_ink_features(image, group.bbox)
        base = tuple(item.base_region.features for item in group.components)
        confidences = tuple(
            item.ocr_confidence for item in base if item.ocr_confidence is not None
        )
        regularities = tuple(item.geometry.regularity for item in base)
        densities = tuple(item.geometry.line_density for item in base)
        centers = tuple(item.bbox.y + item.bbox.height / 2 for item in group.components)
        mean_height = max(1.0, fmean(item.bbox.height for item in group.components))
        alignment = _unit(
            1 - (pstdev(centers) / mean_height if len(centers) > 1 else 0)
        )
        component_area = sum(
            item.bbox.width * item.bbox.height for item in group.components
        )
        group_area = group.bbox.width * group.bbox.height
        compactness = _unit(component_area / group_area)
        component_density = _unit(len(group.components) / max(1, group.bbox.width / 35))
        shapes = _shape_features(image, group.bbox)
        area_ratio = group_area / (test_region.width * test_region.height)
        isolated = _unit(
            (1 - min(1.0, len(group.components) / 4)) * 0.55
            + (1 - min(1.0, area_ratio / 0.035)) * 0.45
        )
        left_distance = (group.bbox.x - test_region.x) / test_region.width
        right_distance = (
            test_region.x + test_region.width - group.bbox.x - group.bbox.width
        ) / test_region.width
        margin_proximity = _unit(1 - min(left_distance, right_distance) / 0.18)
        overlap = max(
            (_intersection_ratio(group.bbox, space) for space in answer_spaces),
            default=0.0,
        )
        mean_saturations = tuple(item.ink.mean_saturation for item in base)
        return EvidenceFeaturesV2(
            mean_saturation=ink.mean_saturation,
            chromatic_pixel_ratio=_unit(
                ink.red_foreground_ratio + ink.blue_foreground_ratio
            ),
            color_consistency=_unit(
                1 - (pstdev(mean_saturations) if len(mean_saturations) > 1 else 0)
            ),
            foreground_ratio=ink.foreground_ratio,
            ink_background_contrast=ink.local_contrast,
            component_density=component_density,
            compactness=compactness,
            fragmentation=_unit(fmean(item.geometry.fragmentation for item in base)),
            alignment=alignment,
            isolated_small_mark_likelihood=isolated,
            diagonal_dominance=shapes[0],
            crossing_stroke_likelihood=shapes[1],
            underline_likelihood=shapes[2],
            circle_likelihood=shapes[3],
            correction_stroke_likelihood=shapes[4],
            ocr_confidence=fmean(confidences) if confidences else None,
            ocr_token_count=sum(
                bool(item.source_word_indices) for item in group.components
            ),
            ocr_regularity=_unit(fmean(regularities)),
            hierarchy_consistency=alignment,
            text_density=_unit(fmean(densities)),
            normalized_x=_unit((group.bbox.x - test_region.x) / test_region.width),
            normalized_y=_unit((group.bbox.y - test_region.y) / test_region.height),
            margin_proximity=margin_proximity,
            answer_space_overlap=overlap,
            group_width_ratio=_unit(group.bbox.width / test_region.width),
            group_height_ratio=_unit(group.bbox.height / test_region.height),
        )


def _shape_features(
    image: np.ndarray, bbox: BoundingBox
) -> tuple[float, float, float, float, float]:
    crop = image[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
    if crop.size == 0:
        raise EvidenceSeparationError("V2 evidence crop is empty")
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=max(8, min(crop.shape[:2]) // 4),
            minLineLength=max(5, min(crop.shape[:2]) // 3),
            maxLineGap=max(2, min(crop.shape[:2]) // 8),
        )
        angles: list[float] = []
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                angles.append(abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))))
        diagonals = sum(20 <= angle <= 70 or 110 <= angle <= 160 for angle in angles)
        horizontals = sum(angle <= 12 or angle >= 168 for angle in angles)
        positive = any(20 <= angle <= 70 for angle in angles)
        negative = any(110 <= angle <= 160 for angle in angles)
        crossing = float(positive and negative)
        contours, _ = cv2.findContours(
            np.where(gray < 210, 255, 0).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        circularities = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area >= 8 and perimeter > 0:
                circularities.append(4 * np.pi * area / (perimeter * perimeter))
    except cv2.error as error:
        raise EvidenceSeparationError(
            "V2 stroke features could not be measured"
        ) from error
    diagonal = _unit(diagonals / max(1, len(angles)))
    underline = _unit(horizontals / max(1, len(angles)))
    circle = _unit(max(circularities, default=0.0))
    correction = _unit(max(crossing, diagonal * underline))
    return diagonal, crossing, underline, circle, correction


def _intersection_ratio(left: BoundingBox, right: BoundingBox) -> float:
    width = max(
        0,
        min(left.x + left.width, right.x + right.width) - max(left.x, right.x),
    )
    height = max(
        0,
        min(left.y + left.height, right.y + right.height) - max(left.y, right.y),
    )
    return width * height / (left.width * left.height)


def _unit(value: float) -> float:
    return min(1.0, max(0.0, value))

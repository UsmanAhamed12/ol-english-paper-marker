"""High-recall local discovery of teacher-risk evidence candidates."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import hypot

import cv2
import numpy as np

from app.domain.models.paper import PaperPage
from app.evaluation.teacher_evidence.models import (
    TeacherCandidateFeatures,
    TeacherDiscoveryCategory,
    TeacherDiscoverySignal,
)
from app.ocr.models import BoundingBox, OCRPageResult, OCRWord
from app.structure.marker_detector import detect_marker_candidates
from app.structure.models import TestMarkerCandidate


@dataclass(frozen=True, slots=True)
class TeacherEvidenceProposal:
    """One unlabeled proposal with local visual and structure provenance."""

    paper_alias: str
    page_number: int
    test_number: int | None
    region: BoundingBox
    candidate_component: BoundingBox
    category: TeacherDiscoveryCategory
    signals: tuple[TeacherDiscoverySignal, ...]
    features: TeacherCandidateFeatures
    reason: str
    score: float


DEFAULT_SELECTION_TARGETS = {
    TeacherDiscoveryCategory.CHROMATIC: 12,
    TeacherDiscoveryCategory.MARGIN_SCORE: 10,
    TeacherDiscoveryCategory.COMPACT_GEOMETRY: 10,
    TeacherDiscoveryCategory.MIXED: 8,
    TeacherDiscoveryCategory.AMBIGUOUS: 3,
    TeacherDiscoveryCategory.HARD_NEGATIVE: 5,
}


def discover_teacher_evidence_candidates(
    page: PaperPage,
    ocr_result: OCRPageResult,
    image: np.ndarray,
    *,
    paper_alias: str,
) -> tuple[TeacherEvidenceProposal, ...]:
    """Generate high-recall hints without assigning any human evidence class."""

    if image.ndim != 3 or image.shape[:2] != (page.height, page.width):
        raise ValueError("Teacher discovery image does not match PaperPage")
    words = ocr_result.evidence.words if ocr_result.evidence is not None else ()
    markers = detect_marker_candidates(page, ocr_result)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    chromatic = ((hsv[:, :, 1] >= 28) & (hsv[:, :, 2] <= 252)).astype(np.uint8)
    dark = (gray <= 210).astype(np.uint8)
    combined = np.maximum(chromatic, dark)
    joined = cv2.morphologyEx(
        combined * 255,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    proposals = [
        proposal
        for contour in contours
        if (
            proposal := _proposal_from_contour(
                contour,
                page=page,
                image=image,
                hsv=hsv,
                gray=gray,
                chromatic=chromatic,
                words=words,
                markers=markers,
                paper_alias=paper_alias,
            )
        )
        is not None
    ]
    proposals.extend(
        _hard_negative_controls(page, image, hsv, gray, words, markers, paper_alias)
    )
    return tuple(sorted(proposals, key=_proposal_order))


def suppress_teacher_candidate_duplicates(
    candidates: tuple[TeacherEvidenceProposal, ...],
    *,
    maximum_iou: float = 0.52,
) -> tuple[TeacherEvidenceProposal, ...]:
    """Deterministically suppress overlapping proposals for the same evidence."""

    if not 0 <= maximum_iou <= 1:
        raise ValueError("Maximum candidate IoU must be in [0, 1]")
    kept: list[TeacherEvidenceProposal] = []
    ordered = sorted(
        candidates,
        key=lambda item: (
            -item.score,
            -len(item.signals),
            item.paper_alias,
            item.page_number,
            item.region.y,
            item.region.x,
            item.category.value,
        ),
    )
    for candidate in ordered:
        if any(
            _same_evidence_area(candidate, existing, maximum_iou) for existing in kept
        ):
            continue
        kept.append(candidate)
    return tuple(sorted(kept, key=_proposal_order))


def select_teacher_evidence_candidates(
    candidates: tuple[TeacherEvidenceProposal, ...],
    *,
    targets: dict[TeacherDiscoveryCategory, int] | None = None,
    target_count: int = 48,
    maximum_per_paper: int = 3,
    maximum_per_page: int = 2,
) -> tuple[TeacherEvidenceProposal, ...]:
    """Balance discovery context without using or manufacturing labels."""

    targets = targets or DEFAULT_SELECTION_TARGETS
    if (
        target_count <= 0
        or maximum_per_paper <= 0
        or maximum_per_page <= 0
        or any(value < 0 for value in targets.values())
    ):
        raise ValueError("Teacher candidate selection limits must be positive")
    selected: list[TeacherEvidenceProposal] = []
    paper_counts: Counter[str] = Counter()
    page_counts: Counter[tuple[str, int]] = Counter()

    def eligible(item: TeacherEvidenceProposal) -> bool:
        key = (item.paper_alias, item.page_number)
        return (
            paper_counts[item.paper_alias] < maximum_per_paper
            and page_counts[key] < maximum_per_page
            and not any(_same_evidence_area(item, other, 0.42) for other in selected)
        )

    def add(item: TeacherEvidenceProposal) -> None:
        selected.append(item)
        paper_counts[item.paper_alias] += 1
        page_counts[(item.paper_alias, item.page_number)] += 1

    for category in TeacherDiscoveryCategory:
        quota = targets.get(category, 0)
        options = sorted(
            (item for item in candidates if item.category is category),
            key=_selection_order,
        )
        for item in options:
            if sum(chosen.category is category for chosen in selected) >= quota:
                break
            if eligible(item):
                add(item)

    for item in sorted(candidates, key=_selection_order):
        if len(selected) >= target_count:
            break
        if item not in selected and eligible(item):
            add(item)
    return tuple(sorted(selected[:target_count], key=_proposal_order))


def contextual_crop(
    component: BoundingBox,
    page_width: int,
    page_height: int,
) -> BoundingBox:
    """Add bounded context around a suspicious component without resizing it."""

    target_width = min(page_width, max(620, min(940, component.width + 420)))
    target_height = min(page_height, max(380, min(720, component.height + 320)))
    x = min(
        max(0, component.x - (target_width - component.width) // 2),
        page_width - target_width,
    )
    y = min(
        max(0, component.y - (target_height - component.height) // 2),
        page_height - target_height,
    )
    return BoundingBox(x=x, y=y, width=target_width, height=target_height)


def _proposal_from_contour(
    contour: np.ndarray,
    *,
    page: PaperPage,
    image: np.ndarray,
    hsv: np.ndarray,
    gray: np.ndarray,
    chromatic: np.ndarray,
    words: tuple[OCRWord, ...],
    markers: tuple[TestMarkerCandidate, ...],
    paper_alias: str,
) -> TeacherEvidenceProposal | None:
    x, y, width, height = (int(value) for value in cv2.boundingRect(contour))
    area = float(cv2.contourArea(contour))
    page_area = page.width * page.height
    if (
        width < 4
        or height < 4
        or area < 16
        or area > page_area * 0.008
        or width > page.width * 0.28
        or height > page.height * 0.18
    ):
        return None
    component = BoundingBox(x=x, y=y, width=width, height=height)
    region = contextual_crop(component, page.width, page.height)
    features = _features(component, region, image, hsv, gray, chromatic, words, contour)
    signals = _signals(component, features, words, page)
    if not signals:
        return None
    category = _category(signals)
    score = _score(category, signals, features)
    return TeacherEvidenceProposal(
        paper_alias=paper_alias,
        page_number=page.page_number,
        test_number=_nearest_test_number(component, markers),
        region=region,
        candidate_component=component,
        category=category,
        signals=signals,
        features=features,
        reason=f"{category.value}+multi_signal_context",
        score=score,
    )


def _features(
    component: BoundingBox,
    region: BoundingBox,
    image: np.ndarray,
    hsv: np.ndarray,
    gray: np.ndarray,
    chromatic: np.ndarray,
    words: tuple[OCRWord, ...],
    contour: np.ndarray,
) -> TeacherCandidateFeatures:
    ys = slice(component.y, component.y + component.height)
    xs = slice(component.x, component.x + component.width)
    patch_gray = gray[ys, xs]
    patch_hsv = hsv[ys, xs]
    patch_chromatic = chromatic[ys, xs]
    foreground = patch_gray < 225
    foreground_count = max(1, int(np.count_nonzero(foreground)))
    edges = cv2.Canny(patch_gray, 60, 160)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=8,
        minLineLength=max(5, min(component.width, component.height) // 3),
        maxLineGap=5,
    )
    angled = 0
    if lines is not None:
        for line in lines.reshape(-1, 4):
            x1, y1, x2, y2 = (int(value) for value in line)
            angle = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
            if 18 <= angle <= 162 and not 72 <= angle <= 108:
                angled += 1
    nearby = tuple(word for word in words if _distance(component, word.bbox) <= 180)
    closest = min((_distance(component, word.bbox) for word in words), default=500.0)
    contour_area = max(1.0, float(cv2.contourArea(contour)))
    region_gray = gray[
        region.y : region.y + region.height,
        region.x : region.x + region.width,
    ]
    return TeacherCandidateFeatures(
        component_area_ratio=min(1.0, contour_area / (image.shape[0] * image.shape[1])),
        chromatic_foreground_ratio=min(
            1.0,
            float(np.count_nonzero(patch_chromatic & foreground)) / foreground_count,
        ),
        mean_saturation=min(
            1.0,
            float(np.mean(patch_hsv[:, :, 1][foreground])) / 255
            if np.any(foreground)
            else 0.0,
        ),
        foreground_ratio=min(1.0, foreground_count / patch_gray.size),
        edge_density=min(1.0, float(np.count_nonzero(edges)) / edges.size),
        local_whitespace_ratio=min(
            1.0, float(np.count_nonzero(region_gray >= 235)) / region_gray.size
        ),
        margin_proximity=_margin_proximity(component, image.shape[1]),
        ocr_proximity=max(0.0, 1.0 - closest / 500),
        nearby_ocr_words=len(nearby),
        angled_line_count=angled,
    )


def _signals(
    component: BoundingBox,
    features: TeacherCandidateFeatures,
    words: tuple[OCRWord, ...],
    page: PaperPage,
) -> tuple[TeacherDiscoverySignal, ...]:
    signals: list[TeacherDiscoverySignal] = []
    if features.chromatic_foreground_ratio >= 0.08 or features.mean_saturation >= 0.10:
        signals.append(TeacherDiscoverySignal.CHROMATIC_INK)
    if (
        features.component_area_ratio <= 0.0015
        and features.local_whitespace_ratio >= 0.45
    ):
        signals.append(TeacherDiscoverySignal.SMALL_ISOLATED_COMPONENT)
    if features.margin_proximity >= 0.58:
        signals.append(TeacherDiscoverySignal.MARGIN_ACTIVITY)
    aspect = component.width / component.height
    if (
        features.component_area_ratio <= 0.001
        and 0.25 <= aspect <= 4.0
        and (features.margin_proximity >= 0.35 or features.ocr_proximity >= 0.45)
    ):
        signals.append(TeacherDiscoverySignal.SCORE_LIKE_GEOMETRY)
    if features.angled_line_count >= 2 and features.edge_density >= 0.04:
        signals.append(TeacherDiscoverySignal.TICK_CROSS_GEOMETRY)
    if aspect >= 3.5 or features.angled_line_count >= 3:
        signals.append(TeacherDiscoverySignal.CORRECTION_STROKE)
    nearby = tuple(word for word in words if _distance(component, word.bbox) <= 180)
    if nearby:
        signals.append(TeacherDiscoverySignal.OCR_CONTEXT_NEARBY)
    high_confidence = any(
        word.confidence is not None and word.confidence >= 0.78 for word in nearby
    )
    if high_confidence:
        signals.append(TeacherDiscoverySignal.HIGH_CONFIDENCE_PRINT_NEARBY)
    if TeacherDiscoverySignal.CHROMATIC_INK in signals and nearby:
        signals.append(TeacherDiscoverySignal.STUDENT_TEACHER_MIXED_RISK)
    if TeacherDiscoverySignal.CHROMATIC_INK in signals and high_confidence:
        signals.append(TeacherDiscoverySignal.PRINT_TEACHER_MIXED_RISK)
    if len(nearby) >= 8:
        signals.append(TeacherDiscoverySignal.PARAGRAPH_CONTEXT)
    elif 1 <= len(nearby) <= 4:
        signals.append(TeacherDiscoverySignal.SHORT_ANSWER_CONTEXT)
    if (
        component.y <= page.height * 0.12
        or component.y + component.height >= page.height * 0.88
    ):
        signals.append(TeacherDiscoverySignal.MARGIN_ACTIVITY)
    return tuple(dict.fromkeys(signals))


def _category(
    signals: tuple[TeacherDiscoverySignal, ...],
) -> TeacherDiscoveryCategory:
    signal_set = set(signals)
    if signal_set & {
        TeacherDiscoverySignal.STUDENT_TEACHER_MIXED_RISK,
        TeacherDiscoverySignal.PRINT_TEACHER_MIXED_RISK,
    }:
        return TeacherDiscoveryCategory.MIXED
    if signal_set & {
        TeacherDiscoverySignal.TICK_CROSS_GEOMETRY,
        TeacherDiscoverySignal.CORRECTION_STROKE,
    }:
        return TeacherDiscoveryCategory.COMPACT_GEOMETRY
    if signal_set & {
        TeacherDiscoverySignal.MARGIN_ACTIVITY,
        TeacherDiscoverySignal.SCORE_LIKE_GEOMETRY,
    }:
        return TeacherDiscoveryCategory.MARGIN_SCORE
    if TeacherDiscoverySignal.CHROMATIC_INK in signal_set:
        return TeacherDiscoveryCategory.CHROMATIC
    return TeacherDiscoveryCategory.AMBIGUOUS


def _score(
    category: TeacherDiscoveryCategory,
    signals: tuple[TeacherDiscoverySignal, ...],
    features: TeacherCandidateFeatures,
) -> float:
    category_bonus = {
        TeacherDiscoveryCategory.MIXED: 0.22,
        TeacherDiscoveryCategory.COMPACT_GEOMETRY: 0.18,
        TeacherDiscoveryCategory.MARGIN_SCORE: 0.16,
        TeacherDiscoveryCategory.CHROMATIC: 0.14,
        TeacherDiscoveryCategory.AMBIGUOUS: 0.08,
        TeacherDiscoveryCategory.HARD_NEGATIVE: 0.02,
    }[category]
    return min(
        1.0,
        category_bonus
        + min(0.35, len(signals) * 0.055)
        + 0.12 * features.chromatic_foreground_ratio
        + 0.10 * features.margin_proximity
        + 0.08 * min(1.0, features.angled_line_count / 4),
    )


def _hard_negative_controls(
    page: PaperPage,
    image: np.ndarray,
    hsv: np.ndarray,
    gray: np.ndarray,
    words: tuple[OCRWord, ...],
    markers: tuple[TestMarkerCandidate, ...],
    alias: str,
) -> tuple[TeacherEvidenceProposal, ...]:
    controls = sorted(
        (
            word
            for word in words
            if word.confidence is not None
            and word.confidence >= 0.90
            and word.bbox.width >= 15
            and word.bbox.height >= 8
        ),
        key=lambda word: (
            -float(word.confidence or 0),
            word.bbox.y,
            word.bbox.x,
        ),
    )[:2]
    result = []
    for word in controls:
        component = word.bbox
        region = contextual_crop(component, page.width, page.height)
        features = _features(
            component,
            region,
            image,
            hsv,
            gray,
            np.zeros_like(gray, dtype=np.uint8),
            words,
            np.array(
                [
                    [[component.x, component.y]],
                    [[component.x + component.width, component.y]],
                    [[component.x + component.width, component.y + component.height]],
                    [[component.x, component.y + component.height]],
                ],
                dtype=np.int32,
            ),
        )
        signals = (
            TeacherDiscoverySignal.PRINTED_CONTROL,
            TeacherDiscoverySignal.HIGH_CONFIDENCE_PRINT_NEARBY,
        )
        result.append(
            TeacherEvidenceProposal(
                paper_alias=alias,
                page_number=page.page_number,
                test_number=_nearest_test_number(component, markers),
                region=region,
                candidate_component=component,
                category=TeacherDiscoveryCategory.HARD_NEGATIVE,
                signals=signals,
                features=features,
                reason="printed_or_guide_hard_negative+context_control",
                score=0.25 + 0.15 * features.margin_proximity,
            )
        )
    return tuple(result)


def _nearest_test_number(
    box: BoundingBox, markers: tuple[TestMarkerCandidate, ...]
) -> int | None:
    preceding = [marker for marker in markers if marker.bbox.y <= box.y]
    return (
        int(max(preceding, key=lambda marker: marker.bbox.y).test_number)
        if preceding
        else None
    )


def _distance(left: BoundingBox, right: BoundingBox) -> float:
    left_x = left.x + left.width / 2
    left_y = left.y + left.height / 2
    right_x = right.x + right.width / 2
    right_y = right.y + right.height / 2
    return hypot(left_x - right_x, left_y - right_y)


def _margin_proximity(box: BoundingBox, page_width: int) -> float:
    center = box.x + box.width / 2
    edge_distance = min(center, page_width - center)
    return max(0.0, min(1.0, 1.0 - edge_distance / (page_width * 0.5)))


def _same_evidence_area(
    left: TeacherEvidenceProposal,
    right: TeacherEvidenceProposal,
    maximum_iou: float,
) -> bool:
    if left.paper_alias != right.paper_alias or left.page_number != right.page_number:
        return False
    overlap = _iou(left.region, right.region)
    containment = _intersection(left.region, right.region) / min(
        left.region.width * left.region.height,
        right.region.width * right.region.height,
    )
    center_distance = _distance(left.candidate_component, right.candidate_component)
    close_centers = center_distance < 0.28 * min(left.region.width, left.region.height)
    related_signals = bool(set(left.signals) & set(right.signals))
    return (
        overlap > maximum_iou
        or containment > 0.84
        or (close_centers and related_signals)
    )


def _intersection(left: BoundingBox, right: BoundingBox) -> int:
    width = max(
        0,
        min(left.x + left.width, right.x + right.width) - max(left.x, right.x),
    )
    height = max(
        0,
        min(left.y + left.height, right.y + right.height) - max(left.y, right.y),
    )
    return width * height


def _iou(left: BoundingBox, right: BoundingBox) -> float:
    intersection = _intersection(left, right)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union else 0.0


def _proposal_order(item: TeacherEvidenceProposal) -> tuple[object, ...]:
    return (
        item.paper_alias,
        item.page_number,
        item.region.y,
        item.region.x,
        item.category.value,
        -item.score,
    )


def _selection_order(item: TeacherEvidenceProposal) -> tuple[object, ...]:
    return (
        -item.score,
        -len(item.signals),
        item.paper_alias,
        item.page_number,
        item.region.y,
        item.region.x,
    )

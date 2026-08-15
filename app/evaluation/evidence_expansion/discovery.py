"""Deterministic candidate discovery for balanced human evidence labeling."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean

import cv2
import numpy as np

from app.domain.models.paper import PaperPage
from app.evaluation.evidence_expansion.models import (
    EvidenceCandidateCategory,
    EvidenceContextTag,
)
from app.ocr.models import BoundingBox, OCRPageResult, OCRWord
from app.structure.marker_detector import detect_marker_candidates
from app.structure.models import TestMarkerCandidate


@dataclass(frozen=True, slots=True)
class DiscoveredEvidenceCandidate:
    """A sampling proposal; its category is explicitly not ground truth."""

    paper_alias: str
    page_number: int
    test_number: int | None
    bbox: BoundingBox
    category: EvidenceCandidateCategory
    context_tags: tuple[EvidenceContextTag, ...]
    reason: str
    score: float


def discover_page_candidates(
    page: PaperPage,
    ocr_result: OCRPageResult,
    image: np.ndarray,
    *,
    paper_alias: str,
) -> tuple[DiscoveredEvidenceCandidate, ...]:
    """Propose one fixed-context crop for each evidence discovery strategy."""

    if image.ndim != 3 or image.shape[:2] != (page.height, page.width):
        raise ValueError("Candidate discovery image does not match PaperPage")
    words = ocr_result.evidence.words if ocr_result.evidence is not None else ()
    lines = _group_lines(words)
    markers = detect_marker_candidates(page, ocr_result)
    proposals: list[DiscoveredEvidenceCandidate] = []

    if lines:
        printed = max(lines, key=_printed_line_score)
        proposals.append(
            _line_candidate(
                page,
                paper_alias,
                printed,
                EvidenceCandidateCategory.PRINTED,
                "regular_high_confidence_ocr_line",
                _printed_line_score(printed),
                markers,
            )
        )
        student = max(lines, key=_irregular_line_score)
        proposals.append(
            _line_candidate(
                page,
                paper_alias,
                student,
                EvidenceCandidateCategory.STUDENT,
                "irregular_or_low_confidence_ocr_line",
                _irregular_line_score(student),
                markers,
            )
        )

    chromatic = _chromatic_windows(image)
    if chromatic:
        box, score = max(chromatic, key=lambda item: item[1])
        proposals.append(
            _candidate(
                page,
                paper_alias,
                _context_box(box, page.width, page.height),
                EvidenceCandidateCategory.TEACHER,
                "isolated_chromatic_component_risk",
                score,
                markers,
                (EvidenceContextTag.COLORED_INK,),
            )
        )

    mixed = _mixed_window(lines, chromatic)
    if mixed is not None:
        box, score = mixed
        proposals.append(
            _candidate(
                page,
                paper_alias,
                _context_box(box, page.width, page.height),
                EvidenceCandidateCategory.MIXED,
                "ocr_and_chromatic_evidence_overlap",
                score,
                markers,
                (EvidenceContextTag.COLORED_INK,),
            )
        )

    blank_box, blank_score = _blank_band(image)
    proposals.append(
        _candidate(
            page,
            paper_alias,
            blank_box,
            EvidenceCandidateCategory.BLANK,
            "sparse_band_with_answer_guide_risk",
            blank_score,
            markers,
            (EvidenceContextTag.SPARSE, EvidenceContextTag.WRITING_GUIDES),
        )
    )
    return reject_overlapping_candidates(tuple(proposals))


def select_balanced_candidates(
    candidates: tuple[DiscoveredEvidenceCandidate, ...],
    *,
    quotas: dict[EvidenceCandidateCategory, int],
    maximum_per_paper: int = 5,
) -> tuple[DiscoveredEvidenceCandidate, ...]:
    """Select fixed category quotas without using labels or transcription text."""

    if maximum_per_paper <= 0 or any(value < 0 for value in quotas.values()):
        raise ValueError("Candidate sampling limits must be non-negative")
    per_paper: defaultdict[str, int] = defaultdict(int)
    selected: list[DiscoveredEvidenceCandidate] = []
    for category in EvidenceCandidateCategory:
        available = sorted(
            (item for item in candidates if item.category is category),
            key=lambda item: (
                -item.score,
                item.paper_alias,
                item.page_number,
                item.bbox.y,
                item.bbox.x,
            ),
        )
        count = 0
        for item in available:
            if count >= quotas.get(category, 0):
                break
            if per_paper[item.paper_alias] >= maximum_per_paper:
                continue
            if any(_same_visual_area(item, chosen) for chosen in selected):
                continue
            selected.append(item)
            per_paper[item.paper_alias] += 1
            count += 1
    return tuple(
        sorted(
            selected,
            key=lambda item: (
                item.paper_alias,
                item.page_number,
                item.bbox.y,
                item.bbox.x,
                item.category.value,
            ),
        )
    )


def reject_overlapping_candidates(
    candidates: tuple[DiscoveredEvidenceCandidate, ...],
    *,
    maximum_iou: float = 0.72,
) -> tuple[DiscoveredEvidenceCandidate, ...]:
    """Reject near-duplicate page regions deterministically."""

    if not 0 <= maximum_iou <= 1:
        raise ValueError("Maximum candidate IoU must be in [0, 1]")
    kept: list[DiscoveredEvidenceCandidate] = []
    for item in sorted(candidates, key=lambda value: (-value.score, value.category)):
        if any(
            item.paper_alias == other.paper_alias
            and item.page_number == other.page_number
            and _iou(item.bbox, other.bbox) > maximum_iou
            for other in kept
        ):
            continue
        kept.append(item)
    return tuple(sorted(kept, key=lambda value: value.category.value))


def _group_lines(words: tuple[OCRWord, ...]) -> tuple[tuple[OCRWord, ...], ...]:
    grouped: defaultdict[tuple[int, int, int], list[OCRWord]] = defaultdict(list)
    for word in words:
        grouped[
            (
                word.block_number or word.bbox.y + 1,
                word.paragraph_number or 1,
                word.line_number or word.bbox.y + 1,
            )
        ].append(word)
    return tuple(
        tuple(sorted(line, key=lambda item: (item.bbox.x, item.word_number or 0)))
        for _, line in sorted(
            grouped.items(), key=lambda item: min(word.bbox.y for word in item[1])
        )
        if line
    )


def _printed_line_score(line: tuple[OCRWord, ...]) -> float:
    confidence = [word.confidence for word in line if word.confidence is not None]
    return (fmean(confidence) if confidence else 0.0) * min(1.0, len(line) / 6)


def _irregular_line_score(line: tuple[OCRWord, ...]) -> float:
    confidence = [word.confidence for word in line if word.confidence is not None]
    low = 1 - (fmean(confidence) if confidence else 0.5)
    heights = [word.bbox.height for word in line]
    irregular = 0.0 if len(heights) < 2 else min(1.0, np.std(heights) / 20)
    return 0.65 * low + 0.35 * float(irregular)


def _line_candidate(
    page: PaperPage,
    alias: str,
    line: tuple[OCRWord, ...],
    category: EvidenceCandidateCategory,
    reason: str,
    score: float,
    markers: tuple[TestMarkerCandidate, ...],
) -> DiscoveredEvidenceCandidate:
    box = _union(tuple(word.bbox for word in line))
    tags = (
        EvidenceContextTag.PARAGRAPH
        if len(line) >= 8
        else EvidenceContextTag.SHORT_ANSWER,
    )
    return _candidate(
        page,
        alias,
        _context_box(box, page.width, page.height),
        category,
        reason,
        score,
        markers,
        tags,
    )


def _candidate(
    page: PaperPage,
    alias: str,
    box: BoundingBox,
    category: EvidenceCandidateCategory,
    reason: str,
    score: float,
    markers: tuple[TestMarkerCandidate, ...],
    tags: tuple[EvidenceContextTag, ...],
) -> DiscoveredEvidenceCandidate:
    test_number = _nearest_test_number(box, markers)
    extra = EvidenceContextTag.MARGIN if box.x < page.width * 0.12 else None
    context = tuple(dict.fromkeys((*tags, *((extra,) if extra else ()))))
    return DiscoveredEvidenceCandidate(
        paper_alias=alias,
        page_number=page.page_number,
        test_number=test_number,
        bbox=box,
        category=category,
        context_tags=context,
        reason=reason,
        score=max(0.0, min(1.0, score)),
    )


def _nearest_test_number(
    box: BoundingBox, markers: tuple[TestMarkerCandidate, ...]
) -> int | None:
    preceding = [marker for marker in markers if marker.bbox.y <= box.y]
    if not preceding:
        return None
    return int(max(preceding, key=lambda marker: marker.bbox.y).test_number)


def _chromatic_windows(image: np.ndarray) -> tuple[tuple[BoundingBox, float], ...]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] >= 55) & (hsv[:, :, 2] <= 250)).astype(np.uint8) * 255
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined)
    height, width = image.shape[:2]
    result = []
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        if 15 <= area <= width * height * 0.02 and w >= 3 and h >= 3:
            isolation = 1.0 if x < width * 0.18 or x + w > width * 0.82 else 0.55
            result.append((BoundingBox(x=x, y=y, width=w, height=h), isolation))
    return tuple(result)


def _mixed_window(
    lines: tuple[tuple[OCRWord, ...], ...],
    chromatic: tuple[tuple[BoundingBox, float], ...],
) -> tuple[BoundingBox, float] | None:
    best: tuple[BoundingBox, float] | None = None
    for line in lines:
        line_box = _union(tuple(word.bbox for word in line))
        for color_box, score in chromatic:
            distance = abs(
                (line_box.y + line_box.height / 2)
                - (color_box.y + color_box.height / 2)
            )
            if distance <= max(120, line_box.height * 3):
                union = _union((line_box, color_box))
                candidate = (union, min(1.0, 0.5 + score / 2))
                if best is None or candidate[1] > best[1]:
                    best = candidate
    return best


def _blank_band(image: np.ndarray) -> tuple[BoundingBox, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    band_height = max(180, height // 8)
    margin = max(20, width // 18)
    best_y, best_score = 0, -1.0
    for y in range(0, max(1, height - band_height + 1), max(40, band_height // 3)):
        crop = gray[y : y + band_height, margin : width - margin]
        foreground = float(np.count_nonzero(crop < 220)) / crop.size
        horizontal = cv2.morphologyEx(
            (crop < 210).astype(np.uint8) * 255,
            cv2.MORPH_OPEN,
            np.ones((1, max(20, width // 25)), np.uint8),
        )
        guide = min(1.0, np.count_nonzero(horizontal) / max(1, crop.size * 0.02))
        score = max(0.0, 1 - foreground * 5) * 0.7 + guide * 0.3
        if score > best_score:
            best_y, best_score = y, score
    return (
        BoundingBox(
            x=margin,
            y=best_y,
            width=max(1, width - margin * 2),
            height=min(band_height, height - best_y),
        ),
        best_score,
    )


def _context_box(box: BoundingBox, page_width: int, page_height: int) -> BoundingBox:
    target_width = min(page_width, max(700, box.width + 240))
    target_height = min(page_height, max(360, box.height + 220))
    x = min(max(0, box.x - (target_width - box.width) // 2), page_width - target_width)
    y = min(
        max(0, box.y - (target_height - box.height) // 2),
        page_height - target_height,
    )
    return BoundingBox(x=x, y=y, width=target_width, height=target_height)


def _union(boxes: tuple[BoundingBox, ...]) -> BoundingBox:
    left = min(box.x for box in boxes)
    top = min(box.y for box in boxes)
    right = max(box.x + box.width for box in boxes)
    bottom = max(box.y + box.height for box in boxes)
    return BoundingBox(x=left, y=top, width=right - left, height=bottom - top)


def _same_visual_area(
    left: DiscoveredEvidenceCandidate, right: DiscoveredEvidenceCandidate
) -> bool:
    return (
        left.paper_alias == right.paper_alias
        and left.page_number == right.page_number
        and _iou(left.bbox, right.bbox) > 0.55
    )


def _iou(left: BoundingBox, right: BoundingBox) -> float:
    width = max(
        0,
        min(left.x + left.width, right.x + right.width) - max(left.x, right.x),
    )
    height = max(
        0,
        min(left.y + left.height, right.y + right.height) - max(left.y, right.y),
    )
    intersection = width * height
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union else 0.0

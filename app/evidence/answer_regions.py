"""Deterministic answer-space candidates from guides and student evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.exceptions import EvidenceSeparationError
from app.domain.models.paper import PaperPage
from app.evidence.models import (
    AnswerRegionSignal,
    EvidenceRegion,
    EvidenceType,
    StudentAnswerRegion,
)
from app.ocr.models import BoundingBox

ANSWER_REGION_STRATEGY_VERSION = "answer-region-v1"


@dataclass(frozen=True, slots=True)
class _AnswerCandidate:
    bbox: BoundingBox
    signals: tuple[AnswerRegionSignal, ...]
    source_indices: tuple[int, ...]
    confidence: float


class AnswerRegionDetector:
    """Locate conservative writing-space and student-cluster candidates."""

    def detect(
        self,
        page: PaperPage,
        *,
        test_number: int,
        region_bbox: BoundingBox,
        evidence_regions: tuple[EvidenceRegion, ...],
    ) -> tuple[StudentAnswerRegion, ...]:
        """Return page-local candidates without claiming handwriting content."""

        source_hash = _sha256(page.image_path)
        image = cv2.imread(str(page.image_path), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3:
            raise EvidenceSeparationError("Answer-region source image is invalid")
        candidates = list(_guide_candidates(image, region_bbox, evidence_regions))
        candidates.extend(_student_clusters(region_bbox, evidence_regions))
        merged = _merge_candidates(candidates, region_bbox)
        if _sha256(page.image_path) != source_hash:
            raise EvidenceSeparationError(
                "Canonical page changed during answer-region detection"
            )
        return tuple(
            StudentAnswerRegion(
                paper_id=page.paper_id,
                page_number=page.page_number,
                test_number=test_number,
                bbox=candidate.bbox,
                confidence=candidate.confidence,
                signals=candidate.signals,
                source_evidence_indices=candidate.source_indices,
                source_image_path=page.image_path,
                detection_strategy=ANSWER_REGION_STRATEGY_VERSION,
            )
            for candidate in merged
        )


def _guide_candidates(
    image: np.ndarray,
    region: BoundingBox,
    evidence: tuple[EvidenceRegion, ...],
) -> tuple[_AnswerCandidate, ...]:
    crop = image[
        region.y : region.y + region.height, region.x : region.x + region.width
    ]
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        binary = np.where(gray < 190, 255, 0).astype(np.uint8)
        joined = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)),
        )
        horizontal = cv2.morphologyEx(
            joined,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, region.width // 8), 1)),
        )
        contours, _ = cv2.findContours(
            horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
    except cv2.error as error:
        raise EvidenceSeparationError(
            "Answer writing guides could not be measured"
        ) from error
    lines = sorted(
        (
            BoundingBox(
                x=region.x + x,
                y=region.y + y,
                width=width,
                height=height,
            )
            for contour in contours
            for x, y, width, height in (cv2.boundingRect(contour),)
            if width >= region.width * 0.20
            and height <= max(18, int(region.height * 0.02))
        ),
        key=lambda box: (box.y, box.x),
    )
    groups: list[list[BoundingBox]] = []
    maximum_gap = max(45, int(region.height * 0.25))
    for line in lines:
        if not groups or line.y - groups[-1][-1].y > maximum_gap:
            groups.append([line])
        else:
            groups[-1].append(line)

    output: list[_AnswerCandidate] = []
    for group in groups:
        if len(group) < 2:
            continue
        box = _guide_group_box(group, region)
        overlapping_students = tuple(
            index
            for index, item in enumerate(evidence)
            if item.evidence_type is EvidenceType.STUDENT_CANDIDATE
            and _intersects(box, item.bbox)
        )
        printed_area = sum(
            _intersection_area(box, item.bbox)
            for item in evidence
            if item.evidence_type is EvidenceType.PRINTED
        )
        signals = [AnswerRegionSignal.WRITING_GUIDES]
        if printed_area / (box.width * box.height) < 0.15:
            signals.append(AnswerRegionSignal.LOW_PRINTED_DENSITY)
        if overlapping_students:
            signals.append(AnswerRegionSignal.STUDENT_EVIDENCE_CLUSTER)
        else:
            signals.append(AnswerRegionSignal.BLANK_WRITING_SPACE)
        output.append(
            _AnswerCandidate(
                bbox=box,
                signals=tuple(signals),
                source_indices=overlapping_students,
                confidence=0.82 if overlapping_students else 0.68,
            )
        )
    return tuple(output)


def _student_clusters(
    region: BoundingBox,
    evidence: tuple[EvidenceRegion, ...],
) -> tuple[_AnswerCandidate, ...]:
    return tuple(
        _AnswerCandidate(
            bbox=_expand(item.bbox, region, 18),
            signals=(AnswerRegionSignal.STUDENT_EVIDENCE_CLUSTER,),
            source_indices=(index,),
            confidence=0.78,
        )
        for index, item in enumerate(evidence)
        if item.evidence_type is EvidenceType.STUDENT_CANDIDATE
    )


def _merge_candidates(
    candidates: list[_AnswerCandidate], region: BoundingBox
) -> tuple[_AnswerCandidate, ...]:
    pending = sorted(candidates, key=lambda item: (item.bbox.y, item.bbox.x))
    merged: list[_AnswerCandidate] = []
    for candidate in pending:
        if not merged or not _intersects(
            _expand(merged[-1].bbox, region, 12), candidate.bbox
        ):
            merged.append(candidate)
            continue
        previous = merged.pop()
        merged.append(
            _AnswerCandidate(
                bbox=_union((previous.bbox, candidate.bbox)),
                signals=tuple(dict.fromkeys((*previous.signals, *candidate.signals))),
                source_indices=tuple(
                    sorted(set((*previous.source_indices, *candidate.source_indices)))
                ),
                confidence=max(previous.confidence, candidate.confidence),
            )
        )
    return tuple(merged)


def _guide_group_box(lines: list[BoundingBox], region: BoundingBox) -> BoundingBox:
    spacing = max(20, int((lines[-1].y - lines[0].y) / max(1, len(lines) - 1)))
    left = min(line.x for line in lines)
    right = max(line.x + line.width for line in lines)
    top = max(region.y, lines[0].y - spacing)
    bottom = min(region.y + region.height, lines[-1].y + spacing)
    return BoundingBox(x=left, y=top, width=right - left, height=bottom - top)


def _expand(box: BoundingBox, limit: BoundingBox, padding: int) -> BoundingBox:
    left = max(limit.x, box.x - padding)
    top = max(limit.y, box.y - padding)
    right = min(limit.x + limit.width, box.x + box.width + padding)
    bottom = min(limit.y + limit.height, box.y + box.height + padding)
    return BoundingBox(x=left, y=top, width=right - left, height=bottom - top)


def _union(boxes: tuple[BoundingBox, ...]) -> BoundingBox:
    left = min(box.x for box in boxes)
    top = min(box.y for box in boxes)
    right = max(box.x + box.width for box in boxes)
    bottom = max(box.y + box.height for box in boxes)
    return BoundingBox(x=left, y=top, width=right - left, height=bottom - top)


def _intersects(left: BoundingBox, right: BoundingBox) -> bool:
    return _intersection_area(left, right) > 0


def _intersection_area(left: BoundingBox, right: BoundingBox) -> int:
    width = max(
        0,
        min(left.x + left.width, right.x + right.width) - max(left.x, right.x),
    )
    height = max(
        0,
        min(left.y + left.height, right.y + right.height) - max(left.y, right.y),
    )
    return width * height


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

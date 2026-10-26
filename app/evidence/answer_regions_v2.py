"""Separate potential answer spaces from detected student-answer evidence."""

from __future__ import annotations

from app.evidence.models import (
    AnswerRegionSignal,
    EvidenceRegion,
    EvidenceType,
    StudentAnswerRegion,
    TestEvidence,
)
from app.evidence.v2_models import (
    AnswerSpace,
    EvidenceSeparatorV2Config,
)
from app.ocr.models import BoundingBox

ANSWER_REGION_V2_VERSION = "student-answer-region-v2"


class AnswerSpaceDetectorV2:
    """Retain V1 writing-guide geometry as possible space, never as an answer."""

    def detect(self, base: TestEvidence) -> tuple[AnswerSpace, ...]:
        """Return only writing-guide spaces with unchanged page provenance."""

        return tuple(
            AnswerSpace(
                paper_id=base.paper_id,
                page_number=base.page_number,
                test_number=base.test_number,
                bbox=region.bbox,
                confidence=region.confidence,
                source_image_path=region.source_image_path,
            )
            for region in base.answer_regions
            if AnswerRegionSignal.WRITING_GUIDES in region.signals
        )


class StudentAnswerRegionDetectorV2:
    """Create answers from student evidence; blank guides remain answer spaces."""

    def __init__(self, config: EvidenceSeparatorV2Config) -> None:
        self._config = config

    def detect(
        self,
        base: TestEvidence,
        evidence_regions: tuple[EvidenceRegion, ...],
        answer_spaces: tuple[AnswerSpace, ...],
    ) -> tuple[StudentAnswerRegion, ...]:
        """Merge compatible student groups without emitting blank-space regions."""

        candidates = [
            _Candidate(
                bbox=_expand(
                    item.bbox,
                    base.region_bbox,
                    max(2, round(item.bbox.height * self._config.answer_padding_ratio)),
                ),
                evidence_indices=(index,),
                confidence=item.confidence,
            )
            for index, item in enumerate(evidence_regions)
            if item.evidence_type is EvidenceType.STUDENT_CANDIDATE
        ]
        merged = _merge(candidates, base.region_bbox, answer_spaces, self._config)
        return tuple(
            StudentAnswerRegion(
                paper_id=base.paper_id,
                page_number=base.page_number,
                test_number=base.test_number,
                bbox=item.bbox,
                confidence=item.confidence,
                signals=(AnswerRegionSignal.STUDENT_EVIDENCE_CLUSTER,),
                source_evidence_indices=item.evidence_indices,
                source_image_path=base.evidence_regions[0].source_image_path
                if base.evidence_regions
                else next(region.source_image_path for region in base.answer_regions),
                detection_strategy=ANSWER_REGION_V2_VERSION,
            )
            for item in merged
        )


class _Candidate:
    __slots__ = ("bbox", "confidence", "evidence_indices")

    def __init__(
        self,
        *,
        bbox: BoundingBox,
        evidence_indices: tuple[int, ...],
        confidence: float,
    ) -> None:
        self.bbox = bbox
        self.evidence_indices = evidence_indices
        self.confidence = confidence


def _merge(
    candidates: list[_Candidate],
    limit: BoundingBox,
    spaces: tuple[AnswerSpace, ...],
    config: EvidenceSeparatorV2Config,
) -> tuple[_Candidate, ...]:
    pending = sorted(candidates, key=lambda item: (item.bbox.y, item.bbox.x))
    output: list[_Candidate] = []
    for candidate in pending:
        match_index = next(
            (
                index
                for index, existing in enumerate(output)
                if _merge_supported(existing.bbox, candidate.bbox, spaces, config)
            ),
            None,
        )
        if match_index is None:
            output.append(candidate)
            continue
        previous = output[match_index]
        union = _union(previous.bbox, candidate.bbox)
        if union.width * union.height > (
            limit.width * limit.height * config.maximum_answer_area_ratio
        ):
            output.append(candidate)
            continue
        output[match_index] = _Candidate(
            bbox=union,
            evidence_indices=tuple(
                sorted(set((*previous.evidence_indices, *candidate.evidence_indices)))
            ),
            confidence=max(previous.confidence, candidate.confidence),
        )
    return tuple(sorted(output, key=lambda item: (item.bbox.y, item.bbox.x)))


def _merge_supported(
    left: BoundingBox,
    right: BoundingBox,
    spaces: tuple[AnswerSpace, ...],
    config: EvidenceSeparatorV2Config,
) -> bool:
    height = max(1, min(left.height, right.height))
    vertical_overlap = (
        _overlap(left.y, left.y + left.height, right.y, right.y + right.height) / height
    )
    horizontal_gap = _gap(left.x, left.x + left.width, right.x, right.x + right.width)
    same_line = vertical_overlap >= 0.35 and horizontal_gap <= (
        config.answer_merge_gap_ratio * height
    )
    same_space = any(
        _intersection_area(space.bbox, left) > 0
        and _intersection_area(space.bbox, right) > 0
        for space in spaces
    )
    vertical_gap = _gap(left.y, left.y + left.height, right.y, right.y + right.height)
    paragraph = (
        config.enable_paragraph_merging
        and same_space
        and vertical_gap <= config.group_vertical_gap_ratio * height
    )
    return same_line or paragraph


def _expand(box: BoundingBox, limit: BoundingBox, padding: int) -> BoundingBox:
    x = max(limit.x, box.x - padding)
    y = max(limit.y, box.y - padding)
    right = min(limit.x + limit.width, box.x + box.width + padding)
    bottom = min(limit.y + limit.height, box.y + box.height + padding)
    return BoundingBox(x=x, y=y, width=right - x, height=bottom - y)


def _union(left: BoundingBox, right: BoundingBox) -> BoundingBox:
    x = min(left.x, right.x)
    y = min(left.y, right.y)
    right_edge = max(left.x + left.width, right.x + right.width)
    bottom = max(left.y + left.height, right.y + right.height)
    return BoundingBox(x=x, y=y, width=right_edge - x, height=bottom - y)


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _gap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, max(a_start, b_start) - min(a_end, b_end))


def _intersection_area(left: BoundingBox, right: BoundingBox) -> int:
    return _overlap(
        left.x, left.x + left.width, right.x, right.x + right.width
    ) * _overlap(left.y, left.y + left.height, right.y, right.y + right.height)

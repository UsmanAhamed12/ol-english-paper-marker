"""Minimal deterministic safety policy applied after EvidenceSeparator V1."""

from __future__ import annotations

from dataclasses import dataclass

from app.evidence.models import (
    AnswerRegionSignal,
    EvidenceRegion,
    EvidenceType,
    StudentAnswerRegion,
    TestEvidence,
)
from app.evidence.safety_models import (
    CandidateDisposition,
    CandidateSafetyDecision,
    SafeTestExtraction,
    SafetyReason,
    TestExtractionStatus,
)
from app.ocr.models import BoundingBox

SAFETY_POLICY_VERSION = "student-candidate-safety-v1"
SAFE_ANSWER_REGION_VERSION = "safe-student-region-v1"


@dataclass(frozen=True, slots=True)
class _RegionCandidate:
    bbox: BoundingBox
    source_indices: tuple[int, ...]
    confidence: float


class StudentCandidateSafetyFilter:
    """Reject obvious non-student evidence and quarantine teacher-risk evidence."""

    def __init__(
        self, region_builder: StudentAnswerRegionBuilder | None = None
    ) -> None:
        self._region_builder = region_builder or StudentAnswerRegionBuilder()

    def apply(self, evidence: TestEvidence) -> SafeTestExtraction:
        """Apply the fixed policy without changing V1 evidence or source images."""

        decisions = tuple(
            self._classify(index, region, evidence)
            for index, region in enumerate(evidence.evidence_regions)
        )
        answer_regions = self._region_builder.build(evidence, decisions)
        has_review = any(
            item.disposition is CandidateDisposition.REVIEW_REQUIRED
            for item in decisions
        )
        has_safe = bool(answer_regions)
        status = (
            TestExtractionStatus.REVIEW_REQUIRED
            if has_review
            else TestExtractionStatus.SAFE
            if has_safe
            else TestExtractionStatus.EMPTY
        )
        return SafeTestExtraction(
            paper_id=evidence.paper_id,
            page_number=evidence.page_number,
            test_number=evidence.test_number,
            region_bbox=evidence.region_bbox,
            status=status,
            decisions=decisions,
            answer_regions=answer_regions,
            policy_version=SAFETY_POLICY_VERSION,
        )

    def _classify(
        self,
        index: int,
        region: EvidenceRegion,
        test: TestEvidence,
    ) -> CandidateSafetyDecision:
        if region.evidence_type is not EvidenceType.STUDENT_CANDIDATE:
            return _decision(
                index,
                region,
                CandidateDisposition.REJECTED_NON_STUDENT,
                SafetyReason.V1_NON_STUDENT,
            )
        expected_space = _inside_expected_answer_space(index, region, test)
        if _is_page_or_grid_artifact(region, test.region_bbox):
            return _decision(
                index,
                region,
                CandidateDisposition.REJECTED_NON_STUDENT,
                SafetyReason.GRID_OR_PAGE_BORDER,
            )
        if _is_printed_guide(region, test.region_bbox):
            return _decision(
                index,
                region,
                CandidateDisposition.REJECTED_NON_STUDENT,
                SafetyReason.PRINTED_GUIDE_OR_RULE,
            )
        if _is_regular_print(region):
            return _decision(
                index,
                region,
                CandidateDisposition.REJECTED_NON_STUDENT,
                SafetyReason.PRINTED_REGULAR_TEXT,
            )
        if _is_tiny_noise(region, test.region_bbox, expected_space):
            return _decision(
                index,
                region,
                CandidateDisposition.REJECTED_NON_STUDENT,
                SafetyReason.TINY_NOISE,
            )
        teacher_reasons = _teacher_risk(region, test, expected_space)
        if teacher_reasons:
            return CandidateSafetyDecision(
                evidence_index=index,
                disposition=CandidateDisposition.REVIEW_REQUIRED,
                reasons=teacher_reasons,
                evidence=region,
            )
        reasons = [SafetyReason.STUDENT_SIGNALS_PRESERVED]
        if expected_space:
            reasons.append(SafetyReason.EXPECTED_ANSWER_SPACE)
        return CandidateSafetyDecision(
            evidence_index=index,
            disposition=CandidateDisposition.SAFE_STUDENT,
            reasons=tuple(reasons),
            evidence=region,
        )


class StudentAnswerRegionBuilder:
    """Build bounded, conservatively merged regions from safe V1 evidence."""

    def build(
        self,
        test: TestEvidence,
        decisions: tuple[CandidateSafetyDecision, ...],
    ) -> tuple[StudentAnswerRegion, ...]:
        """Create regions only for candidates admitted as safe student evidence."""

        return _build_answer_regions(test, decisions)


def as_filtered_test_evidence(result: SafeTestExtraction) -> TestEvidence:
    """Adapt safety outcomes to existing benchmark contracts."""

    regions = tuple(
        decision.evidence
        if decision.disposition is not CandidateDisposition.REVIEW_REQUIRED
        and not (
            decision.evidence.evidence_type is EvidenceType.STUDENT_CANDIDATE
            and decision.disposition is CandidateDisposition.REJECTED_NON_STUDENT
        )
        else decision.evidence.model_copy(
            update={
                "evidence_type": EvidenceType.UNKNOWN,
                "classification_strategy": SAFETY_POLICY_VERSION,
            }
        )
        for decision in result.decisions
    )
    return TestEvidence(
        paper_id=result.paper_id,
        page_number=result.page_number,
        test_number=result.test_number,
        region_bbox=result.region_bbox,
        evidence_regions=regions,
        answer_regions=result.answer_regions,
    )


def _decision(
    index: int,
    evidence: EvidenceRegion,
    disposition: CandidateDisposition,
    reason: SafetyReason,
) -> CandidateSafetyDecision:
    return CandidateSafetyDecision(
        evidence_index=index,
        disposition=disposition,
        reasons=(reason,),
        evidence=evidence,
    )


def _inside_expected_answer_space(
    index: int,
    region: EvidenceRegion,
    test: TestEvidence,
) -> bool:
    return any(
        AnswerRegionSignal.WRITING_GUIDES in answer.signals
        and (
            index in answer.source_evidence_indices
            or _intersection_area(region.bbox, answer.bbox) > 0
        )
        for answer in test.answer_regions
    )


def _is_regular_print(region: EvidenceRegion) -> bool:
    features = region.features
    return (
        features.ocr_confidence is not None
        and features.ocr_confidence >= 0.78
        and features.geometry.regularity >= 0.78
        and features.geometry.line_density >= 0.14
        and features.ink.mean_saturation < 0.18
    )


def _is_printed_guide(region: EvidenceRegion, test: BoundingBox) -> bool:
    aspect = region.bbox.width / region.bbox.height
    height_ratio = region.bbox.height / test.height
    return (
        aspect >= 8.0
        and height_ratio <= 0.035
        and (
            region.features.geometry.regularity >= 0.55
            or region.features.ink.foreground_ratio <= 0.18
        )
    )


def _is_page_or_grid_artifact(region: EvidenceRegion, test: BoundingBox) -> bool:
    box = region.bbox
    edge = (
        box.x <= test.x + 2
        or box.y <= test.y + 2
        or box.x + box.width >= test.x + test.width - 2
        or box.y + box.height >= test.y + test.height - 2
    )
    spans = box.width / test.width >= 0.75 or box.height / test.height >= 0.75
    return (box.width <= 2 or box.height <= 2) or (edge and spans)


def _is_tiny_noise(
    region: EvidenceRegion,
    test: BoundingBox,
    expected_space: bool,
) -> bool:
    relative_area = region.bbox.width * region.bbox.height / (test.width * test.height)
    no_word = not region.source_word_indices
    return (
        not expected_space
        and no_word
        and relative_area < 0.00004
        and region.features.ink.connected_component_count <= 2
    )


def _teacher_risk(
    region: EvidenceRegion,
    test: TestEvidence,
    expected_space: bool,
) -> tuple[SafetyReason, ...]:
    ink = region.features.ink
    geometry = region.features.geometry
    relative_area = (
        region.bbox.width
        * region.bbox.height
        / (test.region_bbox.width * test.region_bbox.height)
    )
    reasons: list[SafetyReason] = []
    chromatic_isolate = (
        ink.mean_saturation >= 0.25
        and geometry.isolation >= 0.65
        and geometry.word_count_in_line <= 1
    )
    if ink.red_foreground_ratio >= 0.15 or chromatic_isolate:
        reasons.append(SafetyReason.TEACHER_CHROMATIC_ISOLATE)
    margin_or_score = (
        geometry.margin_position >= 1.0
        and geometry.isolation >= 0.65
        and relative_area <= 0.02
    )
    if margin_or_score:
        reasons.append(SafetyReason.TEACHER_MARGIN_OR_SCORE_RISK)
    overlaps_other = any(
        other is not region
        and other.evidence_type is EvidenceType.STUDENT_CANDIDATE
        and _intersection_area(region.bbox, other.bbox) > 0
        for other in test.evidence_regions
    )
    if chromatic_isolate and overlaps_other:
        reasons.append(SafetyReason.CORRECTION_STROKE_RISK)
    isolated_outside = (
        not expected_space
        and geometry.isolation >= 0.85
        and geometry.word_count_in_line == 0
        and ink.connected_component_count <= 3
        and relative_area <= 0.003
    )
    if isolated_outside:
        reasons.append(SafetyReason.ISOLATED_MARK_OUTSIDE_ANSWER_SPACE)
    return tuple(dict.fromkeys(reasons))


def _build_answer_regions(
    test: TestEvidence,
    decisions: tuple[CandidateSafetyDecision, ...],
) -> tuple[StudentAnswerRegion, ...]:
    candidates = [
        _RegionCandidate(
            bbox=_expand(item.evidence.bbox, test.region_bbox, padding=8),
            source_indices=(item.evidence_index,),
            confidence=item.evidence.confidence,
        )
        for item in decisions
        if item.disposition is CandidateDisposition.SAFE_STUDENT
    ]
    merged: list[_RegionCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (item.bbox.y, item.bbox.x)):
        target = next(
            (
                index
                for index, existing in enumerate(merged)
                if _same_answer_unit(existing.bbox, candidate.bbox, test.region_bbox)
            ),
            None,
        )
        if target is None:
            merged.append(candidate)
            continue
        existing = merged[target]
        merged[target] = _RegionCandidate(
            bbox=_union(existing.bbox, candidate.bbox),
            source_indices=tuple(
                sorted(set((*existing.source_indices, *candidate.source_indices)))
            ),
            confidence=max(existing.confidence, candidate.confidence),
        )
    return tuple(
        StudentAnswerRegion(
            paper_id=test.paper_id,
            page_number=test.page_number,
            test_number=test.test_number,
            bbox=item.bbox,
            confidence=item.confidence,
            signals=(AnswerRegionSignal.STUDENT_EVIDENCE_CLUSTER,),
            source_evidence_indices=item.source_indices,
            source_image_path=decisions[
                item.source_indices[0]
            ].evidence.source_image_path,
            detection_strategy=SAFE_ANSWER_REGION_VERSION,
        )
        for item in sorted(merged, key=lambda value: (value.bbox.y, value.bbox.x))
    )


def _same_answer_unit(left: BoundingBox, right: BoundingBox, test: BoundingBox) -> bool:
    vertical_overlap = _overlap(
        left.y, left.y + left.height, right.y, right.y + right.height
    )
    smaller_height = min(left.height, right.height)
    horizontal_gap = max(
        0,
        max(left.x, right.x) - min(left.x + left.width, right.x + right.width),
    )
    same_line = vertical_overlap >= smaller_height * 0.45 and horizontal_gap <= max(
        14, smaller_height * 2
    )
    horizontal_overlap = _overlap(
        left.x, left.x + left.width, right.x, right.x + right.width
    )
    smaller_width = min(left.width, right.width)
    vertical_gap = max(
        0,
        max(left.y, right.y) - min(left.y + left.height, right.y + right.height),
    )
    paragraph_lines = (
        horizontal_overlap >= smaller_width * 0.35
        and vertical_gap <= max(left.height, right.height)
        and _union(left, right).width * _union(left, right).height
        <= test.width * test.height * 0.20
    )
    return same_line or paragraph_lines


def _expand(box: BoundingBox, limit: BoundingBox, *, padding: int) -> BoundingBox:
    left = max(limit.x, box.x - padding)
    top = max(limit.y, box.y - padding)
    right = min(limit.x + limit.width, box.x + box.width + padding)
    bottom = min(limit.y + limit.height, box.y + box.height + padding)
    return BoundingBox(x=left, y=top, width=right - left, height=bottom - top)


def _union(left: BoundingBox, right: BoundingBox) -> BoundingBox:
    x = min(left.x, right.x)
    y = min(left.y, right.y)
    right_edge = max(left.x + left.width, right.x + right.width)
    bottom = max(left.y + left.height, right.y + right.height)
    return BoundingBox(x=x, y=y, width=right_edge - x, height=bottom - y)


def _overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> int:
    return max(0, min(left_end, right_end) - max(left_start, right_start))


def _intersection_area(left: BoundingBox, right: BoundingBox) -> int:
    return _overlap(
        left.x, left.x + left.width, right.x, right.x + right.width
    ) * _overlap(left.y, left.y + left.height, right.y, right.y + right.height)

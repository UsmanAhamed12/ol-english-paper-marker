"""Typed outcomes for the minimal post-V1 student-extraction safety boundary."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.evidence.models import EvidenceRegion, StudentAnswerRegion
from app.ocr.models import BoundingBox


class CandidateDisposition(StrEnum):
    """Whether a V1 evidence candidate may enter automatic student OCR."""

    SAFE_STUDENT = "safe_student"
    REVIEW_REQUIRED = "review_required"
    REJECTED_NON_STUDENT = "rejected_non_student"


class SafetyReason(StrEnum):
    """Stable, explainable reasons emitted by the safety filter."""

    UNCERTAIN_CLASSIFICATION = "uncertain_classification"
    TEACHER_EVIDENCE_CLASSIFIED = "teacher_evidence_classified"
    V1_NON_STUDENT = "v1_non_student"
    STUDENT_SIGNALS_PRESERVED = "student_signals_preserved"
    PRINTED_REGULAR_TEXT = "printed_regular_text"
    PRINTED_GUIDE_OR_RULE = "printed_guide_or_rule"
    GRID_OR_PAGE_BORDER = "grid_or_page_border"
    TINY_NOISE = "tiny_noise"
    TEACHER_CHROMATIC_ISOLATE = "teacher_chromatic_isolate"
    TEACHER_MARGIN_OR_SCORE_RISK = "teacher_margin_or_score_risk"
    CORRECTION_STROKE_RISK = "correction_stroke_risk"
    ISOLATED_MARK_OUTSIDE_ANSWER_SPACE = "isolated_mark_outside_answer_space"
    EXPECTED_ANSWER_SPACE = "expected_answer_space"


class TestExtractionStatus(StrEnum):
    """Safe downstream state for one Test region."""

    SAFE = "safe"
    REVIEW_REQUIRED = "review_required"
    EMPTY = "empty"


class CandidateSafetyDecision(BaseModel):
    """One immutable disposition retaining the complete V1 evidence object."""

    model_config = ConfigDict(frozen=True)

    evidence_index: Annotated[int, Field(ge=0)]
    disposition: CandidateDisposition
    reasons: Annotated[tuple[SafetyReason, ...], Field(min_length=1)]
    evidence: EvidenceRegion


class SafeTestExtraction(BaseModel):
    """Filtered V1 evidence and bounded regions for downstream extraction."""

    model_config = ConfigDict(frozen=True)

    paper_id: UUID
    page_number: Annotated[int, Field(gt=0)]
    test_number: Annotated[int, Field(ge=1, le=99)]
    region_bbox: BoundingBox
    status: TestExtractionStatus
    decisions: tuple[CandidateSafetyDecision, ...]
    answer_regions: tuple[StudentAnswerRegion, ...] = ()
    policy_version: Annotated[str, Field(min_length=1)]

    @model_validator(mode="after")
    def provenance_and_status_are_consistent(self) -> Self:
        if any(
            item.evidence.paper_id != self.paper_id
            or item.evidence.page_number != self.page_number
            or item.evidence.test_number != self.test_number
            for item in self.decisions
        ) or any(
            item.paper_id != self.paper_id
            or item.page_number != self.page_number
            or item.test_number != self.test_number
            for item in self.answer_regions
        ):
            raise ValueError("Safety-filter children must match Test provenance")
        if any(
            not _contains(self.region_bbox, item.evidence.bbox)
            for item in self.decisions
        ):
            raise ValueError("Safety-filter evidence must fit its Test region")
        if any(
            not _contains(self.region_bbox, item.bbox) for item in self.answer_regions
        ):
            raise ValueError("Safe answer regions must fit their Test region")
        has_safe = any(
            item.disposition is CandidateDisposition.SAFE_STUDENT
            for item in self.decisions
        )
        has_review = any(
            item.disposition is CandidateDisposition.REVIEW_REQUIRED
            for item in self.decisions
        )
        expected = (
            TestExtractionStatus.REVIEW_REQUIRED
            if has_review
            else TestExtractionStatus.SAFE
            if has_safe
            else TestExtractionStatus.EMPTY
        )
        if self.status is not expected:
            raise ValueError("Safety-filter Test status is inconsistent")
        if bool(self.answer_regions) != has_safe:
            raise ValueError("Safe answer regions must correspond to safe evidence")
        return self


def _contains(container: BoundingBox, child: BoundingBox) -> bool:
    return (
        child.x >= container.x
        and child.y >= container.y
        and child.x + child.width <= container.x + container.width
        and child.y + child.height <= container.y + container.height
    )

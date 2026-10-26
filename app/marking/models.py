"""Immutable, auditable result models for deterministic marking."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.answers.models import AnswerExtractionStatus
from app.marking_scheme.models import CriterionType

NonNegativeMarks = Annotated[Decimal, Field(ge=0)]
NonNegativeDuration = Annotated[float, Field(ge=0)]
PositiveTestNumber = Annotated[int, Field(ge=1, le=99)]


class MarkingStatus(StrEnum):
    """Whether marks are confirmed, unresolved, or structurally unavailable."""

    MARKED = "marked"
    PARTIALLY_MARKED = "partially_marked"
    NOT_ATTEMPTED = "not_attempted"
    REVIEW_REQUIRED = "review_required"
    UNSUPPORTED_CRITERION = "unsupported_criterion"
    OCR_FAILED = "ocr_failed"
    SCHEME_INCOMPLETE = "scheme_incomplete"
    STRUCTURE_INCOMPLETE = "structure_incomplete"


class CriterionDecision(StrEnum):
    """Explainable deterministic outcome for one criterion."""

    EXACT_MATCH = "exact_match"
    NORMALIZED_MATCH = "normalized_match"
    ACCEPTED_VARIANT_MATCH = "accepted_variant_match"
    KEYWORD_MATCH = "keyword_match"
    MULTI_PART_MATCH = "multi_part_match"
    NO_MATCH = "no_match"
    EMPTY_ANSWER = "empty_answer"
    MANUAL_REVIEW = "manual_review"


class MarkingReasonCode(StrEnum):
    """Stable reason codes intended for audit logs and later UI display."""

    EXACT_ANSWER_MATCH = "exact_answer_match"
    NORMALIZED_ANSWER_MATCH = "normalized_answer_match"
    ACCEPTED_VARIANT_MATCH = "accepted_variant_match"
    REQUIRED_KEYWORD_MATCH = "required_keyword_match"
    MISSING_REQUIRED_KEYWORD = "missing_required_keyword"
    MULTI_PART_COMPONENT_MATCH = "multi_part_component_match"
    MULTI_PART_COMPONENT_MISMATCH = "multi_part_component_mismatch"
    NO_DETERMINISTIC_MATCH = "no_deterministic_match"
    EMPTY_STUDENT_ANSWER = "empty_student_answer"
    STUDENT_REVIEW_REQUIRED = "student_review_required"
    STUDENT_OCR_FAILED = "student_ocr_failed"
    UNSUPPORTED_SUBJECTIVE_CRITERION = "unsupported_subjective_criterion"
    UNKNOWN_CRITERION = "unknown_criterion"
    MISSING_MAXIMUM_MARKS = "missing_maximum_marks"
    MISSING_COMPONENT_ALLOCATION = "missing_component_allocation"
    MISSING_STUDENT_TEST = "missing_student_test"
    MISSING_SCHEME_TEST = "missing_scheme_test"
    STUDENT_STRUCTURE_INCOMPLETE = "student_structure_incomplete"


class MarkingWarning(StrEnum):
    """Non-fatal concerns that prevent a fully definitive score."""

    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    SUBJECTIVE_CRITERION_UNSUPPORTED = "subjective_criterion_unsupported"
    MARK_ALLOCATION_UNKNOWN = "mark_allocation_unknown"
    STUDENT_TEST_MISSING = "student_test_missing"
    SCHEME_TEST_MISSING = "scheme_test_missing"
    OCR_EVIDENCE_UNAVAILABLE = "ocr_evidence_unavailable"
    STRUCTURE_EVIDENCE_INCOMPLETE = "structure_evidence_incomplete"
    PAPER_PARTIALLY_MARKED = "paper_partially_marked"


class MarkingProvenance(BaseModel):
    """Versions and source identifiers used for one deterministic decision."""

    model_config = ConfigDict(frozen=True)

    engine_version: Annotated[str, Field(min_length=1)]
    normalization_version: Annotated[str, Field(min_length=1)]
    matcher: Annotated[str, Field(min_length=1)]
    criterion_id: Annotated[str, Field(min_length=1)] | None = None
    criterion_source_page: Annotated[int, Field(gt=0)] | None = None


class StudentEvidenceReference(BaseModel):
    """Original Test-owned student evidence; normalization never replaces it."""

    model_config = ConfigDict(frozen=True)

    paper_id: UUID
    test_number: PositiveTestNumber
    extraction_status: AnswerExtractionStatus
    original_text: str
    pages: tuple[Annotated[int, Field(gt=0)], ...]


class CriterionComponentResult(BaseModel):
    """Explicit partial-credit decision for one allocated component."""

    model_config = ConfigDict(frozen=True)

    component_id: Annotated[str, Field(min_length=1)]
    awarded_marks: NonNegativeMarks
    maximum_marks: NonNegativeMarks
    matched: bool
    matched_value: str | None = None

    @model_validator(mode="after")
    def component_award_is_capped(self) -> Self:
        if self.awarded_marks > self.maximum_marks:
            raise ValueError("Component award cannot exceed its maximum")
        return self


class CriterionMarkResult(BaseModel):
    """One criterion result tied to scheme and student evidence provenance."""

    model_config = ConfigDict(frozen=True)

    criterion_id: Annotated[str, Field(min_length=1)]
    test_number: PositiveTestNumber
    criterion_type: CriterionType
    status: MarkingStatus
    student_evidence: StudentEvidenceReference | None = None
    awarded_marks: NonNegativeMarks | None = None
    maximum_marks: NonNegativeMarks | None = None
    decision: CriterionDecision
    comparison_would_match: bool | None = None
    matched_value: str | None = None
    matched_keywords: tuple[str, ...] = ()
    missing_keywords: tuple[str, ...] = ()
    component_results: tuple[CriterionComponentResult, ...] = ()
    reason_codes: Annotated[tuple[MarkingReasonCode, ...], Field(min_length=1)]
    warnings: tuple[MarkingWarning, ...] = ()
    provenance: MarkingProvenance

    @model_validator(mode="after")
    def criterion_score_is_safe(self) -> Self:
        resolved = {MarkingStatus.MARKED, MarkingStatus.NOT_ATTEMPTED}
        if self.status in resolved and self.awarded_marks is None:
            raise ValueError("Resolved criteria require an awarded mark")
        if self.status not in resolved and self.awarded_marks is not None:
            raise ValueError("Unresolved criteria cannot contain a final award")
        if self.awarded_marks is not None and self.maximum_marks is None:
            raise ValueError("An awarded mark requires a criterion maximum")
        if (
            self.awarded_marks is not None
            and self.maximum_marks is not None
            and self.awarded_marks > self.maximum_marks
        ):
            raise ValueError("Criterion award cannot exceed its maximum")
        component_total = sum(
            (item.awarded_marks for item in self.component_results), Decimal("0")
        )
        if self.component_results and self.awarded_marks != component_total:
            raise ValueError("Criterion award must equal component awards")
        return self


class TestMarkResult(BaseModel):
    """Confirmed and unresolved marks for one exact Test number."""

    model_config = ConfigDict(frozen=True)

    test_number: PositiveTestNumber
    status: MarkingStatus
    criterion_results: tuple[CriterionMarkResult, ...]
    awarded_marks: NonNegativeMarks
    maximum_marks: NonNegativeMarks | None = None
    automatically_marked_marks: NonNegativeMarks
    review_required_marks: NonNegativeMarks
    unsupported_marks: NonNegativeMarks
    warnings: tuple[MarkingWarning, ...] = ()
    processing_duration_ms: NonNegativeDuration
    provenance: MarkingProvenance

    @model_validator(mode="after")
    def test_totals_are_consistent(self) -> Self:
        if any(item.test_number != self.test_number for item in self.criterion_results):
            raise ValueError("Criterion results must belong to their Test")
        confirmed = sum(
            (
                item.awarded_marks
                for item in self.criterion_results
                if item.awarded_marks is not None
            ),
            Decimal("0"),
        )
        if (
            confirmed != self.awarded_marks
            or confirmed != self.automatically_marked_marks
        ):
            raise ValueError("Test confirmed marks must equal criterion awards")
        if self.maximum_marks is not None and self.awarded_marks > self.maximum_marks:
            raise ValueError("Test award cannot exceed its maximum")
        accounted = (
            self.awarded_marks + self.review_required_marks + self.unsupported_marks
        )
        if self.maximum_marks is not None and accounted > self.maximum_marks:
            raise ValueError("Test accounted marks cannot exceed its maximum")
        return self


class PaperMarkingResult(BaseModel):
    """Partially automatable paper result without presenting a false final score."""

    model_config = ConfigDict(frozen=True)

    scheme_id: UUID
    student_paper_id: UUID
    test_results: tuple[TestMarkResult, ...]
    confirmed_marks: NonNegativeMarks
    maximum_known_marks: NonNegativeMarks
    review_required_marks: NonNegativeMarks
    unsupported_marks: NonNegativeMarks
    tests_marked: tuple[PositiveTestNumber, ...] = ()
    tests_review_required: tuple[PositiveTestNumber, ...] = ()
    tests_unsupported: tuple[PositiveTestNumber, ...] = ()
    warnings: tuple[MarkingWarning, ...] = ()
    processing_duration_ms: NonNegativeDuration
    provenance: MarkingProvenance

    @model_validator(mode="after")
    def paper_totals_and_summaries_are_consistent(self) -> Self:
        numbers = tuple(item.test_number for item in self.test_results)
        if numbers != tuple(sorted(set(numbers))):
            raise ValueError("Paper marking Tests must be unique and ordered")
        confirmed = sum(
            (item.awarded_marks for item in self.test_results), Decimal("0")
        )
        if confirmed != self.confirmed_marks:
            raise ValueError("Paper confirmed marks must equal Test awards")
        if self.confirmed_marks > self.maximum_known_marks:
            raise ValueError("Paper confirmed marks cannot exceed known maximum")
        accounted = (
            self.confirmed_marks + self.review_required_marks + self.unsupported_marks
        )
        if accounted > self.maximum_known_marks:
            raise ValueError("Paper accounted marks cannot exceed known maximum")
        expected_marked = tuple(
            item.test_number
            for item in self.test_results
            if item.status in {MarkingStatus.MARKED, MarkingStatus.NOT_ATTEMPTED}
        )
        expected_review = tuple(
            item.test_number
            for item in self.test_results
            if item.review_required_marks > 0
            or item.status
            in {
                MarkingStatus.REVIEW_REQUIRED,
                MarkingStatus.OCR_FAILED,
                MarkingStatus.SCHEME_INCOMPLETE,
                MarkingStatus.STRUCTURE_INCOMPLETE,
            }
        )
        expected_unsupported = tuple(
            item.test_number
            for item in self.test_results
            if item.unsupported_marks > 0
            or item.status is MarkingStatus.UNSUPPORTED_CRITERION
        )
        if (
            self.tests_marked != expected_marked
            or self.tests_review_required != expected_review
            or self.tests_unsupported != expected_unsupported
        ):
            raise ValueError("Paper status summaries must match Test results")
        return self

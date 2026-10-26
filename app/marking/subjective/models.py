"""Typed contracts for guarded rubric-based subjective marking."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.marking.models import CriterionMarkResult, StudentEvidenceReference
from app.marking_scheme.models import CriterionType

NonNegativeMarks = Annotated[Decimal, Field(ge=0)]
NonNegativeDuration = Annotated[float, Field(ge=0)]
UnitInterval = Annotated[float, Field(ge=0, le=1)]
PositiveTestNumber = Annotated[int, Field(ge=1, le=99)]


class SubjectiveMarkingStatus(StrEnum):
    """Trust state of a subjective criterion assessment."""

    MARKED = "marked"
    REVIEW_REQUIRED = "review_required"
    UNSUPPORTED_CRITERION = "unsupported_criterion"
    INVALID_MODEL_RESPONSE = "invalid_model_response"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    OCR_FAILED = "ocr_failed"
    SCHEME_INCOMPLETE = "scheme_incomplete"
    PROVIDER_FAILED = "provider_failed"


class ImprovementAdviceStatus(StrEnum):
    """Trust state for optional syllabus-grounded improvement advice."""

    AVAILABLE = "available"
    NO_IMPROVEMENTS_IDENTIFIED = "no_improvements_identified"
    SYLLABUS_NOT_INDEXED = "syllabus_not_indexed"
    SYLLABUS_RETRIEVAL_FAILED = "syllabus_retrieval_failed"
    INVALID_MODEL_TIPS = "invalid_model_tips"
    REVIEW_REQUIRED = "review_required"
    NOT_GENERATED = "not_generated"


class SubjectiveReasonCode(StrEnum):
    """Stable, privacy-safe explanation codes."""

    RUBRIC_ASSESSMENT_ACCEPTED = "rubric_assessment_accepted"
    EMPTY_STUDENT_ANSWER = "empty_student_answer"
    STUDENT_REVIEW_REQUIRED = "student_review_required"
    STUDENT_OCR_FAILED = "student_ocr_failed"
    LOW_OCR_CONFIDENCE = "low_ocr_confidence"
    LOW_MODEL_CONFIDENCE = "low_model_confidence"
    MISSING_STUDENT_TEST = "missing_student_test"
    MISSING_MAXIMUM_MARKS = "missing_maximum_marks"
    UNKNOWN_CRITERION = "unknown_criterion"
    OBJECTIVE_CRITERION_SKIPPED = "objective_criterion_skipped"
    INVALID_MODEL_RESPONSE = "invalid_model_response"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_TIMEOUT = "provider_timeout"
    OWNERSHIP_MISMATCH = "ownership_mismatch"
    MAXIMUM_MISMATCH = "maximum_mismatch"
    SCORE_OUT_OF_RANGE = "score_out_of_range"


class SubjectiveMarkingWarning(StrEnum):
    """Concerns retained for UI and audit use."""

    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    OCR_EVIDENCE_UNCERTAIN = "ocr_evidence_uncertain"
    MODEL_CONFIDENCE_LOW = "model_confidence_low"
    MODEL_RESPONSE_INVALID = "model_response_invalid"
    PROVIDER_FAILURE = "provider_failure"
    SCHEME_EVIDENCE_INCOMPLETE = "scheme_evidence_incomplete"
    SYLLABUS_EVIDENCE_UNAVAILABLE = "syllabus_evidence_unavailable"
    SYLLABUS_RETRIEVAL_FAILED = "syllabus_retrieval_failed"
    IMPROVEMENT_TIPS_INVALID = "improvement_tips_invalid"


class SyllabusEvidenceReference(BaseModel):
    """Stable source citation for one retrieved syllabus chunk."""

    model_config = ConfigDict(frozen=True)

    source_id: Annotated[str, Field(min_length=1)]
    source_name: Annotated[str, Field(min_length=1)]
    source_sha256: Annotated[str, Field(min_length=1)]
    page_number: Annotated[int, Field(gt=0)]


class SyllabusEvidence(BaseModel):
    """Retrieved syllabus text paired with its traceable source reference."""

    model_config = ConfigDict(frozen=True)

    text: Annotated[str, Field(min_length=1)]
    reference: SyllabusEvidenceReference


class RubricEvidenceReference(BaseModel):
    """The exact scheme criterion provided to the model."""

    model_config = ConfigDict(frozen=True)

    criterion_id: Annotated[str, Field(min_length=1)]
    test_number: PositiveTestNumber
    criterion_type: CriterionType
    description: Annotated[str, Field(min_length=1)]
    maximum_marks: NonNegativeMarks | None = None
    source_page: Annotated[int, Field(gt=0)]


class SubjectiveMarkingProvenance(BaseModel):
    """Provider, prompt, and validation versions used for one decision."""

    model_config = ConfigDict(frozen=True)

    service_version: Annotated[str, Field(min_length=1)]
    validation_version: Annotated[str, Field(min_length=1)]
    prompt_version: Annotated[str, Field(min_length=1)]
    provider: Annotated[str, Field(min_length=1)]
    model_version: Annotated[str, Field(min_length=1)]


class SubjectiveModelAssessment(BaseModel):
    """Strict JSON schema accepted from a local model before domain validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    test_number: PositiveTestNumber
    criterion_id: Annotated[str, Field(min_length=1)]
    criterion_type: CriterionType
    maximum_marks: NonNegativeMarks
    awarded_marks: NonNegativeMarks
    status: SubjectiveMarkingStatus
    matched_requirements: tuple[str, ...] = ()
    missing_requirements: tuple[str, ...] = ()
    reason_codes: tuple[str, ...]
    feedback: Annotated[str, Field(min_length=1, max_length=1000)]
    confidence: UnitInterval

    @model_validator(mode="after")
    def awarded_mark_is_capped(self) -> Self:
        if self.awarded_marks > self.maximum_marks:
            raise ValueError("Model award exceeds the reported maximum")
        if self.status not in {
            SubjectiveMarkingStatus.MARKED,
            SubjectiveMarkingStatus.REVIEW_REQUIRED,
        }:
            raise ValueError("Model status is not permitted for an assessment")
        return self


class SubjectiveCriterionMarkResult(BaseModel):
    """Validated subjective outcome preserving raw evidence and uncertainty."""

    model_config = ConfigDict(frozen=True)

    criterion_id: Annotated[str, Field(min_length=1)]
    test_number: PositiveTestNumber
    criterion_type: CriterionType
    status: SubjectiveMarkingStatus
    student_evidence: StudentEvidenceReference | None = None
    rubric_evidence: RubricEvidenceReference
    awarded_marks: NonNegativeMarks | None = None
    advisory_marks: NonNegativeMarks | None = None
    maximum_marks: NonNegativeMarks | None = None
    matched_requirements: tuple[str, ...] = ()
    missing_requirements: tuple[str, ...] = ()
    reason_codes: Annotated[tuple[SubjectiveReasonCode | str, ...], Field(min_length=1)]
    feedback: str = ""
    improvement_tips: tuple[Annotated[str, Field(min_length=1, max_length=300)], ...] = (
        ()
    )
    syllabus_references: tuple[SyllabusEvidenceReference, ...] = ()
    improvement_advice_status: ImprovementAdviceStatus = (
        ImprovementAdviceStatus.NOT_GENERATED
    )
    confidence: UnitInterval | None = None
    warnings: tuple[SubjectiveMarkingWarning, ...] = ()
    raw_model_response: str | None = None
    processing_duration_ms: NonNegativeDuration
    provenance: SubjectiveMarkingProvenance

    @model_validator(mode="after")
    def final_award_is_safe(self) -> Self:
        if self.status is SubjectiveMarkingStatus.MARKED:
            if self.awarded_marks is None or self.maximum_marks is None:
                raise ValueError("Marked subjective criteria require award and maximum")
        elif self.awarded_marks is not None:
            raise ValueError(
                "Unresolved subjective criteria cannot contain final marks"
            )
        for value in (self.awarded_marks, self.advisory_marks):
            if value is not None and (
                self.maximum_marks is None or value > self.maximum_marks
            ):
                raise ValueError("Subjective marks cannot exceed the criterion maximum")
        return self


class SubjectivePaperMarkingResult(BaseModel):
    """All subjective outcomes, including isolated failures, for one paper."""

    model_config = ConfigDict(frozen=True)

    scheme_id: UUID
    student_paper_id: UUID
    criterion_results: tuple[SubjectiveCriterionMarkResult, ...]
    confirmed_marks: NonNegativeMarks
    review_required_marks: NonNegativeMarks
    unsupported_marks: NonNegativeMarks
    processing_duration_ms: NonNegativeDuration

    @model_validator(mode="after")
    def subjective_totals_are_consistent(self) -> Self:
        confirmed = sum(
            (item.awarded_marks or Decimal("0") for item in self.criterion_results),
            Decimal("0"),
        )
        if confirmed != self.confirmed_marks:
            raise ValueError("Subjective confirmed marks are inconsistent")
        return self


class CombinedTestMarkResult(BaseModel):
    """Objective and subjective criterion results for one Test."""

    model_config = ConfigDict(frozen=True)

    test_number: PositiveTestNumber
    objective_results: tuple[CriterionMarkResult, ...]
    subjective_results: tuple[SubjectiveCriterionMarkResult, ...]
    confirmed_objective_marks: NonNegativeMarks
    confirmed_subjective_marks: NonNegativeMarks
    confirmed_marks: NonNegativeMarks
    unresolved_marks: NonNegativeMarks
    unsupported_marks: NonNegativeMarks
    maximum_known_marks: NonNegativeMarks
    final_score: NonNegativeMarks | None = None

    @model_validator(mode="after")
    def combined_test_totals_are_safe(self) -> Self:
        if self.confirmed_marks != (
            self.confirmed_objective_marks + self.confirmed_subjective_marks
        ):
            raise ValueError("Combined confirmed marks are inconsistent")
        if self.confirmed_marks + self.unresolved_marks + self.unsupported_marks > (
            self.maximum_known_marks
        ):
            raise ValueError("Combined Test marks exceed the known maximum")
        if self.final_score is not None and (
            self.unresolved_marks or self.unsupported_marks
        ):
            raise ValueError("A final score requires every mark to be resolved")
        if self.final_score is not None and self.final_score != self.confirmed_marks:
            raise ValueError("Final Test score must equal confirmed marks")
        return self


class CombinedPaperMarkingResult(BaseModel):
    """Paper-level score with explicit confirmed and unresolved semantics."""

    model_config = ConfigDict(frozen=True)

    scheme_id: UUID
    student_paper_id: UUID
    test_results: tuple[CombinedTestMarkResult, ...]
    confirmed_objective_marks: NonNegativeMarks
    confirmed_subjective_marks: NonNegativeMarks
    confirmed_marks: NonNegativeMarks
    review_required_marks: NonNegativeMarks
    unsupported_marks: NonNegativeMarks
    maximum_known_marks: NonNegativeMarks
    final_score: NonNegativeMarks | None = None
    processing_duration_ms: NonNegativeDuration
    engine_versions: Annotated[tuple[str, ...], Field(min_length=2)]

    @model_validator(mode="after")
    def combined_paper_totals_are_safe(self) -> Self:
        if tuple(item.test_number for item in self.test_results) != tuple(
            sorted({item.test_number for item in self.test_results})
        ):
            raise ValueError("Combined paper Tests must be unique and ordered")
        if self.confirmed_marks != (
            self.confirmed_objective_marks + self.confirmed_subjective_marks
        ):
            raise ValueError("Combined paper confirmed marks are inconsistent")
        if self.confirmed_marks > self.maximum_known_marks:
            raise ValueError("Combined paper marks exceed the known maximum")
        expected_objective = sum(
            (item.confirmed_objective_marks for item in self.test_results),
            Decimal("0"),
        )
        expected_subjective = sum(
            (item.confirmed_subjective_marks for item in self.test_results),
            Decimal("0"),
        )
        expected_review = sum(
            (item.unresolved_marks for item in self.test_results), Decimal("0")
        )
        expected_unsupported = sum(
            (item.unsupported_marks for item in self.test_results), Decimal("0")
        )
        expected_maximum = sum(
            (item.maximum_known_marks for item in self.test_results), Decimal("0")
        )
        if (
            self.confirmed_objective_marks != expected_objective
            or self.confirmed_subjective_marks != expected_subjective
            or self.review_required_marks != expected_review
            or self.unsupported_marks != expected_unsupported
            or self.maximum_known_marks != expected_maximum
        ):
            raise ValueError("Combined paper totals do not match Test results")
        if self.final_score is not None and (
            self.review_required_marks or self.unsupported_marks
        ):
            raise ValueError("A final paper score requires all marks resolved")
        if self.final_score is not None and self.final_score != self.confirmed_marks:
            raise ValueError("Final paper score must equal confirmed marks")
        return self

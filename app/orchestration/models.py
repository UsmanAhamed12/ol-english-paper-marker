"""Application-facing contracts for complete paper marking."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.feedback.models import QuestionFeedback
from app.marking.subjective.models import CombinedTestMarkResult

NonNegativeMarks = Annotated[Decimal, Field(ge=0)]
NonNegativeDuration = Annotated[float, Field(ge=0)]
PositiveTestNumber = Annotated[int, Field(ge=1, le=99)]

ORCHESTRATION_VERSION = "paper-marking-orchestration-v1"


class FinalPaperStatus(StrEnum):
    """Meaningful completion state presented to an application caller."""

    COMPLETED = "completed"
    COMPLETED_WITH_REVIEW = "completed_with_review"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"


class PipelineStage(StrEnum):
    """Major observable stages of the marking workflow."""

    INPUT_VALIDATION = "input_validation"
    STUDENT_PDF_PROCESSING = "student_pdf_processing"
    STRUCTURE_DETECTION = "structure_detection"
    STUDENT_ANSWER_EXTRACTION = "student_answer_extraction"
    MARKING_SCHEME_INGESTION = "marking_scheme_ingestion"
    TEST_ALIGNMENT = "test_alignment"
    OBJECTIVE_MARKING = "objective_marking"
    SUBJECTIVE_MARKING = "subjective_marking"
    FORMATIVE_FEEDBACK = "formative_feedback"
    RESULT_AGGREGATION = "result_aggregation"
    HUMAN_REVIEW = "human_review"


class StageStatus(StrEnum):
    """Safe stage outcome without stack traces or source content."""

    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"


class OrchestrationErrorCode(StrEnum):
    """Stable public errors for unrecoverable pipeline boundaries."""

    INVALID_STUDENT_PDF = "invalid_student_pdf"
    INVALID_MARKING_SCHEME = "invalid_marking_scheme"
    INVALID_OUTPUT_PATH = "invalid_output_path"
    STUDENT_PROCESSING_FAILED = "student_processing_failed"
    SCHEME_INGESTION_FAILED = "scheme_ingestion_failed"
    ALIGNMENT_FAILED = "alignment_failed"
    OBJECTIVE_MARKING_FAILED = "objective_marking_failed"
    SUBJECTIVE_MARKING_PARTIAL = "subjective_marking_partial"
    RESULT_AGGREGATION_FAILED = "result_aggregation_failed"


class PaperMarkingRequest(BaseModel):
    """Identity-safe request accepted by the top-level application service."""

    model_config = ConfigDict(frozen=True)

    student_pdf: Path
    marking_scheme_pdf: Path
    run_id: UUID = Field(default_factory=uuid4)
    output_path: Path | None = None
    pipeline_version: Annotated[str, Field(min_length=1)] = ORCHESTRATION_VERSION

    @model_validator(mode="after")
    def inputs_are_distinct(self) -> Self:
        if self.student_pdf.expanduser().absolute() == (
            self.marking_scheme_pdf.expanduser().absolute()
        ):
            raise ValueError("Student and marking-scheme inputs must be distinct")
        return self


class StageSummary(BaseModel):
    """Timing and safe status for one pipeline stage."""

    model_config = ConfigDict(frozen=True)

    stage: PipelineStage
    status: StageStatus
    duration_ms: NonNegativeDuration
    warnings: tuple[Annotated[str, Field(min_length=1)], ...] = ()
    error_code: OrchestrationErrorCode | None = None

    @model_validator(mode="after")
    def error_matches_status(self) -> Self:
        if (self.status is StageStatus.FAILED) != (self.error_code is not None):
            raise ValueError("Only failed stages may contain an error code")
        return self


class OrchestrationIssue(BaseModel):
    """One safe user-facing pipeline issue."""

    model_config = ConfigDict(frozen=True)

    code: OrchestrationErrorCode
    stage: PipelineStage
    message: Annotated[str, Field(min_length=1)]


class ReviewItem(BaseModel):
    """One unresolved item ready for a later examiner UI."""

    model_config = ConfigDict(frozen=True)

    test_number: PositiveTestNumber
    criterion_id: Annotated[str, Field(min_length=1)] | None = None
    reason: Annotated[str, Field(min_length=1)]
    status: Annotated[str, Field(min_length=1)]
    possible_marks: NonNegativeMarks
    source_reference: Annotated[str, Field(min_length=1)]


class CriterionFeedback(BaseModel):
    """Existing criterion feedback exposed without another model call."""

    model_config = ConfigDict(frozen=True)

    test_number: PositiveTestNumber
    criterion_id: Annotated[str, Field(min_length=1)]
    status: Annotated[str, Field(min_length=1)]
    feedback: Annotated[str, Field(min_length=1)]


class ComponentVersion(BaseModel):
    """One named processing component and its stable version."""

    model_config = ConfigDict(frozen=True)

    component: Annotated[str, Field(min_length=1)]
    version: Annotated[str, Field(min_length=1)]


class OrchestrationProvenance(BaseModel):
    """Complete safe version path for this application result."""

    model_config = ConfigDict(frozen=True)

    orchestration_version: Annotated[str, Field(min_length=1)]
    components: tuple[ComponentVersion, ...]


class FinalPaperMarkingResult(BaseModel):
    """Single application-facing result for success, review, partial, or failure."""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    status: FinalPaperStatus
    student_document_id: UUID | None = None
    marking_scheme_id: UUID | None = None
    detected_tests: tuple[PositiveTestNumber, ...] = ()
    missing_tests: tuple[PositiveTestNumber, ...] = ()
    aligned_tests: tuple[PositiveTestNumber, ...] = ()
    objective_confirmed_marks: NonNegativeMarks = Decimal("0")
    subjective_confirmed_marks: NonNegativeMarks = Decimal("0")
    total_confirmed_marks: NonNegativeMarks = Decimal("0")
    known_maximum_marks: NonNegativeMarks = Decimal("0")
    review_required_marks: NonNegativeMarks = Decimal("0")
    unsupported_marks: NonNegativeMarks = Decimal("0")
    unresolved_marks: NonNegativeMarks = Decimal("0")
    final_score: NonNegativeMarks | None = None
    test_results: tuple[CombinedTestMarkResult, ...] = ()
    feedback: tuple[CriterionFeedback, ...] = ()
    review_items: tuple[ReviewItem, ...] = ()
    question_feedback: tuple[QuestionFeedback, ...] = ()
    stages: tuple[StageSummary, ...]
    warnings: tuple[Annotated[str, Field(min_length=1)], ...] = ()
    errors: tuple[OrchestrationIssue, ...] = ()
    total_duration_ms: NonNegativeDuration
    provenance: OrchestrationProvenance

    @model_validator(mode="after")
    def result_accounting_is_safe(self) -> Self:
        if self.total_confirmed_marks != (
            self.objective_confirmed_marks + self.subjective_confirmed_marks
        ):
            raise ValueError("Final confirmed marks are inconsistent")
        if self.unresolved_marks != (
            self.review_required_marks + self.unsupported_marks
        ):
            raise ValueError("Final unresolved marks are inconsistent")
        if self.total_confirmed_marks > self.known_maximum_marks:
            raise ValueError("Final confirmed marks exceed the known maximum")
        if self.final_score is not None and (
            self.final_score != self.total_confirmed_marks
            or self.unresolved_marks != 0
            or self.status is not FinalPaperStatus.COMPLETED
        ):
            raise ValueError("Final score requires a fully resolved completed paper")
        if self.status is FinalPaperStatus.FAILED:
            if self.test_results or not self.errors or self.final_score is not None:
                raise ValueError("Failed results require errors and no marking output")
        elif self.errors:
            raise ValueError("Completed results cannot contain fatal errors")
        objective_ids = tuple(
            result.criterion_id
            for test in self.test_results
            for result in test.objective_results
        )
        subjective_ids = tuple(
            result.criterion_id
            for test in self.test_results
            for result in test.subjective_results
        )
        ids = objective_ids + subjective_ids
        if len(ids) != len(set(ids)):
            raise ValueError("A criterion cannot be counted by multiple engines")
        feedback_tests = tuple(item.test_number for item in self.question_feedback)
        if len(feedback_tests) != len(set(feedback_tests)):
            raise ValueError("Question feedback must be unique per Test")
        if any(number not in self.detected_tests for number in feedback_tests):
            raise ValueError("Question feedback must belong to a detected Test")
        return self

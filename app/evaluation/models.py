"""Typed contracts for isolated historical marking evaluation."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonNegativeMarks = Annotated[Decimal, Field(ge=0)]
PositiveTestNumber = Annotated[int, Field(ge=1, le=99)]
SafePaperAlias = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")]
NonBlankString = Annotated[str, Field(min_length=1)]
CoveragePercent = Annotated[Decimal, Field(ge=0, le=100)]


class EvaluationOutcomeStatus(StrEnum):
    """Whether an individual question prediction is safe to score."""

    SCORED = "scored"
    REVIEW_REQUIRED = "review_required"
    FAILED = "failed"


class EvaluationFailureStage(StrEnum):
    """Pipeline boundary responsible for an unavailable prediction."""

    OCR = "ocr"
    SEGMENTATION = "segmentation"
    RETRIEVAL = "retrieval"
    GRADING = "grading"
    AGGREGATION = "aggregation"
    OTHER = "other"


class TeacherQuestionReference(BaseModel):
    """One human-verified score, stored only in the evaluation reference set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_alias: SafePaperAlias
    test_number: PositiveTestNumber
    awarded_marks: NonNegativeMarks
    maximum_marks: NonNegativeMarks

    @model_validator(mode="after")
    def teacher_score_is_bounded(self) -> Self:
        if self.awarded_marks > self.maximum_marks:
            raise ValueError("Teacher award cannot exceed the question maximum")
        return self


class HistoricalEvaluationDataset(BaseModel):
    """Versioned human reference marks, separate from runtime grading inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: SafePaperAlias
    dataset_version: NonBlankString
    references: Annotated[tuple[TeacherQuestionReference, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def reference_keys_are_unique(self) -> Self:
        keys = [
            (reference.paper_alias, reference.test_number)
            for reference in self.references
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("Teacher reference question keys must be unique")
        return self


class EvaluationPrediction(BaseModel):
    """Prediction-only outcome; deliberately has no teacher-score field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_alias: SafePaperAlias
    test_number: PositiveTestNumber
    status: EvaluationOutcomeStatus
    predicted_marks: NonNegativeMarks | None = None
    failure_stage: EvaluationFailureStage | None = None

    @model_validator(mode="after")
    def score_matches_outcome(self) -> Self:
        if (self.status is EvaluationOutcomeStatus.SCORED) != (
            self.predicted_marks is not None
        ):
            raise ValueError("Only scored outcomes may contain predicted marks")
        if (
            self.status is EvaluationOutcomeStatus.SCORED
            and self.failure_stage is not None
        ):
            raise ValueError("Scored outcomes cannot have a failure stage")
        return self


class EvaluationPredictionRun(BaseModel):
    """Frozen predictions and run provenance, with no reference-score access."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: UUID
    model_name: NonBlankString
    model_version: NonBlankString
    prompt_version: NonBlankString
    pipeline_version: NonBlankString
    predictions: tuple[EvaluationPrediction, ...] = ()

    @model_validator(mode="after")
    def prediction_keys_are_unique(self) -> Self:
        keys = [
            (prediction.paper_alias, prediction.test_number)
            for prediction in self.predictions
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("Prediction question keys must be unique")
        return self


class MarkMetrics(BaseModel):
    """Paper or question score metrics over only fully scored pairs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_count: Annotated[int, Field(ge=0)]
    mae: NonNegativeMarks | None
    rmse: NonNegativeMarks | None
    mean_signed_error: Decimal | None
    within_1_mark_percent: CoveragePercent | None
    within_2_marks_percent: CoveragePercent | None
    within_5_marks_percent: CoveragePercent | None

    @model_validator(mode="after")
    def metrics_match_sample_count(self) -> Self:
        values = (
            self.mae,
            self.rmse,
            self.mean_signed_error,
            self.within_1_mark_percent,
            self.within_2_marks_percent,
            self.within_5_marks_percent,
        )
        if (self.sample_count == 0) != all(value is None for value in values):
            raise ValueError("Metric values must be present exactly when scored")
        if self.sample_count and any(value is None for value in values):
            raise ValueError("All mark metrics are required when samples are scored")
        return self


class EvaluationCoverage(BaseModel):
    """Expected versus fully scored item counts and percentage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_count: Annotated[int, Field(ge=0)]
    scored_count: Annotated[int, Field(ge=0)]
    coverage_percent: CoveragePercent

    @model_validator(mode="after")
    def coverage_is_consistent(self) -> Self:
        if self.scored_count > self.expected_count:
            raise ValueError("Scored count cannot exceed expected count")
        expected = (
            Decimal("0")
            if self.expected_count == 0
            else Decimal(self.scored_count * 100) / Decimal(self.expected_count)
        )
        if self.coverage_percent != expected:
            raise ValueError("Coverage percentage does not match the counts")
        return self


class QuestionEvaluationSummary(BaseModel):
    """Coverage and metrics for one exam question across papers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    test_number: PositiveTestNumber
    coverage: EvaluationCoverage
    metrics: MarkMetrics


class FailureStageCount(BaseModel):
    """Count of unavailable question outcomes attributed to a stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: EvaluationFailureStage
    count: Annotated[int, Field(gt=0)]


class HistoricalEvaluationReport(BaseModel):
    """Reproducible summary of a prediction run against held-out references."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: SafePaperAlias
    dataset_version: NonBlankString
    dataset_fingerprint: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    run_id: UUID
    model_name: NonBlankString
    model_version: NonBlankString
    prompt_version: NonBlankString
    pipeline_version: NonBlankString
    question_coverage: EvaluationCoverage
    question_metrics: MarkMetrics
    per_question: tuple[QuestionEvaluationSummary, ...]
    paper_coverage: EvaluationCoverage
    paper_total_metrics: MarkMetrics
    missing_prediction_count: Annotated[int, Field(ge=0)]
    unscored_prediction_count: Annotated[int, Field(ge=0)]
    failure_stage_counts: tuple[FailureStageCount, ...]

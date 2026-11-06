"""Framework-independent view models consumed by Streamlit components."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class UIRunState(StrEnum):
    """Rerun-safe UI workflow state."""

    READY = "ready"
    VALIDATING = "validating"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ViewTone(StrEnum):
    """Accessible semantic tone; text labels remain authoritative."""

    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    NEUTRAL = "neutral"


class MetricView(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    value: str


class WarningView(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str


class StageView(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    status: str
    duration: str


class CriterionView(BaseModel):
    model_config = ConfigDict(frozen=True)

    criterion_id: str
    criterion_type: str
    status: str
    marks: str
    decision: str
    reason: str
    feedback: str | None = None
    improvement_tips: tuple[str, ...] = ()
    syllabus_references: tuple[str, ...] = ()
    improvement_advice_status: str | None = None


class TestResultView(BaseModel):
    model_config = ConfigDict(frozen=True)

    test_number: int
    heading: str
    status: str
    confirmed_marks: str
    maximum_marks: str
    unresolved_marks: str
    objective_marks: str
    subjective_marks: str
    criteria: tuple[CriterionView, ...]
    feedback: tuple[str, ...]
    formative_feedback: str | None
    improvement_tips: tuple[str, ...]
    formative_feedback_status: str | None
    warnings: tuple[WarningView, ...]


class ReviewItemView(BaseModel):
    model_config = ConfigDict(frozen=True)

    test_label: str
    criterion_label: str | None
    reason: str
    status: str
    marks_affected: str


class ResultViewModel(BaseModel):
    """Complete safe UI projection of a final backend result."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    status: str
    tone: ViewTone
    score_heading: str
    score_value: str
    raw_score_value: str | None
    percentage: str | None
    unresolved_value: str | None
    is_final_score: bool
    is_demo: bool
    metrics: tuple[MetricView, ...]
    tests: tuple[TestResultView, ...]
    feedback_by_test: tuple[tuple[str, tuple[str, ...]], ...]
    review_items: tuple[ReviewItemView, ...]
    warnings: tuple[WarningView, ...]
    errors: tuple[WarningView, ...]
    stages: tuple[StageView, ...]
    download_json: str

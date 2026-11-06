"""Validated outcomes for local question-level formative feedback."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.answers.models import AnswerExtractionStatus

PositiveTestNumber = Annotated[int, Field(ge=1, le=99)]
NonNegativeDuration = Annotated[float, Field(ge=0)]
FeedbackText = Annotated[str, Field(min_length=1, max_length=800)]
ImprovementTip = Annotated[str, Field(min_length=1, max_length=300)]


class QuestionFeedbackStatus(StrEnum):
    """Whether local feedback was generated and how its evidence should be used."""

    GENERATED = "generated"
    PROVISIONAL = "provisional"
    SKIPPED = "skipped"
    FAILED = "failed"


class QuestionFeedback(BaseModel):
    """Question feedback without any mark-bearing fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    test_number: PositiveTestNumber
    status: QuestionFeedbackStatus
    feedback: FeedbackText | None = None
    improvement_tips: Annotated[tuple[ImprovementTip, ...], Field(max_length=3)] = ()
    answer_status: AnswerExtractionStatus | None = None
    reason_code: Annotated[str, Field(min_length=1)] | None = None
    provider: Annotated[str, Field(min_length=1)] | None = None
    model_version: Annotated[str, Field(min_length=1)] | None = None
    prompt_version: Annotated[str, Field(min_length=1)] | None = None
    processing_duration_ms: NonNegativeDuration = 0

    @model_validator(mode="after")
    def outcome_is_consistent(self) -> Self:
        generated = self.status in {
            QuestionFeedbackStatus.GENERATED,
            QuestionFeedbackStatus.PROVISIONAL,
        }
        if generated != (self.feedback is not None):
            raise ValueError("Generated feedback outcomes require feedback text")
        if generated != (
            self.provider is not None
            and self.model_version is not None
            and self.prompt_version is not None
        ):
            raise ValueError("Generated feedback requires provider provenance")
        if generated and self.reason_code is not None:
            raise ValueError("Generated feedback cannot contain a failure reason")
        if not generated and self.improvement_tips:
            raise ValueError("Skipped or failed feedback cannot contain tips")
        if not generated and self.reason_code is None:
            raise ValueError("Skipped or failed feedback requires a reason")
        if (
            self.status is QuestionFeedbackStatus.PROVISIONAL
            and self.answer_status is AnswerExtractionStatus.EXTRACTED
        ):
            raise ValueError("Provisional feedback requires uncertain extraction")
        if (
            self.status is QuestionFeedbackStatus.GENERATED
            and self.answer_status is not AnswerExtractionStatus.EXTRACTED
        ):
            raise ValueError("Generated feedback requires extracted answer evidence")
        return self

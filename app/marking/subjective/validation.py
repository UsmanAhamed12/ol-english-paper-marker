"""Strict model-response validation before any mark becomes final."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.marking.subjective.models import (
    ImprovementAdviceStatus,
    SubjectiveModelAssessment,
    SubjectiveReasonCode,
)
from app.marking_scheme.models import MarkingCriterion

SUBJECTIVE_VALIDATION_VERSION = "subjective-response-validation-v2"


class _ImprovementTips(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tips: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=300)], ...],
        Field(max_length=3),
    ]


@dataclass(frozen=True)
class ValidatedSubjectiveAssessment:
    """Mark decision and separately validated optional advice."""

    assessment: SubjectiveModelAssessment
    improvement_tips: tuple[str, ...]
    advice_status: ImprovementAdviceStatus


@dataclass(frozen=True)
class SubjectiveValidationFailure:
    """Safe failure metadata that never includes private model content."""

    reason: SubjectiveReasonCode


def validate_subjective_response(
    raw_text: str,
    criterion: MarkingCriterion,
) -> ValidatedSubjectiveAssessment | SubjectiveValidationFailure:
    """Parse strict JSON and enforce scheme ownership and score constraints."""

    try:
        payload = json.loads(raw_text)
        if not isinstance(payload, dict):
            return SubjectiveValidationFailure(
                SubjectiveReasonCode.INVALID_MODEL_RESPONSE
            )
        raw_tips = payload.pop("improvement_tips", None)
        assessment = SubjectiveModelAssessment.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, TypeError, AttributeError):
        return SubjectiveValidationFailure(SubjectiveReasonCode.INVALID_MODEL_RESPONSE)
    if (
        assessment.test_number != criterion.test_number
        or assessment.criterion_id != criterion.criterion_id
        or assessment.criterion_type is not criterion.criterion_type
    ):
        return SubjectiveValidationFailure(SubjectiveReasonCode.OWNERSHIP_MISMATCH)
    if criterion.maximum_marks is None:
        return SubjectiveValidationFailure(SubjectiveReasonCode.MISSING_MAXIMUM_MARKS)
    if assessment.maximum_marks != criterion.maximum_marks:
        return SubjectiveValidationFailure(SubjectiveReasonCode.MAXIMUM_MISMATCH)
    if assessment.awarded_marks > criterion.maximum_marks:
        return SubjectiveValidationFailure(SubjectiveReasonCode.SCORE_OUT_OF_RANGE)
    try:
        tips = _ImprovementTips(tips=raw_tips)
    except ValidationError:
        return ValidatedSubjectiveAssessment(
            assessment=assessment,
            improvement_tips=(),
            advice_status=ImprovementAdviceStatus.INVALID_MODEL_TIPS,
        )
    return ValidatedSubjectiveAssessment(
        assessment=assessment,
        improvement_tips=tips.tips,
        advice_status=(
            ImprovementAdviceStatus.AVAILABLE
            if tips.tips
            else ImprovementAdviceStatus.NO_IMPROVEMENTS_IDENTIFIED
        ),
    )

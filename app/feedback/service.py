"""Local LLM formative feedback that cannot change question marks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.answers.models import AnswerExtractionStatus
from app.core.exceptions import (
    SubjectiveMarkingProviderError,
    SubjectiveMarkingTimeoutError,
)
from app.feedback.models import QuestionFeedback, QuestionFeedbackStatus
from app.marking.subjective.provider import SubjectiveMarkingProvider
from app.marking_scheme.models import (
    AlignedPaper,
    AlignedTest,
    CriterionType,
    MarkingCriterion,
)

QUESTION_FEEDBACK_PROMPT_VERSION = "question-feedback-v2"
_PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "grading"
    / "question-feedback-v2.txt"
)
_MAX_ANSWER_CHARACTERS = 2500
_MAX_CRITERIA_PER_TEST = 20


class _QuestionAdvice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    test_number: Annotated[int, Field(ge=1, le=99)]
    feedback: Annotated[str, Field(min_length=1, max_length=800)]
    improvement_tips: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=300)], ...],
        Field(max_length=3),
    ]


class _AdviceBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    questions: Annotated[tuple[_QuestionAdvice, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def question_numbers_are_unique(self) -> Self:
        numbers = tuple(item.test_number for item in self.questions)
        if len(numbers) != len(set(numbers)):
            raise ValueError("Feedback question numbers must be unique")
        return self


class PaperFeedbackService:
    """Generate evidence-bound advice in one model call, never a mark."""

    def __init__(
        self,
        provider: SubjectiveMarkingProvider,
        *,
        ocr_confidence_threshold: float = 0.80,
    ) -> None:
        if not 0 <= ocr_confidence_threshold <= 1:
            raise ValueError("OCR confidence threshold must be between zero and one")
        self._provider = provider
        self._ocr_threshold = ocr_confidence_threshold

    def generate(self, aligned: AlignedPaper) -> tuple[QuestionFeedback, ...]:
        """Return generated, provisional, or explicitly skipped results per Test."""

        payload: list[dict[str, object]] = []
        targets: dict[int, AlignedTest] = {}
        output: dict[int, QuestionFeedback] = {}

        for item in aligned.tests:
            answer = item.student_answer
            scheme = item.marking_scheme_test
            if answer is None:
                output[item.test_number] = _skipped(
                    item.test_number, "student_answer_missing"
                )
                continue
            if (
                answer.status
                in {
                    AnswerExtractionStatus.EMPTY,
                    AnswerExtractionStatus.OCR_FAILED,
                    AnswerExtractionStatus.STRUCTURE_INCOMPLETE,
                }
                or not answer.text.strip()
            ):
                output[item.test_number] = _skipped(
                    item.test_number,
                    "student_answer_unavailable",
                    answer_status=answer.status,
                )
                continue
            if scheme is None:
                output[item.test_number] = _skipped(
                    item.test_number,
                    "marking_scheme_test_missing",
                    answer_status=answer.status,
                )
                continue
            criteria = tuple(
                criterion
                for criterion in scheme.criteria
                if criterion.criterion_type is not CriterionType.UNKNOWN
            )
            if not criteria:
                output[item.test_number] = _skipped(
                    item.test_number,
                    "no_recognized_scheme_criteria",
                    answer_status=answer.status,
                )
                continue
            targets[item.test_number] = item
            payload.append(_test_payload(answer, criteria))

        if payload:
            prompt = _PROMPT_PATH.read_text(encoding="utf-8").replace(
                "{{QUESTION_DATA_JSON}}",
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            )
            try:
                response = self._provider.assess(prompt)
            except SubjectiveMarkingTimeoutError:
                output.update(_failed(targets, "local_ollama_timeout"))
            except SubjectiveMarkingProviderError:
                output.update(_failed(targets, "local_ollama_unavailable"))
            else:
                output.update(
                    self._parse_response(
                        response.raw_text,
                        targets,
                        provider=response.provider,
                        model_version=response.model_version,
                        duration_ms=response.processing_duration_ms,
                    )
                )
        return tuple(output[number] for number in sorted(output))

    def _parse_response(
        self,
        raw_text: str,
        targets: dict[int, AlignedTest],
        *,
        provider: str,
        model_version: str,
        duration_ms: float,
    ) -> dict[int, QuestionFeedback]:
        try:
            batch = _AdviceBatch.model_validate_json(raw_text)
        except (ValidationError, ValueError):
            return _failed(targets, "invalid_local_feedback_response")
        expected = set(targets)
        actual = {item.test_number for item in batch.questions}
        if actual != expected:
            return _failed(targets, "feedback_question_coverage_mismatch")
        advice_by_number = {item.test_number: item for item in batch.questions}
        results: dict[int, QuestionFeedback] = {}
        for test_number, aligned_test in targets.items():
            answer = aligned_test.student_answer
            if answer is None:
                raise ValueError("Feedback target lost its student answer")
            advice = advice_by_number[test_number]
            provisional = answer.status is AnswerExtractionStatus.REVIEW_REQUIRED or (
                answer.ocr_confidence is not None
                and answer.ocr_confidence < self._ocr_threshold
            )
            results[test_number] = QuestionFeedback(
                test_number=test_number,
                status=(
                    QuestionFeedbackStatus.PROVISIONAL
                    if provisional
                    else QuestionFeedbackStatus.GENERATED
                ),
                feedback=advice.feedback,
                improvement_tips=advice.improvement_tips,
                answer_status=answer.status,
                provider=provider,
                model_version=model_version,
                prompt_version=QUESTION_FEEDBACK_PROMPT_VERSION,
                processing_duration_ms=duration_ms,
            )
        return results


def _test_payload(
    answer: object,
    criteria: tuple[MarkingCriterion, ...],
) -> dict[str, object]:
    from app.answers.models import StructuredStudentAnswer

    if not isinstance(answer, StructuredStudentAnswer):
        raise TypeError("Expected a structured student answer")
    answer_excerpt = answer.text[:_MAX_ANSWER_CHARACTERS]
    rubric = [
        {
            "criterion_id": criterion.criterion_id,
            "criterion_type": criterion.criterion_type.value,
            "description": criterion.description,
            "accepted_answers": criterion.accepted_answers,
            "keywords": criterion.keywords,
            "components": [
                {
                    "accepted_answers": component.accepted_answers,
                    "keywords": component.keywords,
                }
                for component in criterion.components
            ],
        }
        for criterion in criteria[:_MAX_CRITERIA_PER_TEST]
    ]
    return {
        "test_number": answer.test_number,
        "answer_status": answer.status.value,
        "ocr_confidence": answer.ocr_confidence,
        "answer_excerpt": answer_excerpt,
        "answer_truncated": len(answer_excerpt) < len(answer.text),
        "recognized_scheme_criteria": rubric,
        "criteria_truncated": len(criteria) > len(rubric),
    }


def _skipped(
    test_number: int,
    reason_code: str,
    *,
    answer_status: AnswerExtractionStatus | None = None,
) -> QuestionFeedback:
    return QuestionFeedback(
        test_number=test_number,
        status=QuestionFeedbackStatus.SKIPPED,
        answer_status=answer_status,
        reason_code=reason_code,
    )


def _failed(
    targets: dict[int, AlignedTest], reason_code: str
) -> dict[int, QuestionFeedback]:
    return {
        number: QuestionFeedback(
            test_number=number,
            status=QuestionFeedbackStatus.FAILED,
            answer_status=(
                target.student_answer.status
                if target.student_answer is not None
                else None
            ),
            reason_code=reason_code,
        )
        for number, target in targets.items()
    }

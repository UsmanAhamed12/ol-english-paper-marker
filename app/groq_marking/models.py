"""Validated contracts for Groq OCR and whole-paper marking."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonNegativeMarks = Annotated[Decimal, Field(ge=0)]
QuestionNumber = Annotated[str, Field(min_length=1, max_length=40)]


class VisionPageText(BaseModel):
    """Printed and handwritten transcription for one rendered PDF page."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    page_number: Annotated[int, Field(ge=1)]
    printed_text: str
    handwritten_text: str
    understanding: str


class GroqQuestionGrade(BaseModel):
    """One question's answer, rubric-grounded score, and learning feedback."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_number: QuestionNumber
    question_summary: str
    student_answer: str
    awarded_marks: NonNegativeMarks
    maximum_marks: NonNegativeMarks
    explanation: Annotated[str, Field(min_length=1)]
    feedback: str
    improvement_tips: tuple[str, ...]
    review_required: bool = False

    @model_validator(mode="after")
    def score_is_within_question_maximum(self) -> Self:
        if self.awarded_marks > self.maximum_marks:
            raise ValueError("Question marks cannot exceed the rubric maximum")
        if self.maximum_marks == 0 and (
            self.awarded_marks != 0 or not self.review_required
        ):
            raise ValueError(
                "A question without a rubric maximum must be flagged for review"
            )
        return self


class GroqAssessment(BaseModel):
    """Whole-paper LLM output before deterministic total calculation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subject: str
    overall_feedback: str
    improvement_tips: tuple[str, ...]
    questions: Annotated[tuple[GroqQuestionGrade, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def question_numbers_are_unique(self) -> Self:
        question_numbers = tuple(
            question.question_number.casefold() for question in self.questions
        )
        if len(question_numbers) != len(set(question_numbers)):
            raise ValueError("Question numbers must be unique")
        return self


class GroqPaperMarkingResult(BaseModel):
    """Session result with a Python-calculated and validated normalized score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: UUID
    model: Annotated[str, Field(min_length=1)]
    student_pages: tuple[VisionPageText, ...]
    marking_scheme_pages: tuple[VisionPageText, ...]
    assessment: GroqAssessment
    total_awarded_marks: NonNegativeMarks
    total_maximum_marks: NonNegativeMarks
    score_out_of_100: NonNegativeMarks | None
    requires_human_review: bool

    @classmethod
    def from_assessment(
        cls,
        *,
        run_id: UUID,
        model: str,
        student_pages: tuple[VisionPageText, ...],
        marking_scheme_pages: tuple[VisionPageText, ...],
        assessment: GroqAssessment,
    ) -> GroqPaperMarkingResult:
        awarded = sum(
            (question.awarded_marks for question in assessment.questions),
            Decimal("0"),
        )
        maximum = sum(
            (question.maximum_marks for question in assessment.questions),
            Decimal("0"),
        )
        complete_rubric = all(
            question.maximum_marks > 0 for question in assessment.questions
        )
        score = (
            (awarded / maximum * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            if maximum > 0 and complete_rubric
            else None
        )
        review_required = any(
            question.review_required for question in assessment.questions
        )
        return cls(
            run_id=run_id,
            model=model,
            student_pages=student_pages,
            marking_scheme_pages=marking_scheme_pages,
            assessment=assessment,
            total_awarded_marks=awarded,
            total_maximum_marks=maximum,
            score_out_of_100=score,
            requires_human_review=review_required or score is None,
        )

    @model_validator(mode="after")
    def total_and_score_are_consistent(self) -> Self:
        awarded = sum(
            (question.awarded_marks for question in self.assessment.questions),
            Decimal("0"),
        )
        maximum = sum(
            (question.maximum_marks for question in self.assessment.questions),
            Decimal("0"),
        )
        if awarded != self.total_awarded_marks or maximum != self.total_maximum_marks:
            raise ValueError("Paper totals must equal the question-level marks")
        complete_rubric = all(
            question.maximum_marks > 0 for question in self.assessment.questions
        )
        expected_score = (
            (awarded / maximum * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            if maximum > 0 and complete_rubric
            else None
        )
        if self.score_out_of_100 != expected_score:
            raise ValueError("Normalized score must be calculated from validated marks")
        if self.requires_human_review != (
            any(question.review_required for question in self.assessment.questions)
            or expected_score is None
        ):
            raise ValueError("Review status must reflect question and rubric flags")
        return self

"""Validated teacher decisions and safe aggregate updates for graph review."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.marking.models import CriterionDecision, CriterionMarkResult, MarkingStatus
from app.marking.subjective.models import (
    CombinedPaperMarkingResult,
    CombinedTestMarkResult,
    SubjectiveCriterionMarkResult,
    SubjectiveMarkingStatus,
)
from app.marking_scheme.models import AlignedPaper

ReviewMark = Annotated[Decimal, Field(ge=0)]
PositiveTestNumber = Annotated[int, Field(ge=1, le=99)]
ReviewableCriterionType = Literal["objective", "subjective"]


class HumanReviewDecision(BaseModel):
    """Teacher-assigned mark for one unresolved criterion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    test_number: PositiveTestNumber
    criterion_id: Annotated[str, Field(min_length=1)]
    awarded_marks: ReviewMark
    rationale: Annotated[str, Field(min_length=1, max_length=1000)]


class HumanReviewResponse(BaseModel):
    """Auditable resume value returned by the human reviewer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reviewer_id: Annotated[str, Field(min_length=1, max_length=128)]
    decisions: tuple[HumanReviewDecision, ...] = ()

    @model_validator(mode="after")
    def decisions_are_unique(self) -> Self:
        keys = tuple((item.test_number, item.criterion_id) for item in self.decisions)
        if len(keys) != len(set(keys)):
            raise ValueError("A criterion may be reviewed only once per response")
        return self


def build_human_review_payload(
    combined: CombinedPaperMarkingResult,
    aligned: AlignedPaper | None,
    run_id: str,
) -> dict[str, object]:
    """Create JSON-safe review context without exposing internal OCR metadata."""

    aligned_tests = (
        {item.test_number: item for item in aligned.tests}
        if aligned is not None
        else {}
    )
    items: list[dict[str, object]] = []
    for test in combined.test_results:
        aligned_test = aligned_tests.get(test.test_number)
        answer_text = (
            aligned_test.student_answer.text
            if aligned_test is not None and aligned_test.student_answer is not None
            else ""
        )
        rubric_criteria = (
            {
                item.criterion_id: item
                for item in aligned_test.marking_scheme_test.criteria
            }
            if aligned_test is not None and aligned_test.marking_scheme_test is not None
            else {}
        )
        for objective_result in test.objective_results:
            if objective_result.status in {
                MarkingStatus.MARKED,
                MarkingStatus.NOT_ATTEMPTED,
            }:
                continue
            criterion = rubric_criteria.get(objective_result.criterion_id)
            items.append(
                {
                    "test_number": test.test_number,
                    "criterion_id": objective_result.criterion_id,
                    "criterion_type": "objective",
                    "status": objective_result.status.value,
                    "student_answer": answer_text,
                    "rubric": criterion.description if criterion else "",
                    "maximum_marks": (
                        str(objective_result.maximum_marks)
                        if objective_result.maximum_marks is not None
                        else None
                    ),
                    "can_assign_mark": objective_result.maximum_marks is not None,
                }
            )
        for subjective_result in test.subjective_results:
            if subjective_result.status is SubjectiveMarkingStatus.MARKED:
                continue
            items.append(
                {
                    "test_number": test.test_number,
                    "criterion_id": subjective_result.criterion_id,
                    "criterion_type": "subjective",
                    "status": subjective_result.status.value,
                    "student_answer": (
                        subjective_result.student_evidence.original_text
                        if subjective_result.student_evidence is not None
                        else answer_text
                    ),
                    "rubric": subjective_result.rubric_evidence.description,
                    "maximum_marks": (
                        str(subjective_result.maximum_marks)
                        if subjective_result.maximum_marks is not None
                        else None
                    ),
                    "suggested_marks": (
                        str(subjective_result.advisory_marks)
                        if subjective_result.advisory_marks is not None
                        else None
                    ),
                    "confidence": subjective_result.confidence,
                    "can_assign_mark": subjective_result.maximum_marks is not None,
                }
            )
    return {
        "kind": "paper_marking_review",
        "run_id": run_id,
        "items": items,
    }


def apply_human_review(
    combined: CombinedPaperMarkingResult,
    response: HumanReviewResponse,
) -> CombinedPaperMarkingResult:
    """Apply only bounded decisions for unresolved criteria; preserve others."""

    decisions = {
        (item.test_number, item.criterion_id): item for item in response.decisions
    }
    available: dict[tuple[int, str], Decimal] = {}
    for test in combined.test_results:
        for objective_result in test.objective_results:
            if (
                objective_result.status
                not in {MarkingStatus.MARKED, MarkingStatus.NOT_ATTEMPTED}
                and objective_result.maximum_marks is not None
            ):
                available[(test.test_number, objective_result.criterion_id)] = (
                    objective_result.maximum_marks
                )
        for subjective_result in test.subjective_results:
            if (
                subjective_result.status is not SubjectiveMarkingStatus.MARKED
                and subjective_result.maximum_marks is not None
            ):
                available[(test.test_number, subjective_result.criterion_id)] = (
                    subjective_result.maximum_marks
                )

    unknown_decisions = decisions.keys() - available.keys()
    if unknown_decisions:
        raise ValueError("Human review contains an unknown or non-reviewable criterion")
    for key, decision in decisions.items():
        if decision.awarded_marks > available[key]:
            raise ValueError("Human-awarded marks exceed the criterion maximum")

    reviewed_tests: list[CombinedTestMarkResult] = []
    for test in combined.test_results:
        objective_results = tuple(
            _review_objective(item, decisions[(test.test_number, item.criterion_id)])
            if (test.test_number, item.criterion_id) in decisions
            else item
            for item in test.objective_results
        )
        subjective_results = tuple(
            _review_subjective(item, decisions[(test.test_number, item.criterion_id)])
            if (test.test_number, item.criterion_id) in decisions
            else item
            for item in test.subjective_results
        )
        objective_marks = sum(
            (item.awarded_marks or Decimal("0") for item in objective_results),
            Decimal("0"),
        )
        subjective_marks = sum(
            (item.awarded_marks or Decimal("0") for item in subjective_results),
            Decimal("0"),
        )
        review_marks = sum(
            (
                item.maximum_marks or Decimal("0")
                for item in objective_results
                if item.status
                not in {
                    MarkingStatus.MARKED,
                    MarkingStatus.NOT_ATTEMPTED,
                    MarkingStatus.UNSUPPORTED_CRITERION,
                }
            ),
            Decimal("0"),
        ) + sum(
            (
                item.maximum_marks or Decimal("0")
                for item in subjective_results
                if item.status
                not in {
                    SubjectiveMarkingStatus.MARKED,
                    SubjectiveMarkingStatus.UNSUPPORTED_CRITERION,
                }
            ),
            Decimal("0"),
        )
        unsupported_marks = sum(
            (
                item.maximum_marks or Decimal("0")
                for item in objective_results
                if item.status is MarkingStatus.UNSUPPORTED_CRITERION
            ),
            Decimal("0"),
        ) + sum(
            (
                item.maximum_marks or Decimal("0")
                for item in subjective_results
                if item.status is SubjectiveMarkingStatus.UNSUPPORTED_CRITERION
            ),
            Decimal("0"),
        )
        confirmed_marks = objective_marks + subjective_marks
        all_resolved = all(
            item.status in {MarkingStatus.MARKED, MarkingStatus.NOT_ATTEMPTED}
            for item in objective_results
        ) and all(
            item.status is SubjectiveMarkingStatus.MARKED for item in subjective_results
        )
        reviewed_tests.append(
            CombinedTestMarkResult(
                test_number=test.test_number,
                objective_results=objective_results,
                subjective_results=subjective_results,
                confirmed_objective_marks=objective_marks,
                confirmed_subjective_marks=subjective_marks,
                confirmed_marks=confirmed_marks,
                unresolved_marks=review_marks,
                unsupported_marks=unsupported_marks,
                maximum_known_marks=test.maximum_known_marks,
                final_score=(
                    confirmed_marks
                    if review_marks == 0 and unsupported_marks == 0 and all_resolved
                    else None
                ),
            )
        )

    confirmed_objective = sum(
        (item.confirmed_objective_marks for item in reviewed_tests), Decimal("0")
    )
    confirmed_subjective = sum(
        (item.confirmed_subjective_marks for item in reviewed_tests), Decimal("0")
    )
    confirmed = confirmed_objective + confirmed_subjective
    review_required = sum(
        (item.unresolved_marks for item in reviewed_tests), Decimal("0")
    )
    unsupported = sum((item.unsupported_marks for item in reviewed_tests), Decimal("0"))
    fully_resolved = all(item.final_score is not None for item in reviewed_tests)
    engine_versions = combined.engine_versions
    if response.decisions and "human-review-v1" not in engine_versions:
        engine_versions += ("human-review-v1",)
    return CombinedPaperMarkingResult(
        scheme_id=combined.scheme_id,
        student_paper_id=combined.student_paper_id,
        test_results=tuple(reviewed_tests),
        confirmed_objective_marks=confirmed_objective,
        confirmed_subjective_marks=confirmed_subjective,
        confirmed_marks=confirmed,
        review_required_marks=review_required,
        unsupported_marks=unsupported,
        maximum_known_marks=combined.maximum_known_marks,
        final_score=(
            confirmed
            if review_required == 0 and unsupported == 0 and fully_resolved
            else None
        ),
        processing_duration_ms=combined.processing_duration_ms,
        engine_versions=engine_versions,
    )


def _review_objective(
    result: CriterionMarkResult,
    decision: HumanReviewDecision,
) -> CriterionMarkResult:
    return CriterionMarkResult.model_validate(
        {
            **result.model_dump(),
            "status": MarkingStatus.MARKED,
            "awarded_marks": decision.awarded_marks,
            "decision": CriterionDecision.MANUAL_REVIEW,
            "component_results": (),
        }
    )


def _review_subjective(
    result: SubjectiveCriterionMarkResult,
    decision: HumanReviewDecision,
) -> SubjectiveCriterionMarkResult:
    return SubjectiveCriterionMarkResult.model_validate(
        {
            **result.model_dump(),
            "status": SubjectiveMarkingStatus.MARKED,
            "awarded_marks": decision.awarded_marks,
            "feedback": decision.rationale,
            "reason_codes": (*result.reason_codes, "human_reviewed"),
        }
    )

"""Tests for pausing, validating, and resuming human marking review."""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID

import pytest
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.graph.checkpoints import graph_config_for_run
from app.graph.nodes import create_human_review_node
from app.graph.review import (
    HumanReviewDecision,
    HumanReviewResponse,
    apply_human_review,
    build_human_review_payload,
)
from app.graph.routing import route_after_aggregation
from app.graph.state import PaperMarkingState
from app.marking.models import (
    CriterionDecision,
    CriterionMarkResult,
    MarkingProvenance,
    MarkingReasonCode,
    MarkingStatus,
    MarkingWarning,
)
from app.marking.subjective.models import (
    CombinedPaperMarkingResult,
    CombinedTestMarkResult,
    RubricEvidenceReference,
    SubjectiveCriterionMarkResult,
    SubjectiveMarkingProvenance,
    SubjectiveMarkingStatus,
    SubjectiveReasonCode,
)
from app.marking_scheme.models import CriterionType
from app.orchestration.models import FinalPaperStatus, PipelineStage

RUN_ID = UUID("30000000-0000-0000-0000-000000000003")
CRITERION_ID = "test-14-criterion-001"


def unresolved_combined_result() -> CombinedPaperMarkingResult:
    criterion = SubjectiveCriterionMarkResult(
        criterion_id=CRITERION_ID,
        test_number=14,
        criterion_type=CriterionType.HOLISTIC,
        status=SubjectiveMarkingStatus.REVIEW_REQUIRED,
        rubric_evidence=RubricEvidenceReference(
            criterion_id=CRITERION_ID,
            test_number=14,
            criterion_type=CriterionType.HOLISTIC,
            description="Assess relevance and language.",
            maximum_marks=Decimal("5"),
            source_page=2,
        ),
        maximum_marks=Decimal("5"),
        reason_codes=(SubjectiveReasonCode.LOW_MODEL_CONFIDENCE,),
        confidence=0.4,
        processing_duration_ms=0,
        provenance=SubjectiveMarkingProvenance(
            service_version="test-service-v1",
            validation_version="test-validation-v1",
            prompt_version="test-prompt-v1",
            provider="test-provider",
            model_version="test-model-v1",
        ),
    )
    test_result = CombinedTestMarkResult(
        test_number=14,
        objective_results=(),
        subjective_results=(criterion,),
        confirmed_objective_marks=Decimal("0"),
        confirmed_subjective_marks=Decimal("0"),
        confirmed_marks=Decimal("0"),
        unresolved_marks=Decimal("5"),
        unsupported_marks=Decimal("0"),
        maximum_known_marks=Decimal("5"),
    )
    return CombinedPaperMarkingResult(
        scheme_id=UUID("40000000-0000-0000-0000-000000000004"),
        student_paper_id=UUID("50000000-0000-0000-0000-000000000005"),
        test_results=(test_result,),
        confirmed_objective_marks=Decimal("0"),
        confirmed_subjective_marks=Decimal("0"),
        confirmed_marks=Decimal("0"),
        review_required_marks=Decimal("5"),
        unsupported_marks=Decimal("0"),
        maximum_known_marks=Decimal("5"),
        processing_duration_ms=0,
        engine_versions=("objective-v1", "subjective-v1"),
    )


def test_human_review_interrupt_resumes_and_updates_final_score() -> None:
    builder = StateGraph(PaperMarkingState)
    builder.add_node(
        "human_review",
        RunnableLambda(create_human_review_node()),
    )
    builder.add_node(
        "completed",
        RunnableLambda(
            lambda state: {
                "workflow_status": FinalPaperStatus.COMPLETED,
                "review_required": False,
            }
        ),
    )
    builder.add_node(
        "review_required",
        RunnableLambda(
            lambda state: {
                "workflow_status": FinalPaperStatus.COMPLETED_WITH_REVIEW,
                "review_required": True,
            }
        ),
    )
    builder.add_edge(START, "human_review")
    builder.add_conditional_edges(
        "human_review",
        route_after_aggregation,
        {"completed": "completed", "review_required": "review_required"},
    )
    builder.add_edge("completed", END)
    builder.add_edge("review_required", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = graph_config_for_run(RUN_ID)
    initial_state: PaperMarkingState = {
        "run_id": str(RUN_ID),
        "combined_result": unresolved_combined_result(),
    }

    paused = graph.invoke(initial_state, config=config)

    assert paused["__interrupt__"][0].value == {
        "kind": "paper_marking_review",
        "run_id": str(RUN_ID),
        "items": [
            {
                "test_number": 14,
                "criterion_id": CRITERION_ID,
                "criterion_type": "subjective",
                "status": "review_required",
                "student_answer": "",
                "rubric": "Assess relevance and language.",
                "maximum_marks": "5",
                "suggested_marks": None,
                "confidence": 0.4,
                "can_assign_mark": True,
            }
        ],
    }
    resumed = graph.invoke(
        Command(
            resume={
                "reviewer_id": "teacher-7",
                "decisions": [
                    {
                        "test_number": 14,
                        "criterion_id": CRITERION_ID,
                        "awarded_marks": "4",
                        "rationale": "Relevant response with minor language errors.",
                    }
                ],
            }
        ),
        config=config,
    )

    result = resumed["combined_result"]
    assert result.final_score == Decimal("4")
    assert result.review_required_marks == 0
    assert result.test_results[0].subjective_results[0].status is (
        SubjectiveMarkingStatus.MARKED
    )
    assert result.test_results[0].subjective_results[0].feedback == (
        "Relevant response with minor language errors."
    )
    assert result.engine_versions == (
        "objective-v1",
        "subjective-v1",
        "human-review-v1",
    )
    assert resumed["current_stage"] is PipelineStage.HUMAN_REVIEW
    assert resumed["human_review_response"].reviewer_id == "teacher-7"
    assert resumed["workflow_status"] is FinalPaperStatus.COMPLETED
    assert resumed["review_required"] is False


@pytest.mark.parametrize(
    ("decision", "error"),
    [
        (
            HumanReviewDecision(
                test_number=14,
                criterion_id="not-a-criterion",
                awarded_marks=Decimal("1"),
                rationale="Invalid target.",
            ),
            "unknown or non-reviewable",
        ),
        (
            HumanReviewDecision(
                test_number=14,
                criterion_id=CRITERION_ID,
                awarded_marks=Decimal("6"),
                rationale="Over maximum.",
            ),
            "exceed the criterion maximum",
        ),
    ],
)
def test_human_review_rejects_unscoped_or_over_maximum_decisions(
    decision: HumanReviewDecision,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        apply_human_review(
            unresolved_combined_result(),
            HumanReviewResponse(reviewer_id="teacher-7", decisions=(decision,)),
        )


def test_empty_human_review_response_keeps_score_unresolved() -> None:
    result = apply_human_review(
        unresolved_combined_result(),
        HumanReviewResponse(reviewer_id="teacher-7"),
    )

    assert result.final_score is None
    assert result.review_required_marks == Decimal("5")


def test_human_review_updates_unresolved_objective_criterion() -> None:
    criterion = CriterionMarkResult(
        criterion_id=CRITERION_ID,
        test_number=14,
        criterion_type=CriterionType.EXACT_ANSWER,
        status=MarkingStatus.REVIEW_REQUIRED,
        maximum_marks=Decimal("2"),
        decision=CriterionDecision.MANUAL_REVIEW,
        reason_codes=(MarkingReasonCode.STUDENT_REVIEW_REQUIRED,),
        warnings=(MarkingWarning.MANUAL_REVIEW_REQUIRED,),
        provenance=MarkingProvenance(
            engine_version="test-engine-v1",
            normalization_version="test-normalization-v1",
            matcher="test-review",
            criterion_id=CRITERION_ID,
            criterion_source_page=2,
        ),
    )
    test_result = CombinedTestMarkResult(
        test_number=14,
        objective_results=(criterion,),
        subjective_results=(),
        confirmed_objective_marks=Decimal("0"),
        confirmed_subjective_marks=Decimal("0"),
        confirmed_marks=Decimal("0"),
        unresolved_marks=Decimal("2"),
        unsupported_marks=Decimal("0"),
        maximum_known_marks=Decimal("2"),
    )
    combined = CombinedPaperMarkingResult(
        scheme_id=UUID("40000000-0000-0000-0000-000000000004"),
        student_paper_id=UUID("50000000-0000-0000-0000-000000000005"),
        test_results=(test_result,),
        confirmed_objective_marks=Decimal("0"),
        confirmed_subjective_marks=Decimal("0"),
        confirmed_marks=Decimal("0"),
        review_required_marks=Decimal("2"),
        unsupported_marks=Decimal("0"),
        maximum_known_marks=Decimal("2"),
        processing_duration_ms=0,
        engine_versions=("objective-v1", "subjective-v1"),
    )

    reviewed = apply_human_review(
        combined,
        HumanReviewResponse(
            reviewer_id="teacher-7",
            decisions=(
                HumanReviewDecision(
                    test_number=14,
                    criterion_id=CRITERION_ID,
                    awarded_marks=Decimal("1"),
                    rationale="Accepted the partially legible answer.",
                ),
            ),
        ),
    )

    assert reviewed.final_score == Decimal("1")
    assert reviewed.test_results[0].objective_results[0].status is MarkingStatus.MARKED
    assert (
        reviewed.test_results[0].objective_results[0].decision
        is CriterionDecision.MANUAL_REVIEW
    )


def test_human_review_payload_marks_missing_maximum_as_non_assignable() -> None:
    combined = unresolved_combined_result()
    test_result = combined.test_results[0]
    criterion = test_result.subjective_results[0].model_copy(
        update={"maximum_marks": None}
    )
    result_without_maximum = CombinedPaperMarkingResult.model_validate(
        {
            **combined.model_dump(),
            "test_results": (
                {
                    **test_result.model_dump(),
                    "subjective_results": (criterion.model_dump(),),
                },
            ),
        }
    )

    payload = build_human_review_payload(result_without_maximum, None, str(RUN_ID))

    assert payload["items"][0]["maximum_marks"] is None
    assert payload["items"][0]["can_assign_mark"] is False

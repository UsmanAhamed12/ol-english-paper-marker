"""Clearly synthetic results for local presentation testing only."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from uuid import UUID, uuid4

from app.marking.models import (
    CriterionDecision,
    CriterionMarkResult,
    MarkingProvenance,
    MarkingReasonCode,
    MarkingStatus,
)
from app.marking.subjective.models import (
    CombinedTestMarkResult,
    RubricEvidenceReference,
    SubjectiveCriterionMarkResult,
    SubjectiveMarkingProvenance,
    SubjectiveMarkingStatus,
    SubjectiveMarkingWarning,
    SubjectiveReasonCode,
)
from app.marking_scheme.models import CriterionType
from app.orchestration.models import (
    ORCHESTRATION_VERSION,
    ComponentVersion,
    CriterionFeedback,
    FinalPaperMarkingResult,
    FinalPaperStatus,
    OrchestrationErrorCode,
    OrchestrationIssue,
    OrchestrationProvenance,
    PipelineStage,
    ReviewItem,
    StageStatus,
    StageSummary,
)

_PAPER_ID = UUID("70000000-0000-0000-0000-000000000007")
_SCHEME_ID = UUID("71000000-0000-0000-0000-000000000007")


class DemoScenario(StrEnum):
    COMPLETED = "Completed"
    REVIEW = "Review required"
    PARTIAL = "Partially completed"
    FAILED = "Failed"


def synthetic_demo_result(scenario: DemoScenario) -> FinalPaperMarkingResult:
    """Return synthetic data that cannot be mistaken for real marking."""

    run_id = uuid4()
    if scenario is DemoScenario.FAILED:
        code = OrchestrationErrorCode.INVALID_MARKING_SCHEME
        return FinalPaperMarkingResult(
            run_id=run_id,
            status=FinalPaperStatus.FAILED,
            stages=(
                StageSummary(
                    stage=PipelineStage.INPUT_VALIDATION,
                    status=StageStatus.FAILED,
                    duration_ms=12,
                    error_code=code,
                ),
            ),
            errors=(
                OrchestrationIssue(
                    code=code,
                    stage=PipelineStage.INPUT_VALIDATION,
                    message="Synthetic marking-scheme validation failure.",
                ),
            ),
            total_duration_ms=12,
            provenance=_provenance(),
        )

    objective = _objective()
    if scenario is DemoScenario.PARTIAL:
        test = CombinedTestMarkResult(
            test_number=1,
            objective_results=(objective,),
            subjective_results=(),
            confirmed_objective_marks=Decimal("4"),
            confirmed_subjective_marks=Decimal("0"),
            confirmed_marks=Decimal("4"),
            unresolved_marks=Decimal("0"),
            unsupported_marks=Decimal("0"),
            maximum_known_marks=Decimal("4"),
            final_score=Decimal("4"),
        )
        return _result(
            run_id=run_id,
            status=FinalPaperStatus.PARTIALLY_COMPLETED,
            tests=(test,),
            detected=(1,),
            missing=(2,),
            objective=Decimal("4"),
            subjective=Decimal("0"),
            maximum=Decimal("4"),
            review=Decimal("0"),
            review_items=(
                ReviewItem(
                    test_number=2,
                    reason="test_alignment_incomplete",
                    status="review_required",
                    possible_marks=Decimal("0"),
                    source_reference="synthetic-test-02",
                ),
            ),
        )

    subjective = _subjective(marked=scenario is DemoScenario.COMPLETED)
    is_complete = scenario is DemoScenario.COMPLETED
    test = CombinedTestMarkResult(
        test_number=1,
        objective_results=(objective,),
        subjective_results=(subjective,),
        confirmed_objective_marks=Decimal("4"),
        confirmed_subjective_marks=Decimal("6") if is_complete else Decimal("0"),
        confirmed_marks=Decimal("10") if is_complete else Decimal("4"),
        unresolved_marks=Decimal("0") if is_complete else Decimal("6"),
        unsupported_marks=Decimal("0"),
        maximum_known_marks=Decimal("10"),
        final_score=Decimal("10") if is_complete else None,
    )
    return _result(
        run_id=run_id,
        status=(
            FinalPaperStatus.COMPLETED
            if is_complete
            else FinalPaperStatus.COMPLETED_WITH_REVIEW
        ),
        tests=(test,),
        detected=(1,),
        missing=(),
        objective=Decimal("4"),
        subjective=Decimal("6") if is_complete else Decimal("0"),
        maximum=Decimal("10"),
        review=Decimal("0") if is_complete else Decimal("6"),
        review_items=(
            ()
            if is_complete
            else (
                ReviewItem(
                    test_number=1,
                    criterion_id="synthetic-content-01",
                    reason="low_model_confidence",
                    status="review_required",
                    possible_marks=Decimal("6"),
                    source_reference="synthetic-content-01",
                ),
            )
        ),
        feedback=(
            CriterionFeedback(
                test_number=1,
                criterion_id="synthetic-content-01",
                status=subjective.status.value,
                feedback=subjective.feedback,
            ),
        ),
    )


def _result(
    *,
    run_id: UUID,
    status: FinalPaperStatus,
    tests: tuple[CombinedTestMarkResult, ...],
    detected: tuple[int, ...],
    missing: tuple[int, ...],
    objective: Decimal,
    subjective: Decimal,
    maximum: Decimal,
    review: Decimal,
    review_items: tuple[ReviewItem, ...],
    feedback: tuple[CriterionFeedback, ...] = (),
) -> FinalPaperMarkingResult:
    confirmed = objective + subjective
    return FinalPaperMarkingResult(
        run_id=run_id,
        status=status,
        student_document_id=_PAPER_ID,
        marking_scheme_id=_SCHEME_ID,
        detected_tests=detected,
        missing_tests=missing,
        aligned_tests=tuple(test.test_number for test in tests),
        objective_confirmed_marks=objective,
        subjective_confirmed_marks=subjective,
        total_confirmed_marks=confirmed,
        known_maximum_marks=maximum,
        review_required_marks=review,
        unsupported_marks=Decimal("0"),
        unresolved_marks=review,
        final_score=confirmed if status is FinalPaperStatus.COMPLETED else None,
        test_results=tests,
        feedback=feedback,
        review_items=review_items,
        stages=_completed_stages(),
        warnings=("synthetic_demo_only",),
        total_duration_ms=1250,
        provenance=_provenance(),
    )


def _objective() -> CriterionMarkResult:
    return CriterionMarkResult(
        criterion_id="synthetic-exact-01",
        test_number=1,
        criterion_type=CriterionType.EXACT_ANSWER,
        status=MarkingStatus.MARKED,
        awarded_marks=Decimal("4"),
        maximum_marks=Decimal("4"),
        decision=CriterionDecision.EXACT_MATCH,
        reason_codes=(MarkingReasonCode.EXACT_ANSWER_MATCH,),
        provenance=MarkingProvenance(
            engine_version="synthetic-engine-v1",
            normalization_version="synthetic-normalization-v1",
            matcher="synthetic-exact",
            criterion_id="synthetic-exact-01",
            criterion_source_page=1,
        ),
    )


def _subjective(*, marked: bool) -> SubjectiveCriterionMarkResult:
    return SubjectiveCriterionMarkResult(
        criterion_id="synthetic-content-01",
        test_number=1,
        criterion_type=CriterionType.CONTENT_POINT,
        status=(
            SubjectiveMarkingStatus.MARKED
            if marked
            else SubjectiveMarkingStatus.REVIEW_REQUIRED
        ),
        rubric_evidence=RubricEvidenceReference(
            criterion_id="synthetic-content-01",
            test_number=1,
            criterion_type=CriterionType.CONTENT_POINT,
            description="Synthetic demonstration content criterion.",
            maximum_marks=Decimal("6"),
            source_page=1,
        ),
        awarded_marks=Decimal("6") if marked else None,
        advisory_marks=None if marked else Decimal("3"),
        maximum_marks=Decimal("6"),
        matched_requirements=("synthetic requirement",),
        reason_codes=(
            SubjectiveReasonCode.RUBRIC_ASSESSMENT_ACCEPTED
            if marked
            else SubjectiveReasonCode.LOW_MODEL_CONFIDENCE,
        ),
        feedback=(
            "Synthetic response meets the demonstration criterion."
            if marked
            else "Synthetic response requires examiner review."
        ),
        confidence=0.95 if marked else 0.40,
        warnings=(() if marked else (SubjectiveMarkingWarning.MODEL_CONFIDENCE_LOW,)),
        processing_duration_ms=250,
        provenance=SubjectiveMarkingProvenance(
            service_version="synthetic-subjective-v1",
            validation_version="synthetic-validation-v1",
            prompt_version="synthetic-prompt-v1",
            provider="synthetic-provider",
            model_version="synthetic-model-v1",
        ),
    )


def _completed_stages() -> tuple[StageSummary, ...]:
    return tuple(
        StageSummary(stage=stage, status=StageStatus.COMPLETED, duration_ms=100)
        for stage in PipelineStage
    )


def _provenance() -> OrchestrationProvenance:
    return OrchestrationProvenance(
        orchestration_version=ORCHESTRATION_VERSION,
        components=(ComponentVersion(component="demo", version="synthetic-v1"),),
    )

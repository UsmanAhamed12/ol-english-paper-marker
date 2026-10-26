"""Map validated Phase 6 results to stable, shallow UI contracts."""

from __future__ import annotations

from collections import defaultdict

from app.marking.models import CriterionMarkResult
from app.marking.subjective.models import (
    ImprovementAdviceStatus,
    SubjectiveCriterionMarkResult,
)
from app.orchestration.models import (
    FinalPaperMarkingResult,
    FinalPaperStatus,
    OrchestrationErrorCode,
)
from app.orchestration.serialization import final_result_json
from app.ui.presentation.formatting import format_marks, humanize_code, percentage
from app.ui.presentation.models import (
    CriterionView,
    MetricView,
    ResultViewModel,
    ReviewItemView,
    StageView,
    TestResultView,
    ViewTone,
    WarningView,
)

_ERROR_MESSAGES = {
    OrchestrationErrorCode.INVALID_STUDENT_PDF.value: (
        "Student paper could not be processed as a valid PDF."
    ),
    OrchestrationErrorCode.INVALID_MARKING_SCHEME.value: (
        "Marking scheme could not be processed as a valid PDF."
    ),
    OrchestrationErrorCode.INVALID_OUTPUT_PATH.value: (
        "The selected result location is not private runtime storage."
    ),
    OrchestrationErrorCode.STUDENT_PROCESSING_FAILED.value: (
        "Student-paper processing could not be completed."
    ),
    OrchestrationErrorCode.SCHEME_INGESTION_FAILED.value: (
        "The marking scheme could not be read safely."
    ),
    OrchestrationErrorCode.ALIGNMENT_FAILED.value: (
        "Student answers and marking-scheme Tests could not be aligned."
    ),
    OrchestrationErrorCode.OBJECTIVE_MARKING_FAILED.value: (
        "Objective marking could not be completed safely."
    ),
    OrchestrationErrorCode.SUBJECTIVE_MARKING_PARTIAL.value: (
        "Descriptive marking requires manual review."
    ),
    OrchestrationErrorCode.RESULT_AGGREGATION_FAILED.value: (
        "The final marking result could not be prepared safely."
    ),
}

_STAGE_LABELS = {
    "input_validation": "Validating documents",
    "student_pdf_processing": "Processing student paper",
    "structure_detection": "Detecting Test structure",
    "student_answer_extraction": "Extracting student answers",
    "marking_scheme_ingestion": "Reading marking scheme",
    "test_alignment": "Aligning Tests",
    "objective_marking": "Marking objective answers",
    "subjective_marking": "Marking descriptive answers",
    "result_aggregation": "Preparing final result",
}


def map_final_result(
    result: FinalPaperMarkingResult, *, is_demo: bool = False
) -> ResultViewModel:
    """Create the only view-model representation used by the UI."""

    is_final = (
        result.status is FinalPaperStatus.COMPLETED
        and result.final_score is not None
        and result.unresolved_marks == 0
    )
    status, tone = _status(result.status)
    score_heading = "Final score" if is_final else "Confirmed marks"
    score_value = (
        f"{format_marks(result.final_score)} / "
        f"{format_marks(result.known_maximum_marks)}"
        if is_final and result.final_score is not None
        else (
            f"{format_marks(result.total_confirmed_marks)} of "
            f"{format_marks(result.known_maximum_marks)} known marks"
        )
    )
    feedback_by_test: dict[int, list[str]] = defaultdict(list)
    for item in result.feedback:
        feedback_by_test[item.test_number].append(item.feedback)
    tests = tuple(
        _map_test(test, tuple(feedback_by_test.get(test.test_number, ())))
        for test in result.test_results
    )
    warnings = tuple(
        WarningView(code=code, message=humanize_code(code)) for code in result.warnings
    )
    errors = tuple(
        WarningView(
            code=item.code.value,
            message=_ERROR_MESSAGES.get(item.code.value, item.message),
        )
        for item in result.errors
    )
    return ResultViewModel(
        run_id=str(result.run_id),
        status=status,
        tone=tone,
        score_heading=score_heading,
        score_value=score_value,
        percentage=(
            percentage(result.final_score, result.known_maximum_marks)
            if is_final and result.final_score is not None
            else None
        ),
        unresolved_value=(
            None if is_final else f"{format_marks(result.unresolved_marks)} marks"
        ),
        is_final_score=is_final,
        is_demo=is_demo,
        metrics=(
            MetricView(
                label="Confirmed marks",
                value=format_marks(result.total_confirmed_marks),
            ),
            MetricView(
                label="Known maximum",
                value=format_marks(result.known_maximum_marks),
            ),
            MetricView(label="Tests detected", value=str(len(result.detected_tests))),
            MetricView(
                label="Tests resolved",
                value=str(
                    sum(test.final_score is not None for test in result.test_results)
                ),
            ),
            MetricView(label="Review items", value=str(len(result.review_items))),
            MetricView(label="Missing Tests", value=str(len(result.missing_tests))),
        ),
        tests=tests,
        feedback_by_test=tuple(
            (f"Test {number:02d}", tuple(items))
            for number, items in sorted(feedback_by_test.items())
        ),
        review_items=tuple(
            ReviewItemView(
                test_label=f"Test {item.test_number:02d}",
                criterion_label=item.criterion_id,
                reason=humanize_code(item.reason),
                status=humanize_code(item.status),
                marks_affected=format_marks(item.possible_marks),
            )
            for item in result.review_items
        ),
        warnings=warnings,
        errors=errors,
        stages=tuple(
            StageView(
                label=_STAGE_LABELS[stage.stage.value],
                status=humanize_code(stage.status.value),
                duration=f"{stage.duration_ms / 1000:.2f} s",
            )
            for stage in result.stages
        ),
        download_json=final_result_json(result),
    )


def _map_test(test: object, feedback: tuple[str, ...]) -> TestResultView:
    from app.marking.subjective.models import CombinedTestMarkResult

    if not isinstance(test, CombinedTestMarkResult):
        raise TypeError("Unexpected Test result")
    unresolved = test.unresolved_marks + test.unsupported_marks
    status = "Resolved" if test.final_score is not None else "Review required"
    heading = f"Test {test.test_number:02d} — " + (
        f"{format_marks(test.final_score)} / {format_marks(test.maximum_known_marks)}"
        if test.final_score is not None
        else "Review required"
    )
    criteria = tuple(_map_objective(item) for item in test.objective_results) + tuple(
        _map_subjective(item) for item in test.subjective_results
    )
    warning_codes = tuple(
        dict.fromkeys(
            warning.value
            for item in test.objective_results
            for warning in item.warnings
        )
    ) + tuple(
        dict.fromkeys(
            warning.value
            for item in test.subjective_results
            for warning in item.warnings
        )
    )
    return TestResultView(
        test_number=test.test_number,
        heading=heading,
        status=status,
        confirmed_marks=format_marks(test.confirmed_marks),
        maximum_marks=format_marks(test.maximum_known_marks),
        unresolved_marks=format_marks(unresolved),
        objective_marks=format_marks(test.confirmed_objective_marks),
        subjective_marks=format_marks(test.confirmed_subjective_marks),
        criteria=criteria,
        feedback=feedback,
        warnings=tuple(
            WarningView(code=code, message=humanize_code(code))
            for code in dict.fromkeys(warning_codes)
        ),
    )


def _map_objective(result: CriterionMarkResult) -> CriterionView:
    maximum = (
        format_marks(result.maximum_marks)
        if result.maximum_marks is not None
        else "unknown"
    )
    marks = (
        f"{format_marks(result.awarded_marks)} / {maximum}"
        if result.awarded_marks is not None
        else f"Unresolved / {maximum}"
    )
    return CriterionView(
        criterion_id=result.criterion_id,
        criterion_type=humanize_code(result.criterion_type.value),
        status=humanize_code(result.status.value),
        marks=marks,
        decision=humanize_code(result.decision.value),
        reason=humanize_code(result.reason_codes[0].value),
    )


def _map_subjective(result: SubjectiveCriterionMarkResult) -> CriterionView:
    maximum = (
        format_marks(result.maximum_marks)
        if result.maximum_marks is not None
        else "unknown"
    )
    marks = (
        f"{format_marks(result.awarded_marks)} / {maximum}"
        if result.awarded_marks is not None
        else f"Unresolved / {maximum}"
    )
    reason = result.reason_codes[0]
    return CriterionView(
        criterion_id=result.criterion_id,
        criterion_type=humanize_code(result.criterion_type.value),
        status=humanize_code(result.status.value),
        marks=marks,
        decision="Rubric-guided assessment",
        reason=humanize_code(reason.value if hasattr(reason, "value") else str(reason)),
        feedback=result.feedback or None,
        improvement_tips=result.improvement_tips,
        syllabus_references=tuple(
            f"{reference.source_name} · page {reference.page_number}"
            for reference in result.syllabus_references
        ),
        improvement_advice_status=(
            result.improvement_advice_status.value
            if result.improvement_advice_status
            is not ImprovementAdviceStatus.NOT_GENERATED
            else None
        ),
    )


def _status(status: FinalPaperStatus) -> tuple[str, ViewTone]:
    return {
        FinalPaperStatus.COMPLETED: ("Completed", ViewTone.SUCCESS),
        FinalPaperStatus.COMPLETED_WITH_REVIEW: (
            "Review required",
            ViewTone.WARNING,
        ),
        FinalPaperStatus.PARTIALLY_COMPLETED: (
            "Partially completed",
            ViewTone.WARNING,
        ),
        FinalPaperStatus.FAILED: ("Failed", ViewTone.ERROR),
    }[status]

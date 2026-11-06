"""Framework-independent Streamlit presentation semantics."""

from __future__ import annotations

import json

import pytest

from app.answers.models import AnswerExtractionStatus
from app.feedback.models import QuestionFeedback, QuestionFeedbackStatus
from app.ui.demo import DemoScenario, synthetic_demo_result
from app.ui.presentation.formatting import humanize_code
from app.ui.presentation.mapper import map_final_result
from app.ui.presentation.models import ViewTone


def test_completed_result_exposes_legitimate_final_score() -> None:
    view = map_final_result(synthetic_demo_result(DemoScenario.COMPLETED), is_demo=True)
    assert view.status == "Completed"
    assert view.tone is ViewTone.SUCCESS
    assert view.score_heading == "Normalized score out of 100"
    assert view.score_value == "100.00 / 100"
    assert view.raw_score_value == "10 / 10"
    assert view.percentage == "100.00%"
    assert view.unresolved_value is None
    assert view.is_final_score is True
    assert len(view.tests[0].criteria) == 2
    assert view.is_demo is True


def test_question_feedback_and_provisional_status_are_presented() -> None:
    result = synthetic_demo_result(DemoScenario.REVIEW).model_copy(
        update={
            "question_feedback": (
                QuestionFeedback(
                    test_number=1,
                    status=QuestionFeedbackStatus.PROVISIONAL,
                    feedback=(
                        "Check the transcription before interpreting this response."
                    ),
                    improvement_tips=("Use a relevant supporting detail.",),
                    answer_status=AnswerExtractionStatus.REVIEW_REQUIRED,
                    provider="ollama",
                    model_version="llama2:latest",
                    prompt_version="question-feedback-v2",
                ),
            )
        }
    )

    view = map_final_result(result)

    assert view.score_heading == "Confirmed marks"
    assert view.score_value == "4 of 10 known marks"
    assert view.percentage is None
    assert view.tests[0].formative_feedback_status == "provisional"
    assert view.tests[0].formative_feedback == (
        "Check the transcription before interpreting this response."
    )
    assert view.tests[0].improvement_tips == ("Use a relevant supporting detail.",)


def test_review_result_never_presents_confirmed_marks_as_final() -> None:
    view = map_final_result(synthetic_demo_result(DemoScenario.REVIEW))
    assert view.status == "Review required"
    assert view.score_heading == "Confirmed marks"
    assert view.score_value == "4 of 10 known marks"
    assert view.percentage is None
    assert view.unresolved_value == "6 marks"
    assert view.is_final_score is False
    assert view.review_items[0].marks_affected == "6"
    assert view.tests[0].heading == "Test 01 — Review required"


def test_partial_result_preserves_useful_marks_and_missing_test() -> None:
    view = map_final_result(synthetic_demo_result(DemoScenario.PARTIAL))
    metrics = {item.label: item.value for item in view.metrics}
    assert view.status == "Partially completed"
    assert view.tests[0].status == "Resolved"
    assert metrics["Missing Tests"] == "1"
    assert view.is_final_score is False
    assert view.review_items[0].test_label == "Test 02"


def test_failed_result_uses_safe_error_message() -> None:
    view = map_final_result(synthetic_demo_result(DemoScenario.FAILED))
    assert view.status == "Failed"
    assert view.tone is ViewTone.ERROR
    assert view.tests == ()
    assert view.errors[0].message == (
        "Marking scheme could not be processed as a valid PDF."
    )
    assert "Traceback" not in view.download_json


@pytest.mark.parametrize("scenario", tuple(DemoScenario))
def test_every_synthetic_status_serializes_deterministically(
    scenario: DemoScenario,
) -> None:
    result = synthetic_demo_result(scenario)
    first = map_final_result(result, is_demo=True)
    second = map_final_result(result, is_demo=True)
    assert first == second
    assert json.loads(first.download_json)["status"] == result.status.value


def test_warning_code_is_humanized_without_losing_code() -> None:
    view = map_final_result(synthetic_demo_result(DemoScenario.COMPLETED))
    assert view.warnings[0].code == "synthetic_demo_only"
    assert view.warnings[0].message == "Synthetic demo only"
    assert humanize_code("missing_tests:3") == "Missing tests (3)"

"""Tests for held-out historical marking evaluation."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.evaluation.dataset import dataset_fingerprint, load_dataset
from app.evaluation.evaluator import evaluate_run
from app.evaluation.metrics import calculate_mark_metrics
from app.evaluation.models import (
    EvaluationFailureStage,
    EvaluationOutcomeStatus,
    EvaluationPrediction,
    EvaluationPredictionRun,
    HistoricalEvaluationDataset,
    TeacherQuestionReference,
)
from app.evaluation.reports import serialize_report


def _dataset() -> HistoricalEvaluationDataset:
    return HistoricalEvaluationDataset(
        dataset_id="synthetic-set",
        dataset_version="v1",
        references=(
            TeacherQuestionReference(
                paper_alias="paper-a",
                test_number=1,
                awarded_marks=Decimal("4"),
                maximum_marks=Decimal("5"),
            ),
            TeacherQuestionReference(
                paper_alias="paper-a",
                test_number=2,
                awarded_marks=Decimal("8"),
                maximum_marks=Decimal("10"),
            ),
            TeacherQuestionReference(
                paper_alias="paper-b",
                test_number=1,
                awarded_marks=Decimal("3"),
                maximum_marks=Decimal("5"),
            ),
            TeacherQuestionReference(
                paper_alias="paper-b",
                test_number=2,
                awarded_marks=Decimal("6"),
                maximum_marks=Decimal("10"),
            ),
        ),
    )


def _prediction(
    paper_alias: str,
    test_number: int,
    marks: str | None,
    *,
    status: EvaluationOutcomeStatus = EvaluationOutcomeStatus.SCORED,
    stage: EvaluationFailureStage | None = None,
) -> EvaluationPrediction:
    return EvaluationPrediction(
        paper_alias=paper_alias,
        test_number=test_number,
        status=status,
        predicted_marks=Decimal(marks) if marks is not None else None,
        failure_stage=stage,
    )


def _run(*predictions: EvaluationPrediction) -> EvaluationPredictionRun:
    return EvaluationPredictionRun(
        run_id=UUID("d0000000-0000-0000-0000-00000000000d"),
        model_name="synthetic-model",
        model_version="synthetic-model-v1",
        prompt_version="synthetic-prompt-v1",
        pipeline_version="synthetic-pipeline-v1",
        predictions=predictions,
    )


def test_metrics_calculate_errors_bias_and_threshold_rates() -> None:
    metrics = calculate_mark_metrics(
        [
            (Decimal("10"), Decimal("8")),
            (Decimal("10"), Decimal("13")),
        ]
    )

    assert metrics.sample_count == 2
    assert metrics.mae == Decimal("2.5")
    assert metrics.rmse == Decimal("6.5").sqrt()
    assert metrics.mean_signed_error == Decimal("0.5")
    assert metrics.within_1_mark_percent == Decimal("0")
    assert metrics.within_2_marks_percent == Decimal("50")
    assert metrics.within_5_marks_percent == Decimal("100")


def test_empty_metrics_are_explicitly_undefined() -> None:
    metrics = calculate_mark_metrics([])

    assert metrics.sample_count == 0
    assert metrics.mae is None
    assert metrics.rmse is None
    assert metrics.mean_signed_error is None
    assert metrics.within_1_mark_percent is None


def test_prediction_contract_rejects_teacher_reference_fields() -> None:
    with pytest.raises(ValidationError, match="teacher_awarded_marks"):
        EvaluationPrediction.model_validate(
            {
                "paper_alias": "paper-a",
                "test_number": 1,
                "status": "scored",
                "predicted_marks": "4",
                "teacher_awarded_marks": "4",
            }
        )


def test_teacher_reference_rejects_award_above_maximum() -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        TeacherQuestionReference(
            paper_alias="paper-a",
            test_number=1,
            awarded_marks=Decimal("6"),
            maximum_marks=Decimal("5"),
        )


def test_evaluator_reports_question_and_complete_paper_metrics() -> None:
    report = evaluate_run(
        _dataset(),
        _run(
            _prediction("paper-a", 1, "5"),
            _prediction("paper-a", 2, "7"),
            _prediction("paper-b", 1, "3"),
            _prediction(
                "paper-b",
                2,
                None,
                status=EvaluationOutcomeStatus.REVIEW_REQUIRED,
                stage=EvaluationFailureStage.SEGMENTATION,
            ),
        ),
    )

    assert report.question_coverage.expected_count == 4
    assert report.question_coverage.scored_count == 3
    assert report.question_coverage.coverage_percent == Decimal("75")
    assert report.question_metrics.sample_count == 3
    assert report.question_metrics.mean_signed_error == 0
    assert report.per_question[0].test_number == 1
    assert report.per_question[0].metrics.mae == Decimal("0.5")
    assert report.per_question[1].coverage.coverage_percent == Decimal("50")
    assert report.paper_coverage.expected_count == 2
    assert report.paper_coverage.scored_count == 1
    assert report.paper_total_metrics.sample_count == 1
    assert report.paper_total_metrics.mae == 0
    assert report.unscored_prediction_count == 1
    assert report.missing_prediction_count == 0
    assert report.failure_stage_counts[0].stage is EvaluationFailureStage.SEGMENTATION
    assert report.failure_stage_counts[0].count == 1


def test_incomplete_paper_is_not_scored_as_a_partial_total() -> None:
    report = evaluate_run(
        _dataset(),
        _run(_prediction("paper-a", 1, "4")),
    )

    assert report.paper_coverage.scored_count == 0
    assert report.paper_total_metrics.sample_count == 0
    assert report.paper_total_metrics.mae is None
    assert report.missing_prediction_count == 3


def test_predictions_outside_dataset_are_rejected() -> None:
    with pytest.raises(ValueError, match="outside the dataset"):
        evaluate_run(_dataset(), _run(_prediction("paper-a", 3, "0")))


def test_prediction_above_question_maximum_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds the maximum"):
        evaluate_run(
            _dataset(),
            _run(_prediction("paper-a", 1, "6")),
        )


def test_dataset_load_and_fingerprint_are_reproducible(tmp_path: Path) -> None:
    dataset = _dataset()
    path = tmp_path / "synthetic-evaluation.json"
    path.write_text(
        json.dumps(dataset.model_dump(mode="json")),
        encoding="utf-8",
    )

    loaded = load_dataset(path)

    assert loaded == dataset
    assert dataset_fingerprint(loaded) == dataset_fingerprint(dataset)


def test_serialized_report_is_deterministic_and_contains_no_answer_text() -> None:
    report = evaluate_run(
        _dataset(),
        _run(
            _prediction("paper-a", 1, "4"),
            _prediction("paper-a", 2, "8"),
            _prediction("paper-b", 1, "3"),
            _prediction("paper-b", 2, "6"),
        ),
    )

    first = serialize_report(report)

    assert first == serialize_report(report)
    assert "student answer" not in first.lower()
    assert "dataset_fingerprint" in first

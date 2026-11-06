"""Compare frozen predictions with human references after grading completes."""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal

from app.evaluation.dataset import dataset_fingerprint
from app.evaluation.metrics import calculate_mark_metrics
from app.evaluation.models import (
    EvaluationCoverage,
    EvaluationFailureStage,
    EvaluationOutcomeStatus,
    EvaluationPredictionRun,
    FailureStageCount,
    HistoricalEvaluationDataset,
    HistoricalEvaluationReport,
    QuestionEvaluationSummary,
    TeacherQuestionReference,
)


def evaluate_run(
    dataset: HistoricalEvaluationDataset,
    run: EvaluationPredictionRun,
) -> HistoricalEvaluationReport:
    """Score only complete predictions; retain coverage and failure counts."""

    reference_by_key = {
        (reference.paper_alias, reference.test_number): reference
        for reference in dataset.references
    }
    prediction_by_key = {
        (prediction.paper_alias, prediction.test_number): prediction
        for prediction in run.predictions
    }
    unexpected = prediction_by_key.keys() - reference_by_key.keys()
    if unexpected:
        raise ValueError("Prediction run contains questions outside the dataset")

    pairs_by_test: dict[int, list[tuple[Decimal, Decimal]]] = defaultdict(list)
    teacher_totals: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    predicted_totals: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    references_by_paper: dict[str, list[TeacherQuestionReference]] = defaultdict(list)
    references_by_test: dict[int, list[TeacherQuestionReference]] = defaultdict(list)
    failure_stages: Counter[EvaluationFailureStage] = Counter()
    scored_question_count = 0
    missing_prediction_count = 0
    unscored_prediction_count = 0

    for reference in dataset.references:
        key = (reference.paper_alias, reference.test_number)
        references_by_paper[reference.paper_alias].append(reference)
        references_by_test[reference.test_number].append(reference)
        teacher_totals[reference.paper_alias] += reference.awarded_marks
        prediction = prediction_by_key.get(key)
        if prediction is None:
            missing_prediction_count += 1
            continue
        if prediction.status is not EvaluationOutcomeStatus.SCORED:
            unscored_prediction_count += 1
            if prediction.failure_stage is not None:
                failure_stages[prediction.failure_stage] += 1
            continue
        predicted_marks = prediction.predicted_marks
        if predicted_marks is None:
            raise ValueError("Scored prediction is missing its mark")
        if predicted_marks > reference.maximum_marks:
            raise ValueError(
                f"Prediction exceeds the maximum for "
                f"{reference.paper_alias} Test {reference.test_number}"
            )
        pairs_by_test[reference.test_number].append(
            (reference.awarded_marks, predicted_marks)
        )
        predicted_totals[reference.paper_alias] += predicted_marks
        scored_question_count += 1

    complete_papers = {
        alias
        for alias, references in references_by_paper.items()
        if all(
            (reference.paper_alias, reference.test_number) in prediction_by_key
            and prediction_by_key[(reference.paper_alias, reference.test_number)].status
            is EvaluationOutcomeStatus.SCORED
            for reference in references
        )
    }
    paper_pairs = [
        (teacher_totals[alias], predicted_totals[alias])
        for alias in sorted(complete_papers)
    ]

    per_question = tuple(
        _question_summary(
            test_number=test_number,
            references=references_by_test[test_number],
            pairs=pairs_by_test[test_number],
        )
        for test_number in sorted(
            {reference.test_number for reference in dataset.references}
        )
    )
    expected_question_count = len(dataset.references)
    expected_paper_count = len(references_by_paper)

    return HistoricalEvaluationReport(
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        dataset_fingerprint=dataset_fingerprint(dataset),
        run_id=run.run_id,
        model_name=run.model_name,
        model_version=run.model_version,
        prompt_version=run.prompt_version,
        pipeline_version=run.pipeline_version,
        question_coverage=_coverage(expected_question_count, scored_question_count),
        question_metrics=calculate_mark_metrics(
            [pair for test_pairs in pairs_by_test.values() for pair in test_pairs]
        ),
        per_question=per_question,
        paper_coverage=_coverage(expected_paper_count, len(complete_papers)),
        paper_total_metrics=calculate_mark_metrics(paper_pairs),
        missing_prediction_count=missing_prediction_count,
        unscored_prediction_count=unscored_prediction_count,
        failure_stage_counts=tuple(
            FailureStageCount(stage=stage, count=failure_stages[stage])
            for stage in sorted(failure_stages, key=lambda item: item.value)
        ),
    )


def _question_summary(
    test_number: int,
    references: list[TeacherQuestionReference],
    pairs: list[tuple[Decimal, Decimal]],
) -> QuestionEvaluationSummary:
    return QuestionEvaluationSummary(
        test_number=test_number,
        coverage=_coverage(len(references), len(pairs)),
        metrics=calculate_mark_metrics(pairs),
    )


def _coverage(expected_count: int, scored_count: int) -> EvaluationCoverage:
    percentage = (
        Decimal("0")
        if expected_count == 0
        else Decimal(scored_count * 100) / Decimal(expected_count)
    )
    return EvaluationCoverage(
        expected_count=expected_count,
        scored_count=scored_count,
        coverage_percent=percentage,
    )

"""Dependency-free deterministic mark-comparison metrics."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from app.evaluation.models import MarkMetrics


def calculate_mark_metrics(
    score_pairs: Sequence[tuple[Decimal, Decimal]],
) -> MarkMetrics:
    """Calculate errors for (teacher mark, predicted mark) pairs."""

    if not score_pairs:
        return MarkMetrics(
            sample_count=0,
            mae=None,
            rmse=None,
            mean_signed_error=None,
            within_1_mark_percent=None,
            within_2_marks_percent=None,
            within_5_marks_percent=None,
        )

    errors = [predicted - teacher for teacher, predicted in score_pairs]
    count = Decimal(len(errors))
    absolute_errors = [abs(error) for error in errors]
    return MarkMetrics(
        sample_count=len(errors),
        mae=sum(absolute_errors, Decimal("0")) / count,
        rmse=(sum((error * error for error in errors), Decimal("0")) / count).sqrt(),
        mean_signed_error=sum(errors, Decimal("0")) / count,
        within_1_mark_percent=_within_percent(absolute_errors, Decimal("1")),
        within_2_marks_percent=_within_percent(absolute_errors, Decimal("2")),
        within_5_marks_percent=_within_percent(absolute_errors, Decimal("5")),
    )


def _within_percent(errors: list[Decimal], threshold: Decimal) -> Decimal:
    within = sum(error <= threshold for error in errors)
    return Decimal(within * 100) / Decimal(len(errors))

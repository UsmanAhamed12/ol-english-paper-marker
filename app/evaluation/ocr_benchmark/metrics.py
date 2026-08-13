"""Dependency-free OCR edit-distance metrics."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from app.evaluation.ocr_benchmark.models import ErrorRate


def normalize_metric_text(text: str) -> str:
    """Apply the benchmark-only comparison policy without changing evidence."""

    return " ".join(unicodedata.normalize("NFC", text).split())


def _edit_distance(reference: Sequence[str], prediction: Sequence[str]) -> int:
    """Calculate Levenshtein distance using linear auxiliary memory."""

    previous = list(range(len(prediction) + 1))
    for reference_index, reference_item in enumerate(reference, start=1):
        current = [reference_index]
        for prediction_index, prediction_item in enumerate(prediction, start=1):
            substitution_cost = int(reference_item != prediction_item)
            current.append(
                min(
                    current[-1] + 1,
                    previous[prediction_index] + 1,
                    previous[prediction_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def _error_rate(reference: Sequence[str], prediction: Sequence[str]) -> ErrorRate:
    errors = _edit_distance(reference, prediction)
    reference_units = len(reference)
    if reference_units:
        rate: float | None = errors / reference_units
    elif not prediction:
        rate = 0.0
    else:
        rate = None
    return ErrorRate(errors=errors, reference_units=reference_units, rate=rate)


def calculate_cer(reference: str, prediction: str) -> ErrorRate:
    """Calculate character error rate after benchmark-only normalization."""

    normalized_reference = normalize_metric_text(reference)
    normalized_prediction = normalize_metric_text(prediction)
    return _error_rate(normalized_reference, normalized_prediction)


def calculate_wer(reference: str, prediction: str) -> ErrorRate:
    """Calculate word error rate after benchmark-only normalization."""

    reference_words = normalize_metric_text(reference).split()
    prediction_words = normalize_metric_text(prediction).split()
    return _error_rate(reference_words, prediction_words)

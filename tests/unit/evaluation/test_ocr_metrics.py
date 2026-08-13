"""Tests for deterministic OCR character and word error rates."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from app.evaluation.ocr_benchmark.metrics import calculate_cer, calculate_wer
from app.evaluation.ocr_benchmark.models import ErrorRate


@pytest.mark.parametrize(
    ("reference", "prediction", "errors", "units", "rate"),
    [
        ("cat", "cat", 0, 3, 0.0),
        ("cat", "coat", 1, 3, 1 / 3),
        ("coat", "cat", 1, 4, 0.25),
        ("cat", "cut", 1, 3, 1 / 3),
    ],
)
def test_cer_edit_operations(
    reference: str,
    prediction: str,
    errors: int,
    units: int,
    rate: float,
) -> None:
    result = calculate_cer(reference, prediction)

    assert result.errors == errors
    assert result.reference_units == units
    assert result.rate == pytest.approx(rate)


@pytest.mark.parametrize(
    ("reference", "prediction", "errors", "units", "rate"),
    [
        ("one two", "one two", 0, 2, 0.0),
        ("one two", "one new two", 1, 2, 0.5),
        ("one new two", "one two", 1, 3, 1 / 3),
        ("one two", "one too", 1, 2, 0.5),
    ],
)
def test_wer_edit_operations(
    reference: str,
    prediction: str,
    errors: int,
    units: int,
    rate: float,
) -> None:
    result = calculate_wer(reference, prediction)

    assert result.errors == errors
    assert result.reference_units == units
    assert result.rate == pytest.approx(rate)


@pytest.mark.parametrize("metric", [calculate_cer, calculate_wer])
def test_both_empty_has_zero_error(
    metric: Callable[[str, str], ErrorRate],
) -> None:
    result = metric("", "")

    assert result.errors == 0
    assert result.reference_units == 0
    assert result.rate == 0.0


@pytest.mark.parametrize("metric", [calculate_cer, calculate_wer])
def test_empty_reference_with_prediction_has_undefined_rate(
    metric: Callable[[str, str], ErrorRate],
) -> None:
    result = metric("", "unexpected")

    assert result.errors > 0
    assert result.reference_units == 0
    assert result.rate is None


def test_empty_prediction_counts_reference_deletions() -> None:
    cer = calculate_cer("two words", "")
    wer = calculate_wer("two words", "")

    assert cer.errors == cer.reference_units
    assert cer.rate == 1.0
    assert wer.errors == 2
    assert wer.rate == 1.0


def test_unicode_is_compared_after_nfc_normalization() -> None:
    assert calculate_cer("Café", "Cafe\u0301").rate == 0.0


def test_whitespace_is_collapsed_but_case_and_punctuation_are_significant() -> None:
    assert calculate_wer("One,\n  two", "One, two").rate == 0.0
    assert calculate_wer("One, two", "One two").errors == 1
    assert calculate_wer("One two", "one two").errors == 1
    assert calculate_cer("answer.", "answer").errors == 1

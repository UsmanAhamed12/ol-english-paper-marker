"""Tests for conservative OCR text normalization."""

from __future__ import annotations

from app.ocr.normalizer import OCRNormalizer


def test_normalizes_windows_and_legacy_newlines() -> None:
    raw_text = "first\r\nsecond\rthird"

    assert OCRNormalizer().normalize(raw_text) == "first\nsecond\nthird"


def test_normalizes_unicode_to_nfc() -> None:
    raw_text = "Cafe\u0301"

    assert OCRNormalizer().normalize(raw_text) == "Café"


def test_removes_trailing_horizontal_space_and_surrounding_blank_lines() -> None:
    raw_text = "\nline one  \t\n  line two\t\n\n"

    assert OCRNormalizer().normalize(raw_text) == "line one\n  line two"


def test_does_not_correct_spelling_or_grammar() -> None:
    raw_text = "He go to scool yesterday"

    assert OCRNormalizer().normalize(raw_text) == raw_text

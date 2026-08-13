"""Validation tests for OCR extraction and page result models."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.ocr.models import OCRExtraction, OCRPageResult, OCRWarningCode


def _result_data() -> dict[str, object]:
    return {
        "paper_id": uuid4(),
        "page_number": 1,
        "source_image_path": Path("/tmp/page_0001.png"),
        "raw_text": "He go to school yesterday",
        "normalized_text": "He go to school yesterday",
        "confidence": 0.75,
        "provider": "fake-provider",
        "model_version": "test-v1",
        "warnings": (OCRWarningCode.HANDWRITING_AMBIGUITY,),
        "processing_duration_ms": 12.5,
    }


def test_ocr_result_accepts_valid_data_and_preserves_provenance() -> None:
    result = OCRPageResult.model_validate(_result_data())

    assert result.provider == "fake-provider"
    assert result.model_version == "test-v1"
    assert result.raw_text == "He go to school yesterday"
    assert result.warnings == (OCRWarningCode.HANDWRITING_AMBIGUITY,)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_confidence_outside_unit_interval_is_rejected(confidence: float) -> None:
    with pytest.raises(ValidationError):
        OCRExtraction(
            raw_text="text",
            confidence=confidence,
            processing_duration_ms=1,
        )


def test_unavailable_confidence_is_explicit() -> None:
    extraction = OCRExtraction(
        raw_text="text",
        confidence=None,
        processing_duration_ms=1,
    )

    assert extraction.confidence is None


def test_non_positive_page_number_is_rejected() -> None:
    data = _result_data()
    data["page_number"] = 0

    with pytest.raises(ValidationError):
        OCRPageResult.model_validate(data)


def test_relative_source_image_reference_is_rejected() -> None:
    data = _result_data()
    data["source_image_path"] = Path("page_0001.png")

    with pytest.raises(ValidationError, match="absolute"):
        OCRPageResult.model_validate(data)

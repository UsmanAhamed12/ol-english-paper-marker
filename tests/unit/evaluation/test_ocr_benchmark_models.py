"""Tests for OCR benchmark manifest and result validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation.ocr_benchmark.models import (
    BenchmarkDifficulty,
    BenchmarkManifest,
    BenchmarkRegion,
    BenchmarkStatus,
    ErrorRate,
    GroundTruthStatus,
    OCRBenchmarkResult,
    OCRBenchmarkSample,
)


def _sample_data() -> dict[str, object]:
    return {
        "sample_id": "synthetic-sample-01",
        "paper_alias": "synthetic-paper-a",
        "page_number": 1,
        "image_path": Path("synthetic/page.png"),
        "image_width": 1000,
        "image_height": 1400,
        "difficulty": BenchmarkDifficulty.MEDIUM,
        "categories": ("short-answer",),
        "printed_content_present": True,
        "teacher_annotations_present": False,
        "ground_truth_status": GroundTruthStatus.VERIFIED,
        "ground_truth_student_text": "Synthetic student response.",
    }


def test_safe_sample_and_optional_region_are_accepted() -> None:
    data = _sample_data()
    data["region"] = BenchmarkRegion(x=20, y=30, width=500, height=600)

    sample = OCRBenchmarkSample.model_validate(data)

    assert sample.sample_id == "synthetic-sample-01"
    assert sample.transcription_target == "student_answer_text"
    assert sample.region is not None
    assert sample.is_ready


@pytest.mark.parametrize("sample_id", ["Student Name", "../private", "ab"])
def test_unsafe_sample_identifier_is_rejected(sample_id: str) -> None:
    data = _sample_data()
    data["sample_id"] = sample_id

    with pytest.raises(ValidationError):
        OCRBenchmarkSample.model_validate(data)


def test_region_outside_page_is_rejected() -> None:
    data = _sample_data()
    data["region"] = BenchmarkRegion(x=900, y=0, width=101, height=100)

    with pytest.raises(ValidationError, match="fit within"):
        OCRBenchmarkSample.model_validate(data)


def test_verified_sample_requires_manual_transcription() -> None:
    data = _sample_data()
    data["ground_truth_student_text"] = None

    with pytest.raises(ValidationError, match="manual transcription"):
        OCRBenchmarkSample.model_validate(data)


def test_pending_sample_is_valid_but_not_ready() -> None:
    data = _sample_data()
    data["ground_truth_status"] = GroundTruthStatus.PENDING
    data["ground_truth_student_text"] = None

    sample = OCRBenchmarkSample.model_validate(data)

    assert not sample.is_ready


def test_pending_sample_rejects_unverified_transcription() -> None:
    data = _sample_data()
    data["ground_truth_status"] = GroundTruthStatus.PENDING

    with pytest.raises(ValidationError, match="must not contain"):
        OCRBenchmarkSample.model_validate(data)


def test_manifest_rejects_duplicate_sample_ids() -> None:
    sample = OCRBenchmarkSample.model_validate(_sample_data())

    with pytest.raises(ValidationError, match="unique"):
        BenchmarkManifest(samples=(sample, sample))


def test_benchmark_result_requires_consistent_success_payload() -> None:
    metric = ErrorRate(errors=0, reference_units=4, rate=0.0)
    result = OCRBenchmarkResult(
        sample_id="synthetic-sample-01",
        provider="fake-provider",
        model_version="fake-v1",
        ocr_prompt_version="prompt-v1",
        status=BenchmarkStatus.SUCCESS,
        prediction="text",
        cer=metric,
        wer=metric,
        duration_ms=10,
    )

    assert result.status is BenchmarkStatus.SUCCESS

    with pytest.raises(ValidationError, match="require metrics"):
        OCRBenchmarkResult(
            sample_id="synthetic-sample-01",
            provider="fake-provider",
            ocr_prompt_version="prompt-v1",
            status=BenchmarkStatus.SUCCESS,
        )


def test_failed_result_rejects_metrics() -> None:
    with pytest.raises(ValidationError, match="require an error"):
        OCRBenchmarkResult(
            sample_id="synthetic-sample-01",
            provider="fake-provider",
            ocr_prompt_version="prompt-v1",
            status=BenchmarkStatus.FAILURE,
            prediction="must not be scored",
            error="provider failed",
        )

"""Tests for provider-independent OCR benchmark execution and aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest

from app.domain.models.paper import PaperPage
from app.evaluation.ocr_benchmark.models import (
    BenchmarkDifficulty,
    BenchmarkStatus,
    ErrorRate,
    GroundTruthStatus,
    OCRBenchmarkResult,
    OCRBenchmarkSample,
)
from app.evaluation.ocr_benchmark.runner import OCRBenchmarkRunner
from app.ocr.models import OCRExtraction, OCRWarningCode


@dataclass
class FakeProvider:
    """Deterministic test provider with optional failure behavior."""

    extraction: OCRExtraction | None = None
    failure: Exception | None = None
    name: str = "fake-provider"
    model_version: str | None = "fake-v1"

    def extract_page(self, page: PaperPage) -> OCRExtraction:
        if self.failure is not None:
            raise self.failure
        if self.extraction is None:
            raise RuntimeError("test provider has no configured extraction")
        return self.extraction


def _sample(sample_id: str = "synthetic-sample-01") -> OCRBenchmarkSample:
    return OCRBenchmarkSample(
        sample_id=sample_id,
        paper_alias="synthetic-paper-a",
        page_number=1,
        image_path=Path("synthetic/page.png"),
        image_width=100,
        image_height=200,
        difficulty=BenchmarkDifficulty.CLEAR,
        categories=("short-answer",),
        printed_content_present=True,
        teacher_annotations_present=False,
        ground_truth_status=GroundTruthStatus.VERIFIED,
        ground_truth_student_text="student answer",
    )


def _page(tmp_path: Path) -> PaperPage:
    image_path = (tmp_path / "page.png").resolve()
    image_path.write_bytes(b"synthetic image placeholder")
    return PaperPage(
        paper_id=uuid4(),
        page_number=1,
        image_path=image_path,
        width=100,
        height=200,
    )


def _result(
    sample_id: str,
    cer: float,
    wer: float,
    duration: float,
) -> OCRBenchmarkResult:
    return OCRBenchmarkResult(
        sample_id=sample_id,
        provider="fake-provider",
        model_version="fake-v1",
        ocr_prompt_version="prompt-v1",
        status=BenchmarkStatus.SUCCESS,
        prediction="synthetic",
        cer=ErrorRate(errors=1, reference_units=10, rate=cer),
        wer=ErrorRate(errors=1, reference_units=2, rate=wer),
        duration_ms=duration,
    )


def test_runner_uses_ocr_service_and_records_metrics(tmp_path: Path) -> None:
    warning = OCRWarningCode.HANDWRITING_AMBIGUITY
    provider = FakeProvider(
        extraction=OCRExtraction(
            raw_text="student  answer\r\n",
            warnings=(warning,),
            processing_duration_ms=12,
        )
    )

    result = OCRBenchmarkRunner(
        provider,
        ocr_prompt_version="prompt-v1",
    ).run_sample(_sample(), _page(tmp_path))

    assert result.status is BenchmarkStatus.SUCCESS
    assert result.prediction == "student  answer\r\n"
    assert result.cer is not None and result.cer.rate == 0.0
    assert result.wer is not None and result.wer.rate == 0.0
    assert result.duration_ms == 12
    assert result.warnings == (warning,)
    assert result.ocr_prompt_version == "prompt-v1"


def test_provider_failure_is_countable_result(tmp_path: Path) -> None:
    runner = OCRBenchmarkRunner(
        FakeProvider(failure=RuntimeError("engine unavailable")),
        ocr_prompt_version="prompt-v1",
    )

    result = runner.run_sample(_sample(), _page(tmp_path))

    assert result.status is BenchmarkStatus.FAILURE
    assert result.error == "OCR provider failed for page 1"
    assert result.prediction is None
    assert result.cer is None


def test_runner_rejects_pending_ground_truth(tmp_path: Path) -> None:
    sample = _sample().model_copy(
        update={
            "ground_truth_status": GroundTruthStatus.PENDING,
            "ground_truth_student_text": None,
        }
    )
    runner = OCRBenchmarkRunner(
        FakeProvider(),
        ocr_prompt_version="prompt-v1",
    )

    with pytest.raises(ValueError, match="human-verified"):
        runner.run_sample(sample, _page(tmp_path))


def test_summary_reports_mean_median_duration_and_counts() -> None:
    results = (
        _result("synthetic-sample-01", 0.1, 0.2, 10),
        _result("synthetic-sample-02", 0.3, 0.6, 30),
        _result("synthetic-sample-03", 0.8, 1.0, 50),
        OCRBenchmarkResult(
            sample_id="synthetic-sample-04",
            provider="fake-provider",
            model_version="fake-v1",
            ocr_prompt_version="prompt-v1",
            status=BenchmarkStatus.FAILURE,
            error="provider failed",
        ),
    )

    summary = OCRBenchmarkRunner.summarize(results)

    assert summary.total_samples == 4
    assert summary.successful_samples == 3
    assert summary.failed_samples == 1
    assert summary.mean_cer == pytest.approx(0.4)
    assert summary.median_cer == pytest.approx(0.3)
    assert summary.mean_wer == pytest.approx(0.6)
    assert summary.median_wer == pytest.approx(0.6)
    assert summary.mean_processing_duration_ms == pytest.approx(30)

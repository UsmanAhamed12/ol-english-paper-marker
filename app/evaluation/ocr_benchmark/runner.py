"""Provider-independent execution and aggregation for OCR benchmarks."""

from __future__ import annotations

from statistics import fmean, median

from app.core.exceptions import OCRProcessingError
from app.domain.models.paper import PaperPage
from app.evaluation.ocr_benchmark.metrics import calculate_cer, calculate_wer
from app.evaluation.ocr_benchmark.models import (
    BenchmarkStatus,
    OCRBenchmarkResult,
    OCRBenchmarkSample,
    OCRBenchmarkSummary,
)
from app.ocr.base import OCRProvider
from app.ocr.normalizer import OCRNormalizer
from app.ocr.service import OCRService


class OCRBenchmarkRunner:
    """Score provider output against human student-answer transcription."""

    def __init__(
        self,
        provider: OCRProvider,
        *,
        ocr_prompt_version: str,
        normalizer: OCRNormalizer | None = None,
    ) -> None:
        self._provider = provider
        self._ocr_prompt_version = ocr_prompt_version
        self._service = OCRService(provider, normalizer or OCRNormalizer())

    def run_sample(
        self,
        sample: OCRBenchmarkSample,
        page: PaperPage,
    ) -> OCRBenchmarkResult:
        """Run one ready sample while converting provider failure to a result."""

        if not sample.is_ready or sample.ground_truth_student_text is None:
            raise ValueError("benchmark sample requires human-verified ground truth")
        if page.page_number != sample.page_number:
            raise ValueError("benchmark sample and page number do not match")

        try:
            ocr_result = self._service.process_page(page)
        except OCRProcessingError as error:
            return OCRBenchmarkResult(
                sample_id=sample.sample_id,
                provider=self._provider.name,
                model_version=self._provider.model_version,
                ocr_prompt_version=self._ocr_prompt_version,
                status=BenchmarkStatus.FAILURE,
                error=str(error),
            )

        return OCRBenchmarkResult(
            sample_id=sample.sample_id,
            provider=ocr_result.provider,
            model_version=ocr_result.model_version,
            ocr_prompt_version=self._ocr_prompt_version,
            status=BenchmarkStatus.SUCCESS,
            prediction=ocr_result.raw_text,
            cer=calculate_cer(
                sample.ground_truth_student_text,
                ocr_result.raw_text,
            ),
            wer=calculate_wer(
                sample.ground_truth_student_text,
                ocr_result.raw_text,
            ),
            duration_ms=ocr_result.processing_duration_ms,
            warnings=ocr_result.warnings,
        )

    @staticmethod
    def summarize(
        results: tuple[OCRBenchmarkResult, ...],
    ) -> OCRBenchmarkSummary:
        """Aggregate defined rates while retaining all failure counts."""

        successes = [
            result for result in results if result.status is BenchmarkStatus.SUCCESS
        ]
        cer_rates = [
            result.cer.rate
            for result in successes
            if result.cer is not None and result.cer.rate is not None
        ]
        wer_rates = [
            result.wer.rate
            for result in successes
            if result.wer is not None and result.wer.rate is not None
        ]
        durations = [
            result.duration_ms for result in successes if result.duration_ms is not None
        ]
        return OCRBenchmarkSummary(
            total_samples=len(results),
            successful_samples=len(successes),
            failed_samples=len(results) - len(successes),
            mean_cer=fmean(cer_rates) if cer_rates else None,
            median_cer=median(cer_rates) if cer_rates else None,
            mean_wer=fmean(wer_rates) if wer_rates else None,
            median_wer=median(wer_rates) if wer_rates else None,
            mean_processing_duration_ms=fmean(durations) if durations else None,
        )

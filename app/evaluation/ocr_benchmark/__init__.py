"""Reproducible OCR benchmark models, metrics, and runner."""

from app.evaluation.ocr_benchmark.metrics import calculate_cer, calculate_wer
from app.evaluation.ocr_benchmark.models import (
    BenchmarkDifficulty,
    BenchmarkManifest,
    BenchmarkRegion,
    GroundTruthStatus,
    OCRBenchmarkResult,
    OCRBenchmarkSample,
    OCRBenchmarkSummary,
)
from app.evaluation.ocr_benchmark.runner import OCRBenchmarkRunner

__all__ = [
    "BenchmarkDifficulty",
    "BenchmarkManifest",
    "BenchmarkRegion",
    "GroundTruthStatus",
    "OCRBenchmarkResult",
    "OCRBenchmarkRunner",
    "OCRBenchmarkSample",
    "OCRBenchmarkSummary",
    "calculate_cer",
    "calculate_wer",
]

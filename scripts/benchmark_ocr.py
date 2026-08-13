"""Validate or run the private OCR benchmark without exposing ground truth."""

from __future__ import annotations

import argparse
import contextlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pymupdf

from app.core.config import Settings
from app.core.exceptions import ApplicationError, ConfigurationError
from app.domain.models.paper import PaperPage
from app.evaluation.ocr_benchmark.ground_truth import ground_truth_fingerprint
from app.evaluation.ocr_benchmark.manifest import load_manifest
from app.evaluation.ocr_benchmark.models import (
    BenchmarkManifest,
    BenchmarkStatus,
    GroundTruthStatus,
    OCRBenchmarkResult,
    OCRBenchmarkSummary,
)
from app.evaluation.ocr_benchmark.preparation import safe_sample_filename
from app.evaluation.ocr_benchmark.runner import OCRBenchmarkRunner
from app.ocr.base import OCRProvider
from app.ocr.preprocessing.models import PreprocessingVariant
from app.ocr.preprocessing.opencv import OpenCVPreprocessor
from app.ocr.preprocessing.provider import PreprocessedOCRProvider
from app.ocr.prompts import OCR_TRANSCRIPTION_PROMPT_VERSION
from app.ocr.providers.ollama import (
    OllamaClientError,
    OllamaHTTPClient,
    OllamaOCRProvider,
)
from app.ocr.providers.tesseract import TesseractOCRProvider

FROZEN_BENCHMARK_FINGERPRINT = (
    "33a5dc8e46a1cf0631d46da41a8490c4ec10a18194591144425422c61ff73f9a"
)
FROZEN_SAMPLE_COUNT = 8
DEFAULT_MANIFEST = Path("data/evaluation/ocr/benchmark_manifest.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse validation and local benchmark commands."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate private ground truth")
    validate.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)

    run = subparsers.add_parser("run", help="run a sequential local OCR benchmark")
    run.add_argument("--provider", choices=("ollama", "tesseract"), required=True)
    run.add_argument("--model")
    run.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    run.add_argument("--samples-dir", type=Path)
    run.add_argument("--results-dir", type=Path)
    run.add_argument("--smoke", action="store_true")
    run.add_argument(
        "--preprocessing",
        choices=tuple(variant.value for variant in PreprocessingVariant),
        default=PreprocessingVariant.NONE.value,
        help="fixed Tesseract preprocessing variant (default: none)",
    )
    run.add_argument(
        "--expected-fingerprint",
        default=FROZEN_BENCHMARK_FINGERPRINT,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def _validation_payload(manifest: BenchmarkManifest) -> dict[str, object]:
    human_verified_count = sum(
        sample.ground_truth_status is GroundTruthStatus.VERIFIED
        for sample in manifest.samples
    )
    human_verified_empty_count = sum(
        sample.ground_truth_status is GroundTruthStatus.VERIFIED_EMPTY
        for sample in manifest.samples
    )
    pending_count = sum(not sample.is_ready for sample in manifest.samples)
    return {
        "schema_version": manifest.schema_version,
        "sample_count": len(manifest.samples),
        "human_verified_count": human_verified_count,
        "human_verified_empty_count": human_verified_empty_count,
        "pending_count": pending_count,
        "benchmark_ready": manifest.is_ready,
        "ground_truth_fingerprint": (
            ground_truth_fingerprint(manifest) if manifest.is_ready else None
        ),
        "ocr_executed": False,
    }


def _require_frozen_benchmark(
    manifest: BenchmarkManifest,
    expected_fingerprint: str,
) -> str:
    if not manifest.is_ready or len(manifest.samples) != FROZEN_SAMPLE_COUNT:
        raise ConfigurationError("Frozen OCR benchmark is not ready")
    fingerprint = ground_truth_fingerprint(manifest)
    if fingerprint != expected_fingerprint:
        raise ConfigurationError("Frozen OCR ground-truth fingerprint does not match")
    return fingerprint


def _private_directory(
    requested: Path | None,
    *,
    default: Path,
    evaluation_root: Path,
) -> Path:
    path = (requested or default).resolve()
    if not path.is_relative_to(evaluation_root.resolve()):
        raise ConfigurationError("Benchmark artifacts must remain in private storage")
    return path


def _paper_page(
    sample_index: int, manifest: BenchmarkManifest, samples_dir: Path
) -> PaperPage:
    sample = manifest.samples[sample_index]
    image_path = (samples_dir / safe_sample_filename(sample_index + 1)).resolve()
    if not image_path.is_file():
        raise ConfigurationError("Prepared private benchmark image is unavailable")
    try:
        pixmap = pymupdf.Pixmap(str(image_path))  # type: ignore[no-untyped-call]
    except (OSError, RuntimeError, ValueError) as error:
        raise ConfigurationError("Prepared benchmark image is invalid") from error
    return PaperPage(
        paper_id=uuid5(NAMESPACE_URL, f"private-ocr-benchmark/{sample.sample_id}"),
        page_number=sample.page_number,
        image_path=image_path,
        width=pixmap.width,
        height=pixmap.height,
    )


def _safe_model_directory(model: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", model).strip("-.").lower()
    if not value:
        raise ConfigurationError("Ollama OCR model must have a safe identifier")
    return value


def _category_summaries(
    manifest: BenchmarkManifest,
    results: tuple[OCRBenchmarkResult, ...],
) -> dict[str, dict[str, object]]:
    by_category: dict[str, list[OCRBenchmarkResult]] = defaultdict(list)
    result_by_id = {result.sample_id: result for result in results}
    for sample in manifest.samples:
        result = result_by_id.get(sample.sample_id)
        if result is None:
            continue
        by_category[f"difficulty:{sample.difficulty.value}"].append(result)
        for category in sample.categories:
            by_category[f"category:{category}"].append(result)
        if sample.teacher_annotations_present:
            by_category["teacher-annotation-risk"].append(result)
    return {
        name: OCRBenchmarkRunner.summarize(tuple(category_results)).model_dump(
            mode="json"
        )
        for name, category_results in sorted(by_category.items())
    }


def _verified_empty_behavior(
    manifest: BenchmarkManifest,
    results: tuple[OCRBenchmarkResult, ...],
) -> dict[str, int]:
    result_by_id = {result.sample_id: result for result in results}
    verified_empty = [
        sample
        for sample in manifest.samples
        if sample.ground_truth_status is GroundTruthStatus.VERIFIED_EMPTY
        and sample.sample_id in result_by_id
    ]
    successful_empty = 0
    hallucinated = 0
    failed = 0
    for sample in verified_empty:
        result = result_by_id.get(sample.sample_id)
        if result is None or result.status is BenchmarkStatus.FAILURE:
            failed += 1
        elif result.prediction == "":
            successful_empty += 1
        else:
            hallucinated += 1
    return {
        "sample_count": len(verified_empty),
        "empty_predictions": successful_empty,
        "nonempty_predictions": hallucinated,
        "failed": failed,
    }


def _write_private_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_ollama(args: argparse.Namespace) -> int:
    manifest_path: Path = args.manifest.resolve()
    if args.preprocessing != PreprocessingVariant.NONE.value:
        raise ConfigurationError("Preprocessing is supported only by Tesseract")
    manifest = load_manifest(manifest_path)
    fingerprint = _require_frozen_benchmark(manifest, args.expected_fingerprint)
    evaluation_root = manifest_path.parent
    samples_dir = _private_directory(
        args.samples_dir,
        default=evaluation_root / "samples",
        evaluation_root=evaluation_root,
    )
    results_dir = _private_directory(
        args.results_dir,
        default=evaluation_root / "results",
        evaluation_root=evaluation_root,
    )

    settings = Settings()
    if not args.model:
        raise ConfigurationError("Ollama benchmark requires --model")
    client = OllamaHTTPClient(
        base_url=str(settings.ollama_base_url),
        timeout_seconds=settings.ollama_ocr_timeout_seconds,
    )
    ollama_version = client.version()
    model_info = client.model_info(args.model)
    provider = OllamaOCRProvider(
        client=client,
        model=model_info.model,
        model_version=model_info.digest,
    )
    runner = OCRBenchmarkRunner(
        provider,
        ocr_prompt_version=OCR_TRANSCRIPTION_PROMPT_VERSION,
    )

    selected_indexes = range(1) if args.smoke else range(len(manifest.samples))
    try:
        results = tuple(
            runner.run_sample(
                manifest.samples[index],
                _paper_page(index, manifest, samples_dir),
            )
            for index in selected_indexes
        )
    finally:
        with contextlib.suppress(ApplicationError):
            client.set_timeout(30.0)
            client.unload(model_info.model)

    summary: OCRBenchmarkSummary = runner.summarize(results)
    output_path = (
        results_dir
        / _safe_model_directory(model_info.model)
        / ("smoke.json" if args.smoke else "benchmark.json")
    )
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "run_type": "smoke" if args.smoke else "full",
        "provider": provider.name,
        "model": model_info.model,
        "model_digest": model_info.digest,
        "ollama_version": ollama_version,
        "ocr_prompt_version": OCR_TRANSCRIPTION_PROMPT_VERSION,
        "ground_truth_fingerprint": fingerprint,
        "generation_options": {
            "temperature": 0.0,
            "seed": 0,
            "num_predict": 2048,
            "think": False,
            "keep_alive": "5m_during_sequential_run_then_unload",
        },
        "results": [result.model_dump(mode="json") for result in results],
        "summary": summary.model_dump(mode="json"),
        "verified_empty_behavior": _verified_empty_behavior(manifest, results),
        "teacher_contamination_assessment": "pending_manual_review",
    }
    if not args.smoke:
        payload["category_summaries"] = _category_summaries(manifest, results)
    _write_private_results(output_path, payload)

    safe_report = {
        "run_type": payload["run_type"],
        "provider": provider.name,
        "model": model_info.model,
        "model_digest": model_info.digest,
        "ollama_version": ollama_version,
        "ocr_prompt_version": OCR_TRANSCRIPTION_PROMPT_VERSION,
        "ground_truth_fingerprint": fingerprint,
        "result_count": len(results),
        "summary": summary.model_dump(mode="json"),
        "verified_empty_behavior": payload["verified_empty_behavior"],
        "private_results_path": str(output_path),
    }
    print(json.dumps(safe_report, sort_keys=True))
    if args.smoke and summary.failed_samples:
        return 1
    return 0


def _run_tesseract(args: argparse.Namespace) -> int:
    manifest_path: Path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    fingerprint = _require_frozen_benchmark(manifest, args.expected_fingerprint)
    evaluation_root = manifest_path.parent
    samples_dir = _private_directory(
        args.samples_dir,
        default=evaluation_root / "samples",
        evaluation_root=evaluation_root,
    )
    results_dir = _private_directory(
        args.results_dir,
        default=evaluation_root / "results",
        evaluation_root=evaluation_root,
    )
    settings = Settings()
    base_provider = TesseractOCRProvider.from_system(
        language=settings.tesseract_language,
        psm=settings.tesseract_psm,
        timeout_seconds=settings.tesseract_timeout_seconds,
    )
    variant = PreprocessingVariant(args.preprocessing)
    provider: OCRProvider = base_provider
    if variant is not PreprocessingVariant.NONE:
        preprocessed_root = _private_directory(
            None,
            default=evaluation_root / "preprocessed" / variant.value,
            evaluation_root=evaluation_root,
        )
        provider = PreprocessedOCRProvider(
            provider=base_provider,
            preprocessor=OpenCVPreprocessor(variant),
            output_root=preprocessed_root,
        )
    experiment_name = tesseract_result_name(variant)
    runner = OCRBenchmarkRunner(
        provider,
        ocr_prompt_version=f"{experiment_name}-v1",
    )
    selected_indexes = range(1) if args.smoke else range(len(manifest.samples))
    results = tuple(
        runner.run_sample(
            manifest.samples[index],
            _paper_page(index, manifest, samples_dir),
        )
        for index in selected_indexes
    )
    summary = runner.summarize(results)
    output_path = (
        results_dir
        / experiment_name
        / ("smoke.json" if args.smoke else "benchmark.json")
    )
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "run_type": "smoke" if args.smoke else "full",
        "provider": provider.name,
        "model": experiment_name,
        "model_version": provider.model_version,
        "ocr_prompt_version": f"{experiment_name}-v1",
        "ground_truth_fingerprint": fingerprint,
        "configuration": {
            "language": settings.tesseract_language,
            "psm": settings.tesseract_psm,
            "timeout_seconds": settings.tesseract_timeout_seconds,
            "preprocessing": variant.value,
            "operations": [operation.value for operation in variant.operations],
        },
        "results": [result.model_dump(mode="json") for result in results],
        "summary": summary.model_dump(mode="json"),
        "verified_empty_behavior": _verified_empty_behavior(manifest, results),
        "teacher_contamination_assessment": "pending_manual_review",
    }
    if not args.smoke:
        payload["category_summaries"] = _category_summaries(manifest, results)
    _write_private_results(output_path, payload)
    safe_report = {
        "run_type": payload["run_type"],
        "provider": provider.name,
        "model": experiment_name,
        "model_version": provider.model_version,
        "configuration": payload["configuration"],
        "ground_truth_fingerprint": fingerprint,
        "result_count": len(results),
        "summary": summary.model_dump(mode="json"),
        "verified_empty_behavior": payload["verified_empty_behavior"],
        "private_results_path": str(output_path),
    }
    print(json.dumps(safe_report, sort_keys=True))
    return int(args.smoke and summary.failed_samples > 0)


def tesseract_result_name(variant: PreprocessingVariant) -> str:
    """Return a deterministic, private result-directory name."""

    if variant is PreprocessingVariant.NONE:
        return "tesseract-baseline"
    return f"tesseract-{variant.value}"


def main(argv: list[str] | None = None) -> int:
    """Execute one safe benchmark command."""

    try:
        args = parse_args(argv)
        if args.command == "validate":
            manifest = load_manifest(args.manifest)
            print(json.dumps(_validation_payload(manifest), sort_keys=True))
            return 0
        if args.provider == "tesseract":
            return _run_tesseract(args)
        return _run_ollama(args)
    except (ConfigurationError, OllamaClientError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, sort_keys=True))
        return 2
    except (ApplicationError, OSError, ValueError):
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": "Private OCR benchmark validation or execution failed",
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

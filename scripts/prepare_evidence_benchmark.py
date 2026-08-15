"""Prepare private evidence crops, predictions, overlays, and labeling worksheet."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import cv2

from app.core.config import Settings
from app.domain.models.paper import PaperPage
from app.evaluation.evidence_benchmark.manifest import load_evidence_manifest
from app.evaluation.evidence_benchmark.models import (
    EvidenceBenchmarkManifest,
    EvidenceBenchmarkSample,
)
from app.evaluation.evidence_benchmark.preparation import prepare_evidence_benchmark
from app.evidence.answer_regions import AnswerRegionDetector
from app.evidence.overlay import render_evidence_overlay
from app.evidence.separator import EvidenceSeparator
from app.evidence.service import EvidenceSeparationService
from app.ocr.models import OCRPageResult
from app.ocr.normalizer import OCRNormalizer
from app.ocr.providers.tesseract import TesseractOCRProvider
from app.ocr.service import OCRService

DEFAULT_MANIFEST = Path("data/evaluation/evidence/benchmark_manifest.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "prepare"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def validation_summary(manifest: EvidenceBenchmarkManifest) -> dict[str, Any]:
    """Return safe readiness metadata without paths, labels, or page content."""

    categories = Counter(
        category.value for sample in manifest.samples for category in sample.categories
    )
    return {
        "sample_count": len(manifest.samples),
        "pending_count": manifest.pending_count,
        "human_verified_count": len(manifest.samples) - manifest.pending_count,
        "benchmark_ready": manifest.benchmark_ready,
        "category_counts": dict(sorted(categories.items())),
    }


def prepare_private_artifacts(
    manifest_path: Path,
    *,
    settings: Settings | None = None,
    artifact_namespace: str | None = None,
) -> dict[str, Any]:
    """Run local deterministic evidence analysis without evaluating ground truth."""

    resolved_manifest = manifest_path.resolve(strict=True)
    if (
        artifact_namespace is not None
        and re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,31}", artifact_namespace) is None
    ):
        raise ValueError("Private artifact namespace is invalid")
    manifest = load_evidence_manifest(resolved_manifest)
    root = resolved_manifest.parent
    sample_paths, worksheet = prepare_evidence_benchmark(manifest, root)
    settings = settings or Settings()
    ocr_service = OCRService(
        TesseractOCRProvider.from_system(
            language=settings.tesseract_language,
            psm=settings.tesseract_psm,
            timeout_seconds=settings.tesseract_timeout_seconds,
        ),
        OCRNormalizer(),
    )
    evidence_service = EvidenceSeparationService(
        EvidenceSeparator(), AnswerRegionDetector()
    )
    page_cache: dict[tuple[str, int], tuple[PaperPage, OCRPageResult]] = {}
    records: list[dict[str, Any]] = []
    for sample in manifest.samples:
        cache_key = (sample.paper_alias, sample.page_number)
        cached = page_cache.get(cache_key)
        if cached is None:
            page = _sample_page(sample)
            ocr_result = ocr_service.process_page(page)
            page_cache[cache_key] = (page, ocr_result)
        else:
            page, ocr_result = cached
        analyzed = evidence_service.analyze_region(
            page,
            ocr_result,
            test_number=sample.test_number,
            region_bbox=sample.region,
        )
        overlay_root = root / "overlays"
        if artifact_namespace is not None:
            overlay_root = overlay_root / artifact_namespace
        overlay_path = overlay_root / f"{sample.sample_id}.png"
        render_evidence_overlay(
            page,
            analyzed,
            overlay_path,
            crop_bbox=sample.region,
        )
        records.append(
            {
                "sample_id": sample.sample_id,
                "classification_counts": dict(
                    Counter(
                        region.evidence_type.value
                        for region in analyzed.evidence_regions
                    )
                ),
                "answer_region_count": len(analyzed.answer_regions),
                "evidence": analyzed.model_dump(mode="json"),
            }
        )
    result_root = root / "results"
    if artifact_namespace is not None:
        result_root = result_root / artifact_namespace
    result_path = result_root / "candidate_predictions.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        result_path,
        {
            "schema_version": "1.0",
            "benchmark_evaluated": False,
            "reason": "human_labels_pending",
            "samples": records,
        },
    )
    summary = validation_summary(manifest)
    summary.update(
        {
            "sample_images": len(sample_paths),
            "overlay_images": len(records),
            "worksheet": str(worksheet.relative_to(Path.cwd())),
            "private_predictions": str(result_path.relative_to(Path.cwd())),
            "benchmark_evaluated": False,
        }
    )
    return summary


def _sample_page(sample: EvidenceBenchmarkSample) -> PaperPage:
    path = sample.source_image_path.expanduser().resolve(strict=True)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim < 2:
        raise ValueError("Private evidence source page is invalid")
    height, width = image.shape[:2]
    if (width, height) != (sample.page_width, sample.page_height):
        raise ValueError("Private evidence source dimensions changed")
    return PaperPage(
        paper_id=uuid5(NAMESPACE_URL, f"ol-english-evidence/{sample.paper_alias}"),
        page_number=sample.page_number,
        image_path=path,
        width=width,
        height=height,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    manifest_path: Path = arguments.manifest
    if arguments.command == "validate":
        payload = validation_summary(load_evidence_manifest(manifest_path))
    else:
        payload = prepare_private_artifacts(manifest_path)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

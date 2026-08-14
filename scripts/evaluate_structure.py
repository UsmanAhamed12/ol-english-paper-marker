"""Validate or run the private exam-structure benchmark safely."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import cv2

from app.core.config import Settings
from app.domain.models.paper import PaperDocument, PaperPage
from app.evaluation.structure_benchmark.manifest import load_structure_manifest
from app.evaluation.structure_benchmark.models import (
    StructureBenchmarkManifest,
    StructureBenchmarkResult,
)
from app.evaluation.structure_benchmark.runner import (
    evaluate_structure,
    summarize_structure_results,
)
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.pdf_renderer import PDFRenderer
from app.ingestion.validators import PDFValidator
from app.ocr.normalizer import OCRNormalizer
from app.ocr.providers.tesseract import TesseractOCRProvider
from app.ocr.service import OCRService
from app.structure.overlay import render_structure_overlay
from app.structure.service import ExamStructureDetector

DEFAULT_MANIFEST = Path("data/evaluation/structure/manifest.json")


def build_parser() -> argparse.ArgumentParser:
    """Build the private benchmark command line without sensitive defaults."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("validate", "run"),
        help="Validate private metadata or run local Tesseract structure detection",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def run_benchmark(
    manifest_path: Path,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run local OCR and return only safe aggregate/public metadata."""

    resolved_manifest = manifest_path.resolve(strict=True)
    manifest = load_structure_manifest(resolved_manifest)
    evaluation_root = resolved_manifest.parent
    settings = settings or Settings()
    provider = TesseractOCRProvider.from_system(
        language=settings.tesseract_language,
        psm=settings.tesseract_psm,
        timeout_seconds=settings.tesseract_timeout_seconds,
    )
    ocr_service = OCRService(provider, OCRNormalizer())
    detector = ExamStructureDetector()
    results: list[StructureBenchmarkResult] = []

    private_records: list[dict[str, Any]] = []
    for benchmark_paper in manifest.papers:
        document = _load_or_render_document(
            benchmark_paper.source_path,
            benchmark_paper.paper_alias,
            evaluation_root,
            settings,
        )
        if document.page_count != benchmark_paper.expected_page_count:
            raise ValueError("Private structure paper page count changed")
        ocr_results = ocr_service.process_document(document)
        structure = detector.detect(document.pages, ocr_results)
        result = evaluate_structure(benchmark_paper, structure)
        results.append(result)
        overlay_root = evaluation_root / "overlays" / benchmark_paper.paper_alias
        for page, page_structure in zip(document.pages, structure.pages, strict=True):
            render_structure_overlay(
                page,
                page_structure,
                overlay_root / f"page_{page.page_number:04d}.png",
            )
        private_records.append(
            {
                "paper_alias": benchmark_paper.paper_alias,
                "result": result.model_dump(mode="json"),
                "structure": structure.model_dump(mode="json"),
            }
        )

    summary = summarize_structure_results(tuple(results))
    results_path = evaluation_root / "results" / "tesseract-structure.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        results_path,
        {
            "schema_version": "1.0",
            "provider": provider.name,
            "model_version": provider.model_version,
            "summary": summary.model_dump(mode="json"),
            "papers": private_records,
        },
    )
    return {
        "paper_count": summary.paper_count,
        "page_count": summary.page_count,
        "expected_markers": summary.expected_markers,
        "detected_markers": summary.detected_markers,
        "true_positives": summary.true_positives,
        "false_positives": summary.false_positives,
        "false_negatives": summary.false_negatives,
        "duplicate_markers": summary.duplicate_markers,
        "precision": summary.precision,
        "recall": summary.recall,
        "f1": summary.f1,
        "test_number_accuracy": summary.mean_test_number_accuracy,
        "ordering_accuracy": summary.mean_ordering_accuracy,
        "papers": [
            {
                "paper_alias": result.paper_alias,
                "pages": result.page_count,
                "expected": result.expected_markers,
                "detected": result.detected_markers,
                "false_positives": result.false_positives,
                "missing": list(result.missing_test_numbers),
                "ordering_valid": result.ordering_accuracy == 1.0,
            }
            for result in results
        ],
        "private_results_path": str(results_path.relative_to(Path.cwd())),
    }


def _load_or_render_document(
    source_path: Path,
    paper_alias: str,
    evaluation_root: Path,
    settings: Settings,
) -> PaperDocument:
    source = source_path.expanduser().resolve(strict=True)
    paper_id = _paper_id(paper_alias)
    validator = PDFValidator(
        max_file_size_bytes=settings.max_pdf_size_mb * 1024 * 1024,
        max_pages=settings.max_pdf_pages,
    )
    document = PDFLoader(
        validator,
        paper_id_factory=lambda: paper_id,
    ).load(source)
    pages_dir = evaluation_root / "runtime" / paper_id.hex / "pages"
    if not pages_dir.exists():
        return PDFRenderer(
            runtime_data_dir=evaluation_root / "runtime",
            render_dpi=settings.pdf_render_dpi,
        ).render(document)

    pages = tuple(
        _existing_page(pages_dir / f"page_{number:04d}.png", paper_id, number)
        for number in range(1, document.page_count + 1)
    )
    document_data = document.model_dump(exclude={"pages"})
    return PaperDocument.model_validate({**document_data, "pages": pages})


def _existing_page(path: Path, paper_id: UUID, page_number: int) -> PaperPage:
    resolved = path.resolve(strict=True)
    image = cv2.imread(str(resolved), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim < 2:
        raise ValueError("Private canonical structure page is invalid")
    height, width = image.shape[:2]
    return PaperPage(
        paper_id=paper_id,
        page_number=page_number,
        image_path=resolved,
        width=width,
        height=height,
    )


def _paper_id(paper_alias: str) -> UUID:
    return uuid5(NAMESPACE_URL, f"ol-english-structure/{paper_alias}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validation_summary(manifest: StructureBenchmarkManifest) -> dict[str, Any]:
    return {
        "paper_count": len(manifest.papers),
        "page_count": sum(paper.expected_page_count for paper in manifest.papers),
        "expected_markers": sum(
            len(paper.expected_markers) for paper in manifest.papers
        ),
        "human_verified": manifest.human_verified,
        "paper_aliases": [paper.paper_alias for paper in manifest.papers],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run a safe private structure benchmark command."""

    arguments = build_parser().parse_args(argv)
    manifest_path: Path = arguments.manifest
    if arguments.command == "validate":
        payload = _validation_summary(load_structure_manifest(manifest_path))
    else:
        payload = run_benchmark(manifest_path)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

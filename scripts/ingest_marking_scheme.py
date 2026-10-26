"""Ingest one local official marking-scheme PDF and print a safe summary."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.pdf_renderer import PDFRenderer
from app.ingestion.validators import PDFValidator
from app.marking_scheme.ingestion import MarkingSchemeIngestionService
from app.marking_scheme.parser import MarkingSchemeParser
from app.marking_scheme.serialization import write_private_json
from app.ocr.normalizer import OCRNormalizer
from app.ocr.providers.tesseract import TesseractOCRProvider
from app.ocr.service import OCRService


def build_parser() -> argparse.ArgumentParser:
    """Build the local-only development command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Private local marking-scheme PDF")
    return parser


def ingest_path(path: Path, *, settings: Settings | None = None) -> dict[str, Any]:
    """Run shared ingestion/OCR infrastructure and return safe counts only."""

    settings = settings or Settings()
    source = path.expanduser().resolve(strict=True)
    validator = PDFValidator(
        max_file_size_bytes=settings.max_pdf_size_mb * 1024 * 1024,
        max_pages=settings.max_pdf_pages,
    )
    document = PDFLoader(validator).load(source)
    private_root = (settings.runtime_data_dir / "marking-schemes").resolve()
    document = PDFRenderer(
        runtime_data_dir=private_root / "rendered",
        render_dpi=settings.pdf_render_dpi,
    ).render(document)
    provider = TesseractOCRProvider.from_system(
        language=settings.tesseract_language,
        psm=settings.tesseract_psm,
        timeout_seconds=settings.tesseract_timeout_seconds,
    )
    scheme = MarkingSchemeIngestionService(
        OCRService(provider, OCRNormalizer()),
        MarkingSchemeParser(),
    ).ingest(document)
    write_private_json(
        scheme,
        private_root / "results" / f"{scheme.scheme_id.hex}.json",
        private_root=private_root,
    )
    return {
        "pages": scheme.page_count,
        "tests_detected": len(scheme.tests),
        "test_numbers": [item.test_number for item in scheme.tests],
        "criteria_parsed": sum(len(item.criteria) for item in scheme.tests),
        "tests_with_known_maximum_marks": sum(
            item.maximum_marks is not None for item in scheme.tests
        ),
        "tests_with_unknown_maximum_marks": sum(
            item.maximum_marks is None for item in scheme.tests
        ),
        "warnings": len(scheme.warnings)
        + sum(len(item.warnings) for item in scheme.tests),
        "duration_ms": scheme.processing_duration_ms,
        "provider": provider.name,
        "model_version": provider.model_version,
        "private_output": "data/runtime/marking-schemes/results/<scheme-id>.json",
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the local ingestion command without dumping source text."""

    arguments = build_parser().parse_args(argv)
    print(json.dumps(ingest_path(arguments.path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

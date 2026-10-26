"""Extract the local Grade 10 syllabus and index it in persistent ChromaDB."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.ingestion.validators import PDFValidator
from app.syllabus.ingestion import SyllabusIngestionService
from app.vectorstore.chroma_store import ChromaKnowledgeBaseStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Syllabus PDF path (defaults to SYLLABUS_PDF_PATH)",
    )
    return parser


def ingest_path(
    path: Path | None = None,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or Settings()
    source_path = path or settings.syllabus_pdf_path
    validator = PDFValidator(
        max_file_size_bytes=settings.max_pdf_size_mb * 1024 * 1024,
        max_pages=settings.max_pdf_pages,
    )
    service = SyllabusIngestionService(
        ChromaKnowledgeBaseStore.from_settings(settings),
        validator,
    )
    report = service.ingest(source_path)
    return {
        **report,
        "collection": settings.chroma_collection_name,
        "persist_directory": str(settings.chroma_persist_dir),
        "embedding_model": settings.ollama_embedding_model,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    print(json.dumps(ingest_path(arguments.path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

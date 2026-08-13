"""Parser-backed validation for untrusted PDF inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.core.exceptions import (
    InvalidPDFError,
    PDFPageLimitError,
    PDFTooLargeError,
)


@dataclass(frozen=True, slots=True)
class ValidatedPDF:
    """Safe structural facts established by the PDF parser."""

    source_path: Path
    file_size_bytes: int
    page_count: int


class PDFValidator:
    """Validate a PDF before application metadata is constructed."""

    def __init__(self, *, max_file_size_bytes: int, max_pages: int) -> None:
        if max_file_size_bytes <= 0 or max_pages <= 0:
            raise ValueError("PDF validation limits must be positive")
        self._max_file_size_bytes = max_file_size_bytes
        self._max_pages = max_pages

    def validate(self, source_path: Path) -> ValidatedPDF:
        """Return parser-confirmed metadata or raise a safe ingestion error."""

        path = source_path.expanduser()
        if path.suffix.lower() != ".pdf":
            raise InvalidPDFError("Input must use the .pdf extension")
        if not path.exists():
            raise InvalidPDFError("PDF input does not exist")
        if not path.is_file():
            raise InvalidPDFError("PDF input is not a regular file")

        try:
            file_size_bytes = path.stat().st_size
        except OSError as error:
            raise InvalidPDFError("PDF metadata could not be read") from error
        if file_size_bytes == 0:
            raise InvalidPDFError("PDF input is empty")
        if file_size_bytes > self._max_file_size_bytes:
            raise PDFTooLargeError("PDF exceeds the configured file-size limit")

        resolved_path = path.resolve(strict=True)
        try:
            # PyMuPDF's public constructor has no complete type annotation.
            with pymupdf.open(str(resolved_path)) as document:  # type: ignore[no-untyped-call]
                if not document.is_pdf or document.needs_pass:
                    raise InvalidPDFError("Input is not a readable PDF")
                page_count = document.page_count
        except InvalidPDFError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise InvalidPDFError("Input is not a readable PDF") from error

        if page_count <= 0:
            raise InvalidPDFError("PDF must contain at least one page")
        if page_count > self._max_pages:
            raise PDFPageLimitError("PDF exceeds the configured page-count limit")

        return ValidatedPDF(
            source_path=resolved_path,
            file_size_bytes=file_size_bytes,
            page_count=page_count,
        )

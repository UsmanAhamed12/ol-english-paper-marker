"""Construct domain metadata from a validated PDF source."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from uuid import UUID, uuid4

from app.core.exceptions import InvalidPDFError
from app.domain.models.paper import PaperDocument
from app.ingestion.validators import PDFValidator

_HASH_BLOCK_SIZE = 1024 * 1024


class PDFLoader:
    """Validate a PDF and construct its identity-safe document metadata."""

    def __init__(
        self,
        validator: PDFValidator,
        *,
        paper_id_factory: Callable[[], UUID] = uuid4,
    ) -> None:
        self._validator = validator
        self._paper_id_factory = paper_id_factory

    def load(self, source_path: Path) -> PaperDocument:
        """Load metadata without rendering or extracting page content."""

        validated = self._validator.validate(source_path)
        try:
            sha256 = _sha256(validated.source_path)
        except OSError as error:
            raise InvalidPDFError("PDF content could not be read") from error

        confirmed = self._validator.validate(validated.source_path)
        if confirmed != validated:
            raise InvalidPDFError("PDF source changed during loading")

        return PaperDocument(
            paper_id=self._paper_id_factory(),
            source_path=validated.source_path,
            original_filename=validated.source_path.name,
            page_count=validated.page_count,
            file_size_bytes=validated.file_size_bytes,
            sha256=sha256,
        )


def _sha256(path: Path) -> str:
    """Hash a source file in bounded-memory blocks."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_HASH_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()

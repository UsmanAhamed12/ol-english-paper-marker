"""Provider contract for replaceable OCR implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain.models.paper import PaperPage
from app.ocr.models import OCRExtraction


@runtime_checkable
class OCRProvider(Protocol):
    """Extract raw OCR evidence from one canonical rendered page."""

    @property
    def name(self) -> str:
        """Return the stable provider identity used for provenance."""

        ...

    @property
    def model_version(self) -> str | None:
        """Return the provider model/version, or ``None`` when unavailable."""

        ...

    def extract_page(self, page: PaperPage) -> OCRExtraction:
        """Extract raw page text or raise on provider failure."""

        ...

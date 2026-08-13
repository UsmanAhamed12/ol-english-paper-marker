"""Provider-independent image preprocessing contract."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from app.ocr.preprocessing.models import PreprocessingResult


@runtime_checkable
class ImagePreprocessor(Protocol):
    """Create a derived OCR image without changing the source image."""

    @property
    def name(self) -> str:
        """Return the stable preprocessing variant name."""

        ...

    def process(self, source_path: Path, processed_path: Path) -> PreprocessingResult:
        """Produce one separately stored, geometry-preserving image."""

        ...

"""Composition adapter that preprocesses before invoking an OCR provider."""

from __future__ import annotations

from pathlib import Path

from app.core.exceptions import ImagePreprocessingError
from app.domain.models.paper import PaperPage
from app.ocr.base import OCRProvider
from app.ocr.models import OCRExtraction
from app.ocr.preprocessing.base import ImagePreprocessor


class PreprocessedOCRProvider:
    """Invoke an existing provider against a safe derived page image."""

    def __init__(
        self,
        *,
        provider: OCRProvider,
        preprocessor: ImagePreprocessor,
        output_root: Path,
    ) -> None:
        self._provider = provider
        self._preprocessor = preprocessor
        self._output_root = output_root.resolve()

    @property
    def name(self) -> str:
        """Preserve the underlying OCR provider identity."""

        return self._provider.name

    @property
    def model_version(self) -> str | None:
        """Preserve the underlying OCR engine version."""

        return self._provider.model_version

    def extract_page(self, page: PaperPage) -> OCRExtraction:
        """Preprocess separately, then delegate through the existing contract."""

        paper_root = self._output_root / page.paper_id.hex
        output_path = paper_root / f"page_{page.page_number:04d}.png"
        resolved_output = output_path.resolve()
        if not resolved_output.is_relative_to(self._output_root):
            raise ImagePreprocessingError("Unsafe preprocessing output path")
        preprocessing = self._preprocessor.process(page.image_path, resolved_output)
        derived_page = page.model_copy(
            update={"image_path": preprocessing.processed_image_path}
        )
        extraction = self._provider.extract_page(derived_page)
        return extraction.model_copy(update={"preprocessing": preprocessing})

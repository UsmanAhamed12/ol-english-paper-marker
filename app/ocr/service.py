"""Application boundary coordinating provider extraction and normalization."""

from __future__ import annotations

from pydantic import ValidationError

from app.core.exceptions import OCRProcessingError, OCRProviderError
from app.domain.models.paper import PaperDocument, PaperPage
from app.ocr.base import OCRProvider
from app.ocr.models import OCRExtraction, OCRPageResult
from app.ocr.normalizer import OCRNormalizer


class OCRService:
    """Run a replaceable provider without exposing it to downstream code."""

    def __init__(self, provider: OCRProvider, normalizer: OCRNormalizer) -> None:
        self._provider = provider
        self._normalizer = normalizer

    def process_page(self, page: PaperPage) -> OCRPageResult:
        """Extract and normalize one rendered page or fail explicitly."""

        if not page.image_path.is_file():
            raise OCRProcessingError("Rendered page image is unavailable")

        try:
            extraction = self._provider.extract_page(page)
        except OCRProcessingError:
            raise
        except Exception as error:
            raise OCRProviderError(
                f"OCR provider failed for page {page.page_number}"
            ) from error

        if not isinstance(extraction, OCRExtraction):
            raise OCRProviderError("OCR provider returned an invalid extraction")

        try:
            return OCRPageResult(
                paper_id=page.paper_id,
                page_number=page.page_number,
                source_image_path=page.image_path,
                raw_text=extraction.raw_text,
                normalized_text=self._normalizer.normalize(extraction.raw_text),
                confidence=extraction.confidence,
                provider=self._provider.name,
                model_version=self._provider.model_version,
                warnings=extraction.warnings,
                processing_duration_ms=extraction.processing_duration_ms,
                evidence=extraction.evidence,
                preprocessing=extraction.preprocessing,
            )
        except (ValidationError, ValueError) as error:
            raise OCRProviderError(
                "OCR provider returned invalid provenance"
            ) from error

    def process_document(self, document: PaperDocument) -> tuple[OCRPageResult, ...]:
        """Process every rendered page synchronously in canonical page order."""

        if not document.pages:
            raise OCRProcessingError("Paper document has no rendered pages")
        return tuple(self.process_page(page) for page in document.pages)

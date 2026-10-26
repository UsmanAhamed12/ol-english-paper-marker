"""Application boundary for OCR-backed marking-scheme ingestion."""

from __future__ import annotations

from decimal import Decimal
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

from app.core.exceptions import MarkingSchemeIngestionError, OCRProcessingError
from app.domain.models.paper import PaperDocument
from app.marking_scheme.models import (
    MarkingSchemeOCRStatus,
    MarkingSchemePageOCR,
    MarkingSchemeProvenance,
    MarkingSchemeWarning,
    StructuredMarkingScheme,
)
from app.marking_scheme.parser import (
    PARSER_VERSION,
    MarkingSchemeParser,
    SourceLine,
    lines_from_text,
    lines_from_words,
)
from app.ocr.service import OCRService

INGESTION_VERSION = "marking-scheme-ingestion-v1"


class MarkingSchemeIngestionService:
    """OCR and conservatively parse a rendered official marking scheme."""

    def __init__(self, ocr_service: OCRService, parser: MarkingSchemeParser) -> None:
        self._ocr_service = ocr_service
        self._parser = parser

    def ingest(self, document: PaperDocument) -> StructuredMarkingScheme:
        """Ingest pages independently while preserving explicit OCR failures."""

        if not document.pages:
            raise MarkingSchemeIngestionError("Marking scheme has no rendered pages")
        started = perf_counter()
        page_records: list[MarkingSchemePageOCR] = []
        source_lines: list[SourceLine] = []
        for page in document.pages:
            try:
                result = self._ocr_service.process_page(page)
            except OCRProcessingError:
                page_records.append(
                    MarkingSchemePageOCR(
                        page_number=page.page_number,
                        status=MarkingSchemeOCRStatus.OCR_FAILED,
                        provider=self._ocr_service.provider_name,
                        model_version=self._ocr_service.model_version,
                        duration_ms=0,
                        error_code="ocr_processing_failed",
                    )
                )
                continue
            words = result.evidence.words if result.evidence else ()
            page_records.append(
                MarkingSchemePageOCR(
                    page_number=page.page_number,
                    status=MarkingSchemeOCRStatus.EXTRACTED,
                    raw_text=result.raw_text,
                    normalized_text=result.normalized_text,
                    words=words,
                    provider=result.provider,
                    model_version=result.model_version,
                    duration_ms=result.processing_duration_ms,
                    warnings=result.warnings,
                )
            )
            source_lines.extend(
                lines_from_words(page.page_number, words)
                if words
                else lines_from_text(page.page_number, result.normalized_text)
            )
        try:
            tests = self._parser.parse(tuple(source_lines))
        except ValueError as error:
            raise MarkingSchemeIngestionError(
                "Marking scheme could not be structured safely"
            ) from error
        warnings: list[MarkingSchemeWarning] = []
        failed_pages = tuple(
            item.page_number
            for item in page_records
            if item.status is MarkingSchemeOCRStatus.OCR_FAILED
        )
        if failed_pages:
            warnings.append(MarkingSchemeWarning.OCR_PAGE_FAILED)
        maximums = tuple(item.maximum_marks for item in tests)
        total: Decimal | None = None
        if all(item is not None for item in maximums):
            total = sum((item for item in maximums if item is not None), Decimal(0))
        else:
            warnings.append(MarkingSchemeWarning.TOTAL_MAXIMUM_INCOMPLETE)
        return StructuredMarkingScheme(
            scheme_id=uuid5(NAMESPACE_URL, f"ol-english-scheme/{document.sha256}"),
            source_document_id=document.paper_id,
            page_count=document.page_count,
            tests=tests,
            total_available_marks=total,
            page_ocr=tuple(page_records),
            warnings=tuple(warnings),
            provenance=MarkingSchemeProvenance(
                ingestion_version=INGESTION_VERSION,
                parser_version=PARSER_VERSION,
                source_sha256=document.sha256,
                ocr_provider=self._ocr_service.provider_name,
                ocr_model_version=self._ocr_service.model_version,
                successful_ocr_pages=tuple(
                    item.page_number
                    for item in page_records
                    if item.status is MarkingSchemeOCRStatus.EXTRACTED
                ),
                failed_ocr_pages=failed_pages,
            ),
            processing_duration_ms=(perf_counter() - started) * 1000,
        )

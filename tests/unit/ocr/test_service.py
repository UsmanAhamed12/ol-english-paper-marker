"""Tests for OCR service orchestration with a deterministic fake provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app.core.exceptions import OCRProcessingError, OCRProviderError
from app.domain.models.paper import PaperDocument, PaperPage
from app.ocr.base import OCRProvider
from app.ocr.models import OCRExtraction, OCRWarningCode
from app.ocr.normalizer import OCRNormalizer
from app.ocr.service import OCRService


@dataclass
class FakeOCRProvider:
    """Deterministic provider used only to exercise the service contract."""

    extractions: dict[int, OCRExtraction]
    name: str = "fake-provider"
    model_version: str | None = "fake-v1"
    failure: Exception | None = None
    calls: list[int] = field(default_factory=list)

    def extract_page(self, page: PaperPage) -> OCRExtraction:
        self.calls.append(page.page_number)
        if self.failure is not None:
            raise self.failure
        return self.extractions[page.page_number]


def _page(
    tmp_path: Path,
    paper_id: UUID,
    page_number: int,
    *,
    create_image: bool = True,
) -> PaperPage:
    image_path = (tmp_path / f"page_{page_number:04d}.png").resolve()
    if create_image:
        image_path.write_bytes(b"canonical rendered page placeholder")
    return PaperPage(
        paper_id=paper_id,
        page_number=page_number,
        image_path=image_path,
        width=100,
        height=200,
    )


def _document(tmp_path: Path, pages: tuple[PaperPage, ...]) -> PaperDocument:
    source_path = (tmp_path / "source.pdf").resolve()
    source_path.write_bytes(b"source placeholder")
    return PaperDocument(
        paper_id=pages[0].paper_id,
        source_path=source_path,
        original_filename="source.pdf",
        page_count=len(pages),
        file_size_bytes=source_path.stat().st_size,
        sha256="a" * 64,
        pages=pages,
    )


def test_fake_provider_satisfies_protocol(tmp_path: Path) -> None:
    provider = FakeOCRProvider(
        {1: OCRExtraction(raw_text="text", processing_duration_ms=1)}
    )

    assert isinstance(provider, OCRProvider)


def test_service_invokes_provider_and_returns_normalized_provenance(
    tmp_path: Path,
) -> None:
    paper_id = uuid4()
    page = _page(tmp_path, paper_id, 1)
    raw_text = "\nCafe\u0301  \r\nHe go to scool yesterday  \n"
    warning = OCRWarningCode.HANDWRITING_AMBIGUITY
    extraction = OCRExtraction(
        raw_text=raw_text,
        confidence=0.42,
        warnings=(warning,),
        processing_duration_ms=15.5,
    )
    provider = FakeOCRProvider({1: extraction})

    result = OCRService(provider, OCRNormalizer()).process_page(page)

    assert provider.calls == [1]
    assert result.paper_id == paper_id
    assert result.page_number == 1
    assert result.source_image_path == page.image_path
    assert result.provider == "fake-provider"
    assert result.model_version == "fake-v1"
    assert result.raw_text == raw_text
    assert extraction.raw_text == raw_text
    assert result.normalized_text == "Café\nHe go to scool yesterday"
    assert result.confidence == 0.42
    assert result.warnings == (warning,)
    assert result.processing_duration_ms == 15.5


def test_successful_empty_extraction_remains_a_success(tmp_path: Path) -> None:
    page = _page(tmp_path, uuid4(), 1)
    provider = FakeOCRProvider(
        {1: OCRExtraction(raw_text="", confidence=None, processing_duration_ms=1)}
    )

    result = OCRService(provider, OCRNormalizer()).process_page(page)

    assert result.raw_text == ""
    assert result.normalized_text == ""
    assert result.confidence is None


def test_document_processing_preserves_page_order(tmp_path: Path) -> None:
    paper_id = uuid4()
    pages = (
        _page(tmp_path, paper_id, 1),
        _page(tmp_path, paper_id, 2),
    )
    document = _document(tmp_path, pages)
    provider = FakeOCRProvider(
        {
            1: OCRExtraction(raw_text="first", processing_duration_ms=1),
            2: OCRExtraction(raw_text="second", processing_duration_ms=2),
        }
    )

    results = OCRService(provider, OCRNormalizer()).process_document(document)

    assert provider.calls == [1, 2]
    assert [result.page_number for result in results] == [1, 2]
    assert [result.normalized_text for result in results] == ["first", "second"]


def test_provider_failure_becomes_error_not_empty_success(tmp_path: Path) -> None:
    page = _page(tmp_path, uuid4(), 1)
    provider = FakeOCRProvider({}, failure=RuntimeError("engine unavailable"))

    with pytest.raises(OCRProviderError, match="failed for page 1"):
        OCRService(provider, OCRNormalizer()).process_page(page)


def test_missing_rendered_image_fails_before_provider(tmp_path: Path) -> None:
    page = _page(tmp_path, uuid4(), 1, create_image=False)
    provider = FakeOCRProvider(
        {1: OCRExtraction(raw_text="text", processing_duration_ms=1)}
    )

    with pytest.raises(OCRProcessingError, match="unavailable"):
        OCRService(provider, OCRNormalizer()).process_page(page)

    assert provider.calls == []


def test_document_without_rendered_pages_is_rejected(tmp_path: Path) -> None:
    source_path = (tmp_path / "source.pdf").resolve()
    source_path.write_bytes(b"source placeholder")
    document = PaperDocument(
        paper_id=uuid4(),
        source_path=source_path,
        original_filename="source.pdf",
        page_count=1,
        file_size_bytes=source_path.stat().st_size,
        sha256="a" * 64,
    )
    provider = FakeOCRProvider({})

    with pytest.raises(OCRProcessingError, match="no rendered pages"):
        OCRService(provider, OCRNormalizer()).process_document(document)

    assert provider.calls == []

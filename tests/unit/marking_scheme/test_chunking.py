"""Tests for deterministic, Test-aware marking-scheme chunking."""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID

from app.marking_scheme.chunking import MarkingSchemeChunker
from app.marking_scheme.indexing import MarkingSchemeIndexingService
from app.marking_scheme.models import (
    MarkingSchemeOCRStatus,
    MarkingSchemePageOCR,
    MarkingSchemeProvenance,
    StructuredMarkingScheme,
)
from app.marking_scheme.parser import MarkingSchemeParser, SourceLine
from app.vectorstore.base import MetadataFilter
from app.vectorstore.models import KnowledgeDocument, KnowledgeSearchResult


class RecordingKnowledgeBase:
    def __init__(self) -> None:
        self.source_id: str | None = None
        self.documents: tuple[KnowledgeDocument, ...] = ()

    def replace_source(
        self,
        source_id: str,
        documents: tuple[KnowledgeDocument, ...] | list[KnowledgeDocument],
    ) -> None:
        self.source_id = source_id
        self.documents = tuple(documents)

    def search(
        self,
        query: str,
        *,
        k: int,
        metadata_filter: MetadataFilter | None = None,
    ) -> tuple[KnowledgeSearchResult, ...]:
        del query, k, metadata_filter
        return ()


def _scheme(lines: tuple[SourceLine, ...]) -> StructuredMarkingScheme:
    tests = MarkingSchemeParser().parse(lines)
    page_count = max(line.page_number for line in lines)
    page_ocr = tuple(
        MarkingSchemePageOCR(
            page_number=page_number,
            status=MarkingSchemeOCRStatus.EXTRACTED,
            provider="synthetic",
            duration_ms=1,
        )
        for page_number in range(1, page_count + 1)
    )
    maxima = tuple(item.maximum_marks for item in tests)
    total = (
        sum((maximum for maximum in maxima if maximum is not None), Decimal(0))
        if all(maximum is not None for maximum in maxima)
        else None
    )
    return StructuredMarkingScheme(
        scheme_id=UUID("00000000-0000-0000-0000-000000000001"),
        source_document_id=UUID("00000000-0000-0000-0000-000000000002"),
        page_count=page_count,
        tests=tests,
        total_available_marks=total,
        page_ocr=page_ocr,
        provenance=MarkingSchemeProvenance(
            ingestion_version="test-ingestion-v1",
            parser_version="test-parser-v1",
            source_sha256="a" * 64,
            ocr_provider="synthetic",
            successful_ocr_pages=tuple(range(1, page_count + 1)),
        ),
        processing_duration_ms=1,
    )


def test_chunks_keep_criteria_separate_and_filter_by_exact_test() -> None:
    scheme = _scheme(
        (
            SourceLine(page_number=1, text="Test 01: Vocabulary"),
            SourceLine(page_number=1, text="Accepted answers: colour / color (1 mark)"),
            SourceLine(page_number=1, text="Test 02: Reading"),
            SourceLine(page_number=2, text="Keywords: river / water (2 marks)"),
        )
    )

    documents = MarkingSchemeChunker().chunk(scheme)

    assert len(documents) == 2
    assert tuple(item.metadata["test_number"] for item in documents) == (1, 2)
    assert tuple(item.metadata["criterion_id"] for item in documents) == (
        "test-01-criterion-001",
        "test-02-criterion-001",
    )
    assert documents[0].metadata["test_title"] == "Vocabulary"
    assert documents[0].metadata["page_number"] == 1
    assert documents[1].metadata["page_number"] == 2
    assert "Accepted answers: colour / color" in documents[0].text
    assert "Keywords: river / water" in documents[1].text
    assert "river" not in documents[0].text


def test_chunk_ids_and_content_are_stable() -> None:
    scheme = _scheme(
        (
            SourceLine(page_number=1, text="Test 03"),
            SourceLine(page_number=1, text="1. response (1 mark)"),
        )
    )
    chunker = MarkingSchemeChunker()

    first = chunker.chunk(scheme)
    second = chunker.chunk(scheme)

    assert first == second
    assert len(first[0].document_id) == 64


def test_unknown_criterion_is_preserved_as_unclassified_source_evidence() -> None:
    scheme = _scheme(
        (
            SourceLine(page_number=1, text="Test 04"),
            SourceLine(page_number=1, text="Any suitable answer"),
        )
    )

    document = MarkingSchemeChunker().chunk(scheme)[0]

    assert document.metadata["criterion_type"] == "unknown"
    assert document.metadata["has_known_criterion_maximum"] is False
    assert "Description: Any suitable answer" in document.text
    assert "Source text: Any suitable answer" in document.text


def test_test_without_criteria_keeps_source_pages_and_test_level_context() -> None:
    scheme = _scheme(
        (
            SourceLine(page_number=1, text="Test 05: Writing"),
            SourceLine(page_number=2, text="Total: 5 marks"),
        )
    )

    documents = MarkingSchemeChunker().chunk(scheme)

    assert len(documents) == 1
    document = documents[0]
    assert document.metadata["criterion_type"] == "test_summary"
    assert document.metadata["criterion_id"] == "test-05-summary"
    assert document.metadata["source_pages"] == "1,2"
    assert document.metadata["test_maximum_marks"] == "5"
    assert "Source material:" in document.text
    assert "Total: 5 marks" in document.text


def test_indexing_service_replaces_scheme_source_with_traceable_chunks() -> None:
    scheme = _scheme(
        (
            SourceLine(page_number=1, text="Test 01: Vocabulary"),
            SourceLine(page_number=1, text="Accepted answers: colour / color (1 mark)"),
        )
    )
    store = RecordingKnowledgeBase()

    report = MarkingSchemeIndexingService(store).index(scheme)

    assert store.source_id == str(scheme.scheme_id)
    assert report["chunks_indexed"] == 1
    assert report["tests_indexed"] == 1
    assert store.documents[0].metadata["source_type"] == "marking_scheme"
    assert store.documents[0].metadata["test_number"] == 1
    assert store.documents[0].metadata["source_sha256"] == "a" * 64

"""Tests for syllabus indexing and the local Chroma adapter."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import pymupdf
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.core.exceptions import KnowledgeBaseIngestionError
from app.ingestion.validators import PDFValidator
from app.syllabus.ingestion import SyllabusIngestionService
from app.vectorstore.chroma_store import ChromaKnowledgeBaseStore
from app.vectorstore.models import KnowledgeDocument


class InMemoryKnowledgeBase:
    """Record source replacement calls without an embedding service."""

    def __init__(self) -> None:
        self.sources: dict[str, tuple[KnowledgeDocument, ...]] = {}

    def replace_source(
        self,
        source_id: str,
        documents: Sequence[KnowledgeDocument],
    ) -> None:
        self.sources[source_id] = tuple(documents)

    def search(
        self,
        query: str,
        *,
        k: int,
        metadata_filter: dict[str, str] | None = None,
    ) -> tuple[object, ...]:
        del query, k, metadata_filter
        return ()


class DeterministicEmbeddings(Embeddings):
    """Tiny test-only embedding function for isolated Chroma behavior tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_embedding(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return _embedding(text)


def test_syllabus_text_is_chunked_with_page_provenance_and_stable_ids(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "syllabus.pdf"
    _write_pdf(
        pdf_path,
        [
            "Grade 10 English reading writing and speaking curriculum.\n" * 30,
            "Assessment supports comprehension and language use.\n" * 8,
        ],
    )
    store = InMemoryKnowledgeBase()
    service = SyllabusIngestionService(
        store,
        PDFValidator(max_file_size_bytes=1_000_000, max_pages=10),
        chunk_size=180,
        chunk_overlap=20,
    )

    report = service.ingest(pdf_path)
    first_chunks = store.sources["grade-10-english-syllabus"]
    repeated_report = service.ingest(pdf_path)
    repeated_chunks = store.sources["grade-10-english-syllabus"]

    assert report["pages"] == 2
    assert report["pages_with_text"] == 2
    assert report["chunks_indexed"] > 2
    assert report["source_sha256"] == repeated_report["source_sha256"]
    assert [chunk.document_id for chunk in first_chunks] == [
        chunk.document_id for chunk in repeated_chunks
    ]
    assert all(chunk.metadata["source_type"] == "syllabus" for chunk in first_chunks)
    assert {chunk.metadata["page_number"] for chunk in first_chunks} == {1, 2}
    assert all(chunk.metadata["grade"] == 10 for chunk in first_chunks)


def test_syllabus_without_extractable_text_is_rejected(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty.pdf"
    _write_pdf(pdf_path, [""])
    service = SyllabusIngestionService(
        InMemoryKnowledgeBase(),
        PDFValidator(max_file_size_bytes=1_000_000, max_pages=10),
    )

    try:
        service.ingest(pdf_path)
    except KnowledgeBaseIngestionError as error:
        assert str(error) == "Syllabus PDF contains no extractable text"
    else:
        raise AssertionError("an image-only/empty syllabus must not be indexed")


def test_chroma_store_replaces_source_and_filters_retrieval(tmp_path: Path) -> None:
    vector_store = Chroma(
        collection_name="syllabus_test",
        embedding_function=DeterministicEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
    )
    store = ChromaKnowledgeBaseStore(vector_store)
    first = KnowledgeDocument(
        document_id="old-syllabus-chunk",
        text="Old curriculum text.",
        metadata={"source_id": "grade-10-english-syllabus", "source_type": "syllabus"},
    )
    replacement = KnowledgeDocument(
        document_id="new-syllabus-chunk",
        text="Current curriculum reading and writing outcomes.",
        metadata={"source_id": "grade-10-english-syllabus", "source_type": "syllabus"},
    )
    marking_scheme = KnowledgeDocument(
        document_id="scheme-chunk",
        text="Award one mark for the correct answer.",
        metadata={"source_id": "scheme-2025", "source_type": "marking_scheme"},
    )

    store.replace_source("grade-10-english-syllabus", [first])
    store.replace_source("grade-10-english-syllabus", [replacement])
    store.replace_source("scheme-2025", [marking_scheme])
    results = store.search(
        "reading writing outcomes",
        k=3,
        metadata_filter={"source_type": "syllabus"},
    )

    assert len(results) == 1
    assert results[0].text == replacement.text
    assert results[0].metadata["source_type"] == "syllabus"


def _write_pdf(path: Path, page_texts: list[str]) -> None:
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def _embedding(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [byte / 255.0 for byte in digest[:16]]

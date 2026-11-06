"""Tests for exact-Test marking-scheme retrieval."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.core.exceptions import KnowledgeBaseRetrievalError
from app.retrieval.retriever import MarkingSchemeRetriever
from app.vectorstore.base import MetadataFilter
from app.vectorstore.chroma_store import ChromaKnowledgeBaseStore
from app.vectorstore.models import KnowledgeDocument, KnowledgeSearchResult

SCHEME_ID = UUID("00000000-0000-0000-0000-000000000001")


class DeterministicEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_embedding(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return _embedding(text)


class RecordingKnowledgeBase:
    def __init__(self, results: tuple[KnowledgeSearchResult, ...]) -> None:
        self.results = results
        self.query: str | None = None
        self.k: int | None = None
        self.metadata_filter: MetadataFilter | None = None

    def replace_source(
        self,
        source_id: str,
        documents: Sequence[KnowledgeDocument],
    ) -> None:
        del source_id, documents

    def search(
        self,
        query: str,
        *,
        k: int,
        metadata_filter: MetadataFilter | None = None,
    ) -> tuple[KnowledgeSearchResult, ...]:
        self.query = query
        self.k = k
        self.metadata_filter = metadata_filter
        return self.results


def test_retriever_filters_by_scheme_and_exact_test_then_validates_metadata() -> None:
    result = KnowledgeSearchResult(
        text="Award one mark for the correct answer.",
        metadata=_metadata(test_number=4),
    )
    store = RecordingKnowledgeBase((result,))

    evidence = MarkingSchemeRetriever(store).retrieve(
        "student response and expected answer",
        scheme_id=SCHEME_ID,
        test_number=4,
        k=3,
    )

    assert store.query == "student response and expected answer"
    assert store.k == 3
    assert store.metadata_filter == {
        "source_type": "marking_scheme",
        "source_id": str(SCHEME_ID),
        "scheme_id": str(SCHEME_ID),
        "test_number": 4,
    }
    assert len(evidence) == 1
    assert evidence[0].scheme_id == SCHEME_ID
    assert evidence[0].test_number == 4
    assert evidence[0].source_pages == (2, 3)


def test_retriever_uses_numeric_test_filter_in_chroma(tmp_path: Path) -> None:
    vector_store = Chroma(
        collection_name="retrieval_test",
        embedding_function=DeterministicEmbeddings(),
        persist_directory=str(tmp_path / "chroma"),
    )
    store = ChromaKnowledgeBaseStore(vector_store)
    store.replace_source(
        str(SCHEME_ID),
        (
            KnowledgeDocument(
                document_id="test-four-rubric",
                text="Test four accepted answer criterion.",
                metadata=_metadata(test_number=4),
            ),
            KnowledgeDocument(
                document_id="test-five-rubric",
                text="Test five accepted answer criterion.",
                metadata=_metadata(test_number=5),
            ),
        ),
    )

    evidence = MarkingSchemeRetriever(store).retrieve(
        "accepted answer",
        scheme_id=SCHEME_ID,
        test_number=4,
        k=2,
    )

    assert len(evidence) == 1
    assert evidence[0].test_number == 4
    assert evidence[0].criterion_id == "test-04-criterion-001"


def test_retriever_returns_empty_when_no_rubric_matches() -> None:
    retriever = MarkingSchemeRetriever(RecordingKnowledgeBase(()))

    assert (
        retriever.retrieve(
            "answer text",
            scheme_id=SCHEME_ID,
            test_number=4,
            k=3,
        )
        == ()
    )


@pytest.mark.parametrize("returned_test_number", [5, True])
def test_retriever_rejects_cross_test_results(returned_test_number: int) -> None:
    store = RecordingKnowledgeBase(
        (
            KnowledgeSearchResult(
                text="wrong Test",
                metadata=_metadata(test_number=returned_test_number),
            ),
        )
    )

    with pytest.raises(KnowledgeBaseRetrievalError, match="does not match"):
        MarkingSchemeRetriever(store).retrieve(
            "answer text",
            scheme_id=SCHEME_ID,
            test_number=4,
            k=3,
        )


def test_retriever_rejects_invalid_source_pages() -> None:
    metadata = _metadata(test_number=4)
    metadata["source_pages"] = "2,0"
    store = RecordingKnowledgeBase(
        (KnowledgeSearchResult(text="rubric", metadata=metadata),)
    )

    with pytest.raises(KnowledgeBaseRetrievalError, match="invalid source pages"):
        MarkingSchemeRetriever(store).retrieve(
            "answer text",
            scheme_id=SCHEME_ID,
            test_number=4,
            k=3,
        )


def test_retriever_rejects_invalid_source_provenance() -> None:
    metadata = _metadata(test_number=4)
    metadata["source_sha256"] = "not-a-hash"
    store = RecordingKnowledgeBase(
        (KnowledgeSearchResult(text="rubric", metadata=metadata),)
    )

    with pytest.raises(KnowledgeBaseRetrievalError, match="invalid provenance"):
        MarkingSchemeRetriever(store).retrieve(
            "answer text",
            scheme_id=SCHEME_ID,
            test_number=4,
            k=3,
        )


def _metadata(*, test_number: int) -> dict[str, str | int]:
    return {
        "source_type": "marking_scheme",
        "source_id": str(SCHEME_ID),
        "scheme_id": str(SCHEME_ID),
        "source_sha256": "a" * 64,
        "test_number": test_number,
        "criterion_id": f"test-{test_number:02d}-criterion-001",
        "criterion_type": "exact_answer",
        "page_number": 2,
        "source_pages": "2,3",
    }


def _embedding(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [byte / 255.0 for byte in digest[:16]]

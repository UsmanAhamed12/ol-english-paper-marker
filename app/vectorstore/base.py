"""Application-facing vector-store contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from app.vectorstore.models import KnowledgeDocument, KnowledgeSearchResult

MetadataFilter = Mapping[str, str]


class KnowledgeBaseStore(Protocol):
    """Operations needed by document ingestion and retrieval services."""

    def replace_source(
        self,
        source_id: str,
        documents: Sequence[KnowledgeDocument],
    ) -> None:
        """Upsert a source's chunks and remove obsolete chunks for that source."""

    def search(
        self,
        query: str,
        *,
        k: int,
        metadata_filter: MetadataFilter | None = None,
    ) -> tuple[KnowledgeSearchResult, ...]:
        """Return the most relevant chunks matching the optional metadata filter."""

"""Index parsed marking-scheme chunks in the local knowledge base."""

from __future__ import annotations

from app.core.exceptions import KnowledgeBaseIngestionError
from app.marking_scheme.chunking import MarkingSchemeChunker
from app.marking_scheme.models import StructuredMarkingScheme
from app.vectorstore.base import KnowledgeBaseStore


class MarkingSchemeIndexingService:
    """Embed and persist question-aware chunks for one marking scheme."""

    def __init__(
        self,
        vector_store: KnowledgeBaseStore,
        chunker: MarkingSchemeChunker | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._chunker = chunker or MarkingSchemeChunker()

    def index(self, scheme: StructuredMarkingScheme) -> dict[str, int | str]:
        """Replace this scheme's Chroma records and return a safe summary."""

        chunks = self._chunker.chunk(scheme)
        if not chunks:
            raise KnowledgeBaseIngestionError(
                "Marking scheme produced no retrieval chunks"
            )
        source_id = str(scheme.scheme_id)
        self._vector_store.replace_source(source_id, chunks)
        return {
            "source_id": source_id,
            "source_sha256": scheme.provenance.source_sha256,
            "tests_indexed": len(scheme.tests),
            "chunks_indexed": len(chunks),
        }

"""ChromaDB adapter using the maintained LangChain integration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import chromadb.errors
import httpx
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from app.core.config import Settings
from app.core.exceptions import KnowledgeBaseRetrievalError
from app.vectorstore.base import MetadataFilter
from app.vectorstore.models import KnowledgeDocument, KnowledgeSearchResult


class ChromaKnowledgeBaseStore:
    """Persist and search local knowledge chunks in a Chroma collection."""

    def __init__(self, vector_store: Chroma) -> None:
        self._vector_store = vector_store

    @classmethod
    def from_settings(cls, settings: Settings) -> ChromaKnowledgeBaseStore:
        """Construct the local Chroma collection and Ollama embedding client."""

        embeddings = OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=str(settings.ollama_base_url).rstrip("/"),
        )
        vector_store = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=embeddings,
            persist_directory=str(settings.chroma_persist_dir),
        )
        return cls(vector_store)

    def replace_source(
        self,
        source_id: str,
        documents: Sequence[KnowledgeDocument],
    ) -> None:
        """Upsert new chunks, then remove stale chunks from the same source."""

        if not documents:
            raise ValueError("A knowledge source must contain at least one chunk")
        document_ids = [document.document_id for document in documents]
        self._vector_store.add_documents(
            [
                Document(page_content=document.text, metadata=dict(document.metadata))
                for document in documents
            ],
            ids=document_ids,
        )
        existing = self._vector_store.get(
            where={"source_id": source_id},
            include=["metadatas"],
        )
        stale_ids = sorted(set(existing["ids"]) - set(document_ids))
        if stale_ids:
            self._vector_store.delete(ids=stale_ids)

    def search(
        self,
        query: str,
        *,
        k: int,
        metadata_filter: MetadataFilter | None = None,
    ) -> tuple[KnowledgeSearchResult, ...]:
        """Search the collection with optional Chroma metadata constraints."""

        chroma_filter: dict[str, Any] | None = None
        if metadata_filter:
            filter_items = tuple(metadata_filter.items())
            if len(filter_items) == 1:
                chroma_filter = {filter_items[0][0]: filter_items[0][1]}
            else:
                chroma_filter = {
                    "$and": [{key: {"$eq": value}} for key, value in filter_items]
                }
        try:
            documents = self._vector_store.similarity_search(
                query,
                k=k,
                filter=chroma_filter,
            )
        except (
            chromadb.errors.ChromaError,
            httpx.HTTPError,
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            raise KnowledgeBaseRetrievalError(
                "Local knowledge-base retrieval failed"
            ) from error
        return tuple(
            KnowledgeSearchResult(
                text=document.page_content,
                metadata=document.metadata,
            )
            for document in documents
        )

"""Extract and chunk the local English syllabus for Chroma indexing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.exceptions import KnowledgeBaseIngestionError
from app.ingestion.validators import PDFValidator
from app.vectorstore.base import KnowledgeBaseStore
from app.vectorstore.models import KnowledgeDocument, MetadataValue

SYLLABUS_SOURCE_ID = "grade-10-english-syllabus"
SYLLABUS_CHUNK_SIZE = 1000
SYLLABUS_CHUNK_OVERLAP = 150


class SyllabusIngestionService:
    """Index syllabus text with page-level traceability and stable chunk IDs."""

    def __init__(
        self,
        vector_store: KnowledgeBaseStore,
        pdf_validator: PDFValidator,
        *,
        chunk_size: int = SYLLABUS_CHUNK_SIZE,
        chunk_overlap: int = SYLLABUS_CHUNK_OVERLAP,
    ) -> None:
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("Syllabus chunk settings are invalid")
        self._vector_store = vector_store
        self._pdf_validator = pdf_validator
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    def ingest(self, pdf_path: Path) -> dict[str, int | str]:
        """Extract selectable PDF text, index chunks, and return safe counts."""

        validated = self._pdf_validator.validate(pdf_path)
        source_sha256 = _sha256(validated.source_path)
        chunks: list[KnowledgeDocument] = []
        extracted_pages = 0
        try:
            with pymupdf.open(str(validated.source_path)) as pdf:  # type: ignore[no-untyped-call]
                for page_number, page in enumerate(pdf, start=1):
                    page_text = page.get_text("text").strip()
                    if not page_text:
                        continue
                    extracted_pages += 1
                    metadata: dict[str, MetadataValue] = {
                        "source_type": "syllabus",
                        "source_id": SYLLABUS_SOURCE_ID,
                        "source_name": validated.source_path.name,
                        "source_sha256": source_sha256,
                        "subject": "English",
                        "grade": 10,
                        "page_number": page_number,
                    }
                    split_chunks = self._splitter.create_documents(
                        [page_text],
                        metadatas=[metadata],
                    )
                    for chunk_number, split_chunk in enumerate(split_chunks):
                        start_index = int(split_chunk.metadata.get("start_index", 0))
                        chunk_id = _chunk_id(
                            source_sha256,
                            page_number,
                            chunk_number,
                        )
                        chunk_metadata: dict[str, MetadataValue] = {
                            **metadata,
                            "chunk_number": chunk_number,
                            "start_index": start_index,
                        }
                        chunks.append(
                            KnowledgeDocument(
                                document_id=chunk_id,
                                text=split_chunk.page_content,
                                metadata=chunk_metadata,
                            )
                        )
        except (OSError, RuntimeError, ValueError) as error:
            raise KnowledgeBaseIngestionError(
                "Syllabus PDF text could not be extracted"
            ) from error

        if not chunks:
            raise KnowledgeBaseIngestionError(
                "Syllabus PDF contains no extractable text"
            )
        self._vector_store.replace_source(SYLLABUS_SOURCE_ID, chunks)
        return {
            "pages": validated.page_count,
            "pages_with_text": extracted_pages,
            "chunks_indexed": len(chunks),
            "source_sha256": source_sha256,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunk_id(source_sha256: str, page_number: int, chunk_number: int) -> str:
    identity = f"{SYLLABUS_SOURCE_ID}:{source_sha256}:{page_number}:{chunk_number}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()

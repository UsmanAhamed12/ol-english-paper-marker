"""Question-aware retrieval for locally indexed marking schemes."""

from __future__ import annotations

import re
from uuid import UUID

from app.core.exceptions import KnowledgeBaseRetrievalError
from app.retrieval.models import MarkingSchemeEvidence
from app.vectorstore.base import KnowledgeBaseStore, MetadataFilter
from app.vectorstore.models import KnowledgeSearchResult

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class MarkingSchemeRetriever:
    """Filter rubric candidates to one scheme and exact Test before ranking."""

    def __init__(self, vector_store: KnowledgeBaseStore) -> None:
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        *,
        scheme_id: UUID,
        test_number: int,
        k: int,
    ) -> tuple[MarkingSchemeEvidence, ...]:
        """Return validated rubric chunks for exactly one scheme Test."""

        if not query.strip():
            raise ValueError("A non-empty retrieval query is required")
        if isinstance(test_number, bool) or test_number <= 0:
            raise ValueError("test_number must be a positive integer")
        if isinstance(k, bool) or k <= 0:
            raise ValueError("k must be a positive integer")

        expected_scheme_id = str(scheme_id)
        metadata_filter: MetadataFilter = {
            "source_type": "marking_scheme",
            "source_id": expected_scheme_id,
            "scheme_id": expected_scheme_id,
            "test_number": test_number,
        }
        results = self._vector_store.search(
            query.strip(),
            k=k,
            metadata_filter=metadata_filter,
        )
        return tuple(
            _validate_evidence(result, scheme_id, test_number) for result in results
        )


def _validate_evidence(
    result: KnowledgeSearchResult,
    expected_scheme_id: UUID,
    expected_test_number: int,
) -> MarkingSchemeEvidence:
    metadata = result.metadata
    scheme_id_text = str(expected_scheme_id)
    metadata_test_number = metadata.get("test_number")
    if (
        metadata.get("source_type") != "marking_scheme"
        or metadata.get("source_id") != scheme_id_text
        or metadata.get("scheme_id") != scheme_id_text
        or not isinstance(metadata_test_number, int)
        or isinstance(metadata_test_number, bool)
        or metadata_test_number != expected_test_number
    ):
        raise KnowledgeBaseRetrievalError(
            "Retrieved rubric evidence does not match the requested scheme and Test"
        )

    source_sha256 = metadata.get("source_sha256")
    criterion_id = metadata.get("criterion_id")
    criterion_type = metadata.get("criterion_type")
    page_number = metadata.get("page_number")
    source_pages = metadata.get("source_pages")
    if (
        not isinstance(source_sha256, str)
        or _SHA256_PATTERN.fullmatch(source_sha256) is None
        or not isinstance(criterion_id, str)
        or not criterion_id
        or not isinstance(criterion_type, str)
        or not criterion_type
        or not isinstance(page_number, int)
        or isinstance(page_number, bool)
        or page_number <= 0
        or not isinstance(source_pages, str)
    ):
        raise KnowledgeBaseRetrievalError(
            "Retrieved rubric evidence has invalid provenance"
        )

    try:
        parsed_source_pages = tuple(int(page) for page in source_pages.split(","))
        if not parsed_source_pages or any(page <= 0 for page in parsed_source_pages):
            raise ValueError("Source pages must be positive integers")
        return MarkingSchemeEvidence(
            text=result.text,
            scheme_id=expected_scheme_id,
            source_sha256=source_sha256,
            test_number=expected_test_number,
            criterion_id=criterion_id,
            criterion_type=criterion_type,
            page_number=page_number,
            source_pages=parsed_source_pages,
        )
    except (ValueError, TypeError) as error:
        raise KnowledgeBaseRetrievalError(
            "Retrieved rubric evidence has invalid source pages"
        ) from error

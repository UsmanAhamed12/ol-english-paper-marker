"""Typed evidence returned by question-aware marking-scheme retrieval."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class MarkingSchemeEvidence(BaseModel):
    """One rubric chunk with validated scheme, Test, and page provenance."""

    model_config = ConfigDict(frozen=True)

    text: Annotated[str, Field(min_length=1)]
    scheme_id: UUID
    source_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    test_number: Annotated[int, Field(gt=0)]
    criterion_id: Annotated[str, Field(min_length=1)]
    criterion_type: Annotated[str, Field(min_length=1)]
    page_number: Annotated[int, Field(gt=0)]
    source_pages: tuple[Annotated[int, Field(gt=0)], ...]

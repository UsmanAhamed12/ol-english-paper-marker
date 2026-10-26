"""Framework-independent records passed to and returned from vector stores."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

MetadataValue = str | int | float | bool


class KnowledgeDocument(BaseModel):
    """One text chunk with stable identity and scalar retrieval metadata."""

    model_config = ConfigDict(frozen=True)

    document_id: Annotated[str, Field(min_length=1)]
    text: Annotated[str, Field(min_length=1)]
    metadata: dict[str, MetadataValue]


class KnowledgeSearchResult(BaseModel):
    """A retrieved reference chunk and its source metadata."""

    model_config = ConfigDict(frozen=True)

    text: Annotated[str, Field(min_length=1)]
    metadata: dict[str, MetadataValue]

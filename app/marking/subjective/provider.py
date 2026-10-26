"""Provider abstraction for replaceable local subjective assessment."""

from __future__ import annotations

from typing import Annotated, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class SubjectiveProviderResponse(BaseModel):
    """Exact provider output and safe execution metadata."""

    model_config = ConfigDict(frozen=True)

    raw_text: str
    provider: Annotated[str, Field(min_length=1)]
    model_version: Annotated[str, Field(min_length=1)]
    processing_duration_ms: Annotated[float, Field(ge=0)]


@runtime_checkable
class SubjectiveMarkingProvider(Protocol):
    """Local provider contract consumed by the application service."""

    @property
    def name(self) -> str: ...

    @property
    def model_version(self) -> str: ...

    def assess(self, prompt: str) -> SubjectiveProviderResponse:
        """Return one exact raw structured-assessment response."""

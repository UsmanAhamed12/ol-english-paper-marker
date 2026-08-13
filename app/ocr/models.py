"""Validated provider output and normalized OCR page results."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
DurationMilliseconds = Annotated[float, Field(ge=0.0)]
NonBlankString = Annotated[str, Field(min_length=1)]


class OCRWarningCode(StrEnum):
    """Non-fatal concerns a provider can attach to an extraction."""

    LOW_CONFIDENCE = "low_confidence"
    PARTIAL_EXTRACTION = "partial_extraction"
    HANDWRITING_AMBIGUITY = "handwriting_ambiguity"
    IMAGE_RESOLUTION_CONCERN = "image_resolution_concern"


class OCRExtraction(BaseModel):
    """Raw evidence returned by a provider before normalization."""

    model_config = ConfigDict(frozen=True)

    raw_text: str
    confidence: Confidence | None = None
    warnings: tuple[OCRWarningCode, ...] = ()
    processing_duration_ms: DurationMilliseconds


class OCRPageResult(OCRExtraction):
    """Normalized OCR output with complete page and provider provenance."""

    paper_id: UUID
    page_number: Annotated[int, Field(gt=0)]
    source_image_path: Path
    normalized_text: str
    provider: NonBlankString
    model_version: NonBlankString | None = None

    @field_validator("source_image_path")
    @classmethod
    def source_image_path_must_be_absolute(cls, value: Path) -> Path:
        """Keep provenance independent of the current working directory."""

        if not value.is_absolute():
            raise ValueError("source_image_path must be absolute")
        return value

    @field_validator("provider", "model_version")
    @classmethod
    def provenance_must_not_be_whitespace_only(cls, value: str | None) -> str | None:
        """Reject provenance values that carry no useful identity."""

        if value is not None and not value.strip():
            raise ValueError("provider provenance must not be blank")
        return value

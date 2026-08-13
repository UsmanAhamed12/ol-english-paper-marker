"""Typed provenance for OCR image preprocessing experiments."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PreprocessingOperation(StrEnum):
    """Conservative operations supported by the fixed Phase 4C.2 experiment."""

    GRAYSCALE = "grayscale"
    DENOISE = "mild_denoise"
    THRESHOLD = "otsu_threshold"


class PreprocessingVariant(StrEnum):
    """Predetermined, sample-independent experiment variants."""

    NONE = "none"
    GRAYSCALE = "grayscale"
    GRAYSCALE_DENOISE = "grayscale-denoise"
    GRAYSCALE_THRESHOLD = "grayscale-threshold"
    GRAYSCALE_DENOISE_THRESHOLD = "grayscale-denoise-threshold"

    @property
    def operations(self) -> tuple[PreprocessingOperation, ...]:
        """Return the ordered operations for this fixed variant."""

        variants = {
            self.NONE: (),
            self.GRAYSCALE: (PreprocessingOperation.GRAYSCALE,),
            self.GRAYSCALE_DENOISE: (
                PreprocessingOperation.GRAYSCALE,
                PreprocessingOperation.DENOISE,
            ),
            self.GRAYSCALE_THRESHOLD: (
                PreprocessingOperation.GRAYSCALE,
                PreprocessingOperation.THRESHOLD,
            ),
            self.GRAYSCALE_DENOISE_THRESHOLD: (
                PreprocessingOperation.GRAYSCALE,
                PreprocessingOperation.DENOISE,
                PreprocessingOperation.THRESHOLD,
            ),
        }
        return variants[self]


class PreprocessingResult(BaseModel):
    """Immutable provenance for one derived image."""

    model_config = ConfigDict(frozen=True)

    source_image_path: Path
    processed_image_path: Path
    source_width: Annotated[int, Field(gt=0)]
    source_height: Annotated[int, Field(gt=0)]
    processed_width: Annotated[int, Field(gt=0)]
    processed_height: Annotated[int, Field(gt=0)]
    operations: Annotated[tuple[PreprocessingOperation, ...], Field(min_length=1)]
    processing_duration_ms: Annotated[float, Field(ge=0.0)]

    @field_validator("source_image_path", "processed_image_path")
    @classmethod
    def paths_must_be_absolute(cls, value: Path) -> Path:
        """Keep evidence paths independent of the current directory."""

        if not value.is_absolute():
            raise ValueError("preprocessing image paths must be absolute")
        return value

    @model_validator(mode="after")
    def preserve_source_and_geometry(self) -> Self:
        """Reject in-place output or a geometry-changing result."""

        if self.source_image_path == self.processed_image_path:
            raise ValueError("processed image path must differ from source")
        if (self.source_width, self.source_height) != (
            self.processed_width,
            self.processed_height,
        ):
            raise ValueError("preprocessing must preserve image dimensions")
        return self

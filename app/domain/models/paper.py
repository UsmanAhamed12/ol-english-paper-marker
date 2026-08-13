"""Domain models produced by the PDF ingestion boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PositiveDimension = Annotated[int, Field(gt=0)]


class PaperPage(BaseModel):
    """Metadata for one rendered, human-numbered paper page."""

    model_config = ConfigDict(frozen=True)

    paper_id: UUID
    page_number: Annotated[int, Field(gt=0)]
    image_path: Path
    width: PositiveDimension
    height: PositiveDimension
    image_format: Literal["PNG"] = "PNG"

    @field_validator("image_path")
    @classmethod
    def image_path_must_be_absolute(cls, value: Path) -> Path:
        """Keep runtime references unambiguous across working directories."""

        if not value.is_absolute():
            raise ValueError("image_path must be absolute")
        return value


class PaperDocument(BaseModel):
    """Validated source metadata and optional rendered page references."""

    model_config = ConfigDict(frozen=True)

    paper_id: UUID
    source_path: Path
    original_filename: Annotated[str, Field(min_length=1)]
    page_count: Annotated[int, Field(gt=0)]
    file_size_bytes: Annotated[int, Field(gt=0)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    pages: tuple[PaperPage, ...] = ()

    @field_validator("source_path")
    @classmethod
    def source_path_must_be_absolute(cls, value: Path) -> Path:
        """Keep the immutable source reference explicit and stable."""

        if not value.is_absolute():
            raise ValueError("source_path must be absolute")
        return value

    @model_validator(mode="after")
    def validate_rendered_pages(self) -> Self:
        """Ensure a rendered document has one ordered page set of its own."""

        if not self.pages:
            return self
        if len(self.pages) != self.page_count:
            raise ValueError("rendered page count must match source page count")
        expected_numbers = list(range(1, self.page_count + 1))
        if [page.page_number for page in self.pages] != expected_numbers:
            raise ValueError("rendered pages must be ordered from page 1")
        if any(page.paper_id != self.paper_id for page in self.pages):
            raise ValueError("rendered pages must belong to this paper")
        return self

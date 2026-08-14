"""Immutable domain models for detected exam Test structure."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ocr.models import BoundingBox

UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveTestNumber = Annotated[int, Field(ge=1, le=99)]


class MarkerDetectionStrategy(StrEnum):
    """Explain how a possible Test heading was recognized."""

    EXACT_TOKENS = "exact_tokens"
    COMPACT_TOKEN = "compact_token"
    OCR_CONFUSION = "ocr_confusion"
    FUZZY_KEYWORD = "fuzzy_keyword"


class TestMarkerCandidate(BaseModel):
    """One plausible marker before document-sequence selection."""

    model_config = ConfigDict(frozen=True)

    test_number: PositiveTestNumber
    raw_text: Annotated[str, Field(min_length=1)]
    page_number: Annotated[int, Field(gt=0)]
    bbox: BoundingBox
    confidence: UnitInterval
    text_similarity: UnitInterval
    numeric_confidence: UnitInterval
    ocr_confidence: UnitInterval | None = None
    geometry_confidence: UnitInterval
    strategy: MarkerDetectionStrategy
    source_word_indices: Annotated[tuple[int, ...], Field(min_length=1)]


class TestMarker(TestMarkerCandidate):
    """Accepted marker with explainable document-sequence support."""

    sequence_confidence: UnitInterval

    @property
    def label(self) -> str:
        """Return a safe normalized display label."""

        return f"Test {self.test_number:02d}"


class TestPageRegion(BaseModel):
    """Page-local portion of a possibly cross-page Test region."""

    model_config = ConfigDict(frozen=True)

    page_number: Annotated[int, Field(gt=0)]
    bbox: BoundingBox


class TestRegion(BaseModel):
    """Ordered spatial region from one marker up to the next marker."""

    model_config = ConfigDict(frozen=True)

    test_number: PositiveTestNumber
    label: Annotated[str, Field(pattern=r"^Test [0-9]{2}$")]
    marker: TestMarker
    page_regions: Annotated[tuple[TestPageRegion, ...], Field(min_length=1)]
    start_page: Annotated[int, Field(gt=0)]
    end_page: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def region_matches_marker_and_pages(self) -> Self:
        if self.test_number != self.marker.test_number:
            raise ValueError("Test region number must match its marker")
        if self.label != self.marker.label:
            raise ValueError("Test region label must be normalized")
        page_numbers = [region.page_number for region in self.page_regions]
        if page_numbers != list(range(self.start_page, self.end_page + 1)):
            raise ValueError("Test region page spans must be contiguous")
        if self.start_page != self.marker.page_number:
            raise ValueError("Test region must start on its marker page")
        return self


class ExamPageStructure(BaseModel):
    """Detection evidence and accepted spatial spans for one page."""

    model_config = ConfigDict(frozen=True)

    page_number: Annotated[int, Field(gt=0)]
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    candidates: tuple[TestMarkerCandidate, ...] = ()
    markers: tuple[TestMarker, ...] = ()
    regions: tuple[TestPageRegion, ...] = ()

    @model_validator(mode="after")
    def evidence_fits_page(self) -> Self:
        boxes = (
            *(candidate.bbox for candidate in self.candidates),
            *(marker.bbox for marker in self.markers),
            *(region.bbox for region in self.regions),
        )
        if any(
            box.x + box.width > self.width or box.y + box.height > self.height
            for box in boxes
        ):
            raise ValueError("Structure evidence must fit within page geometry")
        if any(region.page_number != self.page_number for region in self.regions):
            raise ValueError("Page regions must belong to their containing page")
        return self


class ExamStructure(BaseModel):
    """Complete ordered Test structure with explicit uncertainty."""

    model_config = ConfigDict(frozen=True)

    paper_id: UUID
    page_count: Annotated[int, Field(gt=0)]
    pages: Annotated[tuple[ExamPageStructure, ...], Field(min_length=1)]
    tests: tuple[TestRegion, ...] = ()
    missing_test_numbers: tuple[PositiveTestNumber, ...] = ()
    duplicate_test_numbers: tuple[PositiveTestNumber, ...] = ()
    rejected_candidates: tuple[TestMarkerCandidate, ...] = ()

    @model_validator(mode="after")
    def validate_document_order(self) -> Self:
        if len(self.pages) != self.page_count:
            raise ValueError("Exam structure page count must match page evidence")
        if [page.page_number for page in self.pages] != list(
            range(1, self.page_count + 1)
        ):
            raise ValueError("Exam structure pages must be ordered")
        test_numbers = [region.test_number for region in self.tests]
        if test_numbers != sorted(set(test_numbers)):
            raise ValueError("Detected Tests must be unique and increasing")
        return self

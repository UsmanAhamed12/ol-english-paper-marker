"""Immutable domain models for spatial evidence and answer-area candidates."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.ocr.models import BoundingBox

UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveTestNumber = Annotated[int, Field(ge=1, le=99)]


class EvidenceType(StrEnum):
    """Conservative evidence attribution; uncertainty remains first-class."""

    PRINTED = "printed"
    STUDENT_CANDIDATE = "student_candidate"
    TEACHER_CANDIDATE = "teacher_candidate"
    UNKNOWN = "unknown"


class EvidenceSignal(StrEnum):
    """Explainable signals contributing to an evidence classification."""

    HIGH_OCR_CONFIDENCE = "high_ocr_confidence"
    LOW_OCR_CONFIDENCE = "low_ocr_confidence"
    REGULAR_GEOMETRY = "regular_geometry"
    IRREGULAR_BASELINE = "irregular_baseline"
    IRREGULAR_HEIGHT = "irregular_height"
    IRREGULAR_SPACING = "irregular_spacing"
    DENSE_TEXT = "dense_text"
    FRAGMENTED_STROKES = "fragmented_strokes"
    CHROMATIC_INK = "chromatic_ink"
    RED_INK_DOMINANT = "red_ink_dominant"
    BLUE_INK_DOMINANT = "blue_ink_dominant"
    ISOLATED_MARK = "isolated_mark"
    MARGIN_POSITION = "margin_position"
    HIGH_LOCAL_CONTRAST = "high_local_contrast"


class AnswerRegionSignal(StrEnum):
    """Spatial evidence supporting a student-answer-area candidate."""

    WRITING_GUIDES = "writing_guides"
    STUDENT_EVIDENCE_CLUSTER = "student_evidence_cluster"
    LOW_PRINTED_DENSITY = "low_printed_density"
    BLANK_WRITING_SPACE = "blank_writing_space"


class InkFeatures(BaseModel):
    """Measured local raster features; none of them identifies an author alone."""

    model_config = ConfigDict(frozen=True)

    mean_saturation: UnitInterval
    saturation_std: UnitInterval
    foreground_ratio: UnitInterval
    red_foreground_ratio: UnitInterval
    blue_foreground_ratio: UnitInterval
    dark_foreground_ratio: UnitInterval
    local_contrast: UnitInterval
    edge_density: UnitInterval
    connected_component_count: Annotated[int, Field(ge=0)]


class GeometryFeatures(BaseModel):
    """Normalized OCR-line geometry associated with one evidence candidate."""

    model_config = ConfigDict(frozen=True)

    regularity: UnitInterval
    baseline_irregularity: UnitInterval
    height_irregularity: UnitInterval
    spacing_irregularity: UnitInterval
    line_density: UnitInterval
    fragmentation: UnitInterval
    isolation: UnitInterval
    margin_position: UnitInterval
    word_count_in_line: Annotated[int, Field(ge=0)]


class EvidenceFeatures(BaseModel):
    """Provider and image measurements used by the deterministic classifier."""

    model_config = ConfigDict(frozen=True)

    ink: InkFeatures
    geometry: GeometryFeatures
    ocr_confidence: UnitInterval | None = None


class EvidenceClassification(BaseModel):
    """One explainable, non-ground-truth attribution decision."""

    model_config = ConfigDict(frozen=True)

    evidence_type: EvidenceType
    score: UnitInterval
    signals: tuple[EvidenceSignal, ...]
    strategy_version: Annotated[str, Field(min_length=1)]

    @field_validator("signals")
    @classmethod
    def signals_must_be_unique(
        cls, value: tuple[EvidenceSignal, ...]
    ) -> tuple[EvidenceSignal, ...]:
        if len(value) != len(set(value)):
            raise ValueError("Evidence signals must be unique")
        return value


class EvidenceRegion(BaseModel):
    """Page-local evidence with geometry, features, and classification provenance."""

    model_config = ConfigDict(frozen=True)

    paper_id: UUID
    page_number: Annotated[int, Field(gt=0)]
    test_number: PositiveTestNumber
    bbox: BoundingBox
    evidence_type: EvidenceType
    confidence: UnitInterval
    signals: tuple[EvidenceSignal, ...]
    features: EvidenceFeatures
    source_word_indices: tuple[Annotated[int, Field(ge=0)], ...] = ()
    source_image_path: Path
    classification_strategy: Annotated[str, Field(min_length=1)]

    @field_validator("source_image_path")
    @classmethod
    def source_path_must_be_absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("Evidence source image path must be absolute")
        return value


class StudentAnswerRegion(BaseModel):
    """Candidate answer-space envelope, not a claim about recognized text."""

    model_config = ConfigDict(frozen=True)

    paper_id: UUID
    page_number: Annotated[int, Field(gt=0)]
    test_number: PositiveTestNumber
    bbox: BoundingBox
    confidence: UnitInterval
    signals: Annotated[tuple[AnswerRegionSignal, ...], Field(min_length=1)]
    source_evidence_indices: tuple[Annotated[int, Field(ge=0)], ...] = ()
    source_image_path: Path
    detection_strategy: Annotated[str, Field(min_length=1)]

    @field_validator("source_image_path")
    @classmethod
    def source_path_must_be_absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("Answer-region source image path must be absolute")
        return value


class TestEvidence(BaseModel):
    """Evidence and answer candidates for one Test on one page."""

    model_config = ConfigDict(frozen=True)

    paper_id: UUID
    page_number: Annotated[int, Field(gt=0)]
    test_number: PositiveTestNumber
    region_bbox: BoundingBox
    evidence_regions: tuple[EvidenceRegion, ...] = ()
    answer_regions: tuple[StudentAnswerRegion, ...] = ()

    @model_validator(mode="after")
    def children_match_test(self) -> Self:
        provenance_invalid = any(
            region.paper_id != self.paper_id
            or region.page_number != self.page_number
            or region.test_number != self.test_number
            for region in self.evidence_regions
        ) or any(
            region.paper_id != self.paper_id
            or region.page_number != self.page_number
            or region.test_number != self.test_number
            for region in self.answer_regions
        )
        if provenance_invalid:
            raise ValueError("Evidence children must match their Test provenance")
        boxes = (
            *(region.bbox for region in self.evidence_regions),
            *(region.bbox for region in self.answer_regions),
        )
        if any(not _contains(self.region_bbox, box) for box in boxes):
            raise ValueError("Evidence children must fit within their Test region")
        return self


class PageEvidence(BaseModel):
    """All analyzed Test evidence on one canonical page."""

    model_config = ConfigDict(frozen=True)

    paper_id: UUID
    page_number: Annotated[int, Field(gt=0)]
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    source_image_path: Path
    tests: tuple[TestEvidence, ...] = ()

    @model_validator(mode="after")
    def tests_fit_page(self) -> Self:
        page_box = BoundingBox(x=0, y=0, width=self.width, height=self.height)
        if any(
            test.paper_id != self.paper_id
            or test.page_number != self.page_number
            or not _contains(page_box, test.region_bbox)
            for test in self.tests
        ):
            raise ValueError("Page evidence must match and fit its canonical page")
        return self


class DocumentEvidence(BaseModel):
    """Ordered page evidence for one paper."""

    model_config = ConfigDict(frozen=True)

    paper_id: UUID
    pages: Annotated[tuple[PageEvidence, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def pages_are_ordered(self) -> Self:
        if any(page.paper_id != self.paper_id for page in self.pages):
            raise ValueError("Document evidence pages must belong to one paper")
        numbers = [page.page_number for page in self.pages]
        if numbers != sorted(set(numbers)):
            raise ValueError("Document evidence pages must be unique and ordered")
        return self


def _contains(container: BoundingBox, child: BoundingBox) -> bool:
    return (
        child.x >= container.x
        and child.y >= container.y
        and child.x + child.width <= container.x + container.width
        and child.y + child.height <= container.y + container.height
    )

"""Typed private ground truth and result models for Test-marker evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ocr.models import BoundingBox

SafeIdentifier = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]


class StructureGroundTruthMarker(BaseModel):
    """Human-verified marker location without private page transcription."""

    model_config = ConfigDict(frozen=True)

    test_number: Annotated[int, Field(ge=1, le=99)]
    page_number: Annotated[int, Field(gt=0)]
    bbox: BoundingBox


class StructureBenchmarkPaper(BaseModel):
    """One private paper and its manually verified Test markers."""

    model_config = ConfigDict(frozen=True)

    paper_alias: SafeIdentifier
    source_path: Path
    expected_page_count: Annotated[int, Field(gt=0)]
    expected_markers: Annotated[
        tuple[StructureGroundTruthMarker, ...], Field(min_length=1)
    ]

    @model_validator(mode="after")
    def markers_are_unique_and_ordered(self) -> Self:
        identities = [
            (marker.test_number, marker.page_number) for marker in self.expected_markers
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("Structure ground-truth markers must be unique")
        positions = [
            (marker.page_number, marker.bbox.y, marker.test_number)
            for marker in self.expected_markers
        ]
        if positions != sorted(positions):
            raise ValueError("Structure ground-truth markers must be document ordered")
        if any(
            marker.page_number > self.expected_page_count
            for marker in self.expected_markers
        ):
            raise ValueError("Structure marker page exceeds paper page count")
        return self


class StructureBenchmarkManifest(BaseModel):
    """Private, human-verified structure benchmark manifest."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    human_verified: Literal[True]
    papers: Annotated[tuple[StructureBenchmarkPaper, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def aliases_are_unique(self) -> Self:
        aliases = [paper.paper_alias for paper in self.papers]
        if len(aliases) != len(set(aliases)):
            raise ValueError("Structure benchmark aliases must be unique")
        return self


class StructureBenchmarkResult(BaseModel):
    """Safe per-paper structure metrics without OCR text or source identity."""

    model_config = ConfigDict(frozen=True)

    paper_alias: SafeIdentifier
    page_count: Annotated[int, Field(gt=0)]
    expected_markers: Annotated[int, Field(ge=0)]
    detected_markers: Annotated[int, Field(ge=0)]
    true_positives: Annotated[int, Field(ge=0)]
    false_positives: Annotated[int, Field(ge=0)]
    false_negatives: Annotated[int, Field(ge=0)]
    duplicate_markers: Annotated[int, Field(ge=0)]
    precision: UnitInterval
    recall: UnitInterval
    f1: UnitInterval
    test_number_accuracy: UnitInterval
    ordering_accuracy: UnitInterval
    missing_test_numbers: tuple[Annotated[int, Field(ge=1, le=99)], ...]


class StructureBenchmarkSummary(BaseModel):
    """Micro-aggregated structure metrics retaining error counts."""

    model_config = ConfigDict(frozen=True)

    paper_count: Annotated[int, Field(gt=0)]
    page_count: Annotated[int, Field(gt=0)]
    expected_markers: Annotated[int, Field(ge=0)]
    detected_markers: Annotated[int, Field(ge=0)]
    true_positives: Annotated[int, Field(ge=0)]
    false_positives: Annotated[int, Field(ge=0)]
    false_negatives: Annotated[int, Field(ge=0)]
    duplicate_markers: Annotated[int, Field(ge=0)]
    precision: UnitInterval
    recall: UnitInterval
    f1: UnitInterval
    mean_test_number_accuracy: UnitInterval
    mean_ordering_accuracy: UnitInterval

"""Typed immutable models for teacher-focused evidence candidate discovery."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ocr.models import BoundingBox

UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]


class TeacherDiscoveryCategory(StrEnum):
    """Sampling strata only; none is a human evidence label."""

    CHROMATIC = "chromatic_ink_risk"
    MARGIN_SCORE = "margin_score_risk"
    COMPACT_GEOMETRY = "tick_cross_correction_risk"
    MIXED = "mixed_teacher_context_risk"
    AMBIGUOUS = "ambiguous_mark_risk"
    HARD_NEGATIVE = "hard_negative_control"


class TeacherDiscoverySignal(StrEnum):
    """Local visual hints retained as candidate provenance, never labels."""

    CHROMATIC_INK = "chromatic_ink"
    SMALL_ISOLATED_COMPONENT = "small_isolated_component"
    MARGIN_ACTIVITY = "margin_activity"
    SCORE_LIKE_GEOMETRY = "score_like_geometry"
    TICK_CROSS_GEOMETRY = "tick_cross_geometry"
    CORRECTION_STROKE = "correction_stroke_risk"
    OCR_CONTEXT_NEARBY = "ocr_context_nearby"
    HIGH_CONFIDENCE_PRINT_NEARBY = "high_confidence_print_nearby"
    STUDENT_TEACHER_MIXED_RISK = "student_teacher_mixed_risk"
    PRINT_TEACHER_MIXED_RISK = "print_teacher_mixed_risk"
    PARAGRAPH_CONTEXT = "paragraph_teacher_risk"
    SHORT_ANSWER_CONTEXT = "short_answer_teacher_risk"
    TEST_STRUCTURE_CONTEXT = "test_structure_context"
    PRINTED_CONTROL = "printed_or_guide_hard_negative"


class TeacherCandidateFeatures(BaseModel):
    """Normalized discovery measurements without an authorship conclusion."""

    model_config = ConfigDict(frozen=True)

    component_area_ratio: UnitInterval
    chromatic_foreground_ratio: UnitInterval
    mean_saturation: UnitInterval
    foreground_ratio: UnitInterval
    edge_density: UnitInterval
    local_whitespace_ratio: UnitInterval
    margin_proximity: UnitInterval
    ocr_proximity: UnitInterval
    nearby_ocr_words: Annotated[int, Field(ge=0)]
    angled_line_count: Annotated[int, Field(ge=0)]


class TeacherEvidenceSample(BaseModel):
    """One private, unlabeled crop awaiting human visual inspection."""

    model_config = ConfigDict(frozen=True)

    sample_id: Annotated[str, Field(pattern=r"^evidence_teacher_v1_[0-9]{3}$")]
    paper_alias: Annotated[str, Field(pattern=r"^paper-[a-z]{1,2}$")]
    page_number: Annotated[int, Field(gt=0)]
    test_number: Annotated[int, Field(ge=1, le=99)] | None = None
    source_image_path: Path
    source_image_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    page_width: Annotated[int, Field(gt=0)]
    page_height: Annotated[int, Field(gt=0)]
    region: BoundingBox
    candidate_component: BoundingBox
    discovery_category: TeacherDiscoveryCategory
    discovery_signals: Annotated[
        tuple[TeacherDiscoverySignal, ...], Field(min_length=1)
    ]
    features: TeacherCandidateFeatures
    selection_rank: Annotated[int, Field(gt=0)]
    discovery_reason: Annotated[str, Field(pattern=r"^[a-z0-9_+.-]{3,120}$")]
    candidate_generation_version: Literal["teacher-risk-discovery-v1"] = (
        "teacher-risk-discovery-v1"
    )

    @model_validator(mode="after")
    def validate_geometry_and_provenance(self) -> Self:
        page = BoundingBox(x=0, y=0, width=self.page_width, height=self.page_height)
        if not _contains(page, self.region):
            raise ValueError("Teacher candidate crop exceeds its canonical page")
        if not _contains(self.region, self.candidate_component):
            raise ValueError("Teacher candidate component must fit its context crop")
        if len(self.discovery_signals) != len(set(self.discovery_signals)):
            raise ValueError("Teacher discovery signals must be unique")
        if not self.source_image_path.is_absolute():
            raise ValueError("Teacher candidate source image path must be absolute")
        return self

    @property
    def categories(self) -> tuple[StrEnum, ...]:
        """Expose neutral discovery hints to the generic local labeler."""

        return (self.discovery_category, *self.discovery_signals)


class TeacherEvidenceManifest(BaseModel):
    """Separate versioned candidate set with no embedded human labels."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: Literal["evidence-teacher-v1"] = "evidence-teacher-v1"
    candidate_generation_version: Literal["teacher-risk-discovery-v1"] = (
        "teacher-risk-discovery-v1"
    )
    samples: Annotated[tuple[TeacherEvidenceSample, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def identifiers_are_unique_and_ordered(self) -> Self:
        identifiers = [sample.sample_id for sample in self.samples]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Teacher evidence sample IDs must be unique")
        if identifiers != sorted(identifiers):
            raise ValueError(
                "Teacher evidence samples must be deterministically ordered"
            )
        return self

    @property
    def pending_count(self) -> int:
        return len(self.samples)

    @property
    def benchmark_ready(self) -> bool:
        return False


def _contains(container: BoundingBox, child: BoundingBox) -> bool:
    return (
        child.x >= container.x
        and child.y >= container.y
        and child.x + child.width <= container.x + container.width
        and child.y + child.height <= container.y + container.height
    )

"""Typed, immutable models for deterministic evidence separation V2."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.evidence.models import EvidenceRegion, EvidenceType, StudentAnswerRegion
from app.ocr.models import BoundingBox

UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]


class EvidenceReasonV2(StrEnum):
    """Machine-readable evidence contributing to a V2 decision."""

    REGULAR_OCR_LINE = "regular_ocr_line"
    HIGH_OCR_SUPPORT = "high_ocr_support"
    DENSE_ALIGNED_TEXT = "dense_aligned_text"
    IRREGULAR_GEOMETRY = "irregular_geometry"
    FRAGMENTED_COMPONENTS = "fragmented_components"
    WRITING_SPACE_CONTEXT = "writing_space_context"
    CONTIGUOUS_WRITING_GROUP = "contiguous_writing_group"
    CHROMATIC_DIFFERENCE = "chromatic_difference"
    ISOLATED_SMALL_MARK = "isolated_small_mark"
    MARGIN_ANNOTATION = "margin_annotation"
    TICK_OR_CROSS_SHAPE = "tick_or_cross_shape"
    SCORE_LIKE_GEOMETRY = "score_like_geometry"
    CORRECTION_STROKE = "correction_stroke"
    CONTRADICTORY_AUTHORSHIP = "contradictory_authorship"
    WEAK_EVIDENCE = "weak_evidence"
    NARROW_SCORE_MARGIN = "narrow_score_margin"


class EvidenceSeparatorV2Config(BaseModel):
    """Centralized, serializable V2 thresholds for a controlled experiment."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["2.0"] = "2.0"
    variant: Annotated[str, Field(pattern=r"^v2-[a-d]$")]
    enable_grouping: bool = True
    enable_teacher_risk: bool = True
    enable_ambiguity_margin: bool = True
    enable_paragraph_merging: bool = True
    group_horizontal_gap_ratio: Annotated[float, Field(gt=0, le=8)] = 2.2
    group_vertical_gap_ratio: Annotated[float, Field(gt=0, le=5)] = 0.9
    group_min_vertical_overlap: UnitInterval = 0.30
    maximum_group_area_ratio: UnitInterval = 0.30
    printed_threshold: UnitInterval = 0.58
    student_threshold: UnitInterval = 0.46
    teacher_threshold: UnitInterval = 0.52
    minimum_decision_score: UnitInterval = 0.42
    ambiguity_margin: UnitInterval = 0.08
    teacher_suppression_threshold: UnitInterval = 0.46
    answer_padding_ratio: Annotated[float, Field(ge=0, le=1)] = 0.18
    answer_merge_gap_ratio: Annotated[float, Field(ge=0, le=5)] = 0.65
    maximum_answer_area_ratio: UnitInterval = 0.28

    def fingerprint(self) -> str:
        """Hash canonical semantic configuration, independent of formatting."""

        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


class EvidenceComponent(BaseModel):
    """One V1 evidence candidate retained as immutable V2 input provenance."""

    model_config = ConfigDict(frozen=True)

    component_index: Annotated[int, Field(ge=0)]
    bbox: BoundingBox
    source_word_indices: tuple[Annotated[int, Field(ge=0)], ...] = ()
    base_region: EvidenceRegion


class EvidenceGroup(BaseModel):
    """Conservatively grouped components that may represent one visual unit."""

    model_config = ConfigDict(frozen=True)

    bbox: BoundingBox
    component_indices: Annotated[tuple[int, ...], Field(min_length=1)]
    components: Annotated[tuple[EvidenceComponent, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def components_match_indices_and_geometry(self) -> Self:
        if self.component_indices != tuple(
            item.component_index for item in self.components
        ):
            raise ValueError("Evidence group component indices must match")
        if len(self.component_indices) != len(set(self.component_indices)):
            raise ValueError("Evidence group component indices must be unique")
        if any(not _contains(self.bbox, item.bbox) for item in self.components):
            raise ValueError("Evidence group must contain every component")
        return self


class EvidenceFeaturesV2(BaseModel):
    """Provider-independent aggregate color, geometry, OCR, and context evidence."""

    model_config = ConfigDict(frozen=True)

    mean_saturation: UnitInterval
    chromatic_pixel_ratio: UnitInterval
    color_consistency: UnitInterval
    foreground_ratio: UnitInterval
    ink_background_contrast: UnitInterval
    component_density: UnitInterval
    compactness: UnitInterval
    fragmentation: UnitInterval
    alignment: UnitInterval
    isolated_small_mark_likelihood: UnitInterval
    diagonal_dominance: UnitInterval
    crossing_stroke_likelihood: UnitInterval
    underline_likelihood: UnitInterval
    circle_likelihood: UnitInterval
    correction_stroke_likelihood: UnitInterval
    ocr_confidence: UnitInterval | None = None
    ocr_token_count: Annotated[int, Field(ge=0)]
    ocr_regularity: UnitInterval
    hierarchy_consistency: UnitInterval
    text_density: UnitInterval
    normalized_x: UnitInterval
    normalized_y: UnitInterval
    margin_proximity: UnitInterval
    answer_space_overlap: UnitInterval
    group_width_ratio: UnitInterval
    group_height_ratio: UnitInterval


class EvidenceScoresV2(BaseModel):
    """Explainable rule scores; values are not calibrated probabilities."""

    model_config = ConfigDict(frozen=True)

    printed: UnitInterval
    student: UnitInterval
    teacher: UnitInterval
    ambiguity: UnitInterval


class EvidenceDecisionV2(BaseModel):
    """A deterministic V2 attribution with score and reason provenance."""

    model_config = ConfigDict(frozen=True)

    evidence_type: EvidenceType
    scores: EvidenceScoresV2
    reasons: tuple[EvidenceReasonV2, ...]
    strategy_version: Literal["evidence-separation-v2"] = "evidence-separation-v2"

    @field_validator("reasons")
    @classmethod
    def reasons_are_unique(
        cls, value: tuple[EvidenceReasonV2, ...]
    ) -> tuple[EvidenceReasonV2, ...]:
        if len(value) != len(set(value)):
            raise ValueError("V2 decision reasons must be unique")
        return value


class AnswerSpace(BaseModel):
    """A place where an answer could be written, not detected student evidence."""

    model_config = ConfigDict(frozen=True)

    paper_id: UUID
    page_number: Annotated[int, Field(gt=0)]
    test_number: Annotated[int, Field(ge=1, le=99)]
    bbox: BoundingBox
    confidence: UnitInterval
    source_image_path: Path
    detection_strategy: Literal["answer-space-v2"] = "answer-space-v2"

    @field_validator("source_image_path")
    @classmethod
    def source_is_absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("Answer-space source path must be absolute")
        return value


class TestEvidenceV2(BaseModel):
    """V2 evidence, answer spaces, and actual student-answer candidates."""

    model_config = ConfigDict(frozen=True)

    paper_id: UUID
    page_number: Annotated[int, Field(gt=0)]
    test_number: Annotated[int, Field(ge=1, le=99)]
    region_bbox: BoundingBox
    groups: tuple[EvidenceGroup, ...] = ()
    group_features: tuple[EvidenceFeaturesV2, ...] = ()
    decisions: tuple[EvidenceDecisionV2, ...] = ()
    evidence_regions: tuple[EvidenceRegion, ...] = ()
    answer_spaces: tuple[AnswerSpace, ...] = ()
    student_answer_regions: tuple[StudentAnswerRegion, ...] = ()
    configuration_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def children_fit_region(self) -> Self:
        if not (
            len(self.groups)
            == len(self.group_features)
            == len(self.decisions)
            == len(self.evidence_regions)
        ):
            raise ValueError("V2 groups, features, decisions, and regions must align")
        invalid = (
            any(
                item.paper_id != self.paper_id
                or item.page_number != self.page_number
                or item.test_number != self.test_number
                or not _contains(self.region_bbox, item.bbox)
                for item in self.evidence_regions
            )
            or any(
                item.paper_id != self.paper_id
                or item.page_number != self.page_number
                or item.test_number != self.test_number
                or not _contains(self.region_bbox, item.bbox)
                for item in self.answer_spaces
            )
            or any(
                item.paper_id != self.paper_id
                or item.page_number != self.page_number
                or item.test_number != self.test_number
                or not _contains(self.region_bbox, item.bbox)
                for item in self.student_answer_regions
            )
        )
        if invalid:
            raise ValueError("V2 evidence children must match and fit their Test")
        return self


def _contains(container: BoundingBox, child: BoundingBox) -> bool:
    return (
        child.x >= container.x
        and child.y >= container.y
        and child.x + child.width <= container.x + container.width
        and child.y + child.height <= container.y + container.height
    )

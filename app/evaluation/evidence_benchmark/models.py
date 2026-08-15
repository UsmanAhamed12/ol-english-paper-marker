"""Typed private candidates awaiting human evidence and answer-region labels."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.evidence.models import EvidenceType
from app.ocr.models import BoundingBox

SafeIdentifier = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")]


class EvidenceSampleCategory(StrEnum):
    """Selection coverage, not a human evidence label."""

    MOSTLY_PRINTED = "mostly_printed_candidate"
    CLEAR_HANDWRITING = "clear_handwriting_candidate"
    DIFFICULT_HANDWRITING = "difficult_handwriting_candidate"
    TEACHER_MARK = "teacher_mark_candidate"
    TEACHER_SCORE = "teacher_score_candidate"
    MIXED = "mixed_evidence_candidate"
    PARAGRAPH = "paragraph_answer_candidate"
    SHORT_ANSWER = "short_answer_candidate"
    BLANK_ANSWER = "blank_answer_candidate"


class HumanEvidenceStatus(StrEnum):
    """Explicit human-label readiness state."""

    PENDING = "pending"
    HUMAN_VERIFIED = "human_verified"


class GroundTruthAnswerRegion(BaseModel):
    """Human-labeled answer box in sample-crop coordinates."""

    model_config = ConfigDict(frozen=True)

    bbox: BoundingBox


class EvidenceBenchmarkSample(BaseModel):
    """One private crop requiring a conservative human label."""

    model_config = ConfigDict(frozen=True)

    sample_id: Annotated[str, Field(pattern=r"^(?:sample|evidence_v2)_[0-9]{3}$")]
    paper_alias: SafeIdentifier
    page_number: Annotated[int, Field(gt=0)]
    test_number: Annotated[int, Field(ge=1, le=99)]
    source_image_path: Path
    page_width: Annotated[int, Field(gt=0)]
    page_height: Annotated[int, Field(gt=0)]
    region: BoundingBox
    categories: Annotated[tuple[EvidenceSampleCategory, ...], Field(min_length=1)]
    human_status: HumanEvidenceStatus = HumanEvidenceStatus.PENDING
    ground_truth_evidence_type: EvidenceType | None = None
    answer_regions_verified: bool = False
    ground_truth_answer_regions: tuple[GroundTruthAnswerRegion, ...] = ()

    @model_validator(mode="after")
    def validate_private_ground_truth_state(self) -> Self:
        if (
            self.region.x + self.region.width > self.page_width
            or self.region.y + self.region.height > self.page_height
        ):
            raise ValueError("Evidence benchmark region exceeds source page")
        if len(self.categories) != len(set(self.categories)):
            raise ValueError("Evidence benchmark categories must be unique")
        if any(
            answer.bbox.x + answer.bbox.width > self.region.width
            or answer.bbox.y + answer.bbox.height > self.region.height
            for answer in self.ground_truth_answer_regions
        ):
            raise ValueError("Ground-truth answer box exceeds sample crop")
        if self.human_status is HumanEvidenceStatus.PENDING:
            if self.ground_truth_evidence_type is not None:
                raise ValueError(
                    "Pending evidence sample cannot have a ground-truth label"
                )
            if self.answer_regions_verified or self.ground_truth_answer_regions:
                raise ValueError(
                    "Pending evidence sample cannot have answer ground truth"
                )
        elif (
            self.ground_truth_evidence_type is None or not self.answer_regions_verified
        ):
            raise ValueError(
                "Human-verified sample requires both evidence and answer labels"
            )
        return self


class EvidenceBenchmarkManifest(BaseModel):
    """Private benchmark that remains unready until every sample is verified."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    samples: Annotated[tuple[EvidenceBenchmarkSample, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def samples_are_unique(self) -> Self:
        identifiers = [sample.sample_id for sample in self.samples]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Evidence benchmark sample IDs must be unique")
        return self

    @property
    def benchmark_ready(self) -> bool:
        return all(
            sample.human_status is HumanEvidenceStatus.HUMAN_VERIFIED
            for sample in self.samples
        )

    @property
    def pending_count(self) -> int:
        return sum(
            sample.human_status is HumanEvidenceStatus.PENDING
            for sample in self.samples
        )

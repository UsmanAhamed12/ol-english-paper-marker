"""Typed models for the private, human-labeled evidence-v2 dataset."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ocr.models import BoundingBox


class EvidenceCandidateCategory(StrEnum):
    """Discovery hints used for sampling, never human ground truth."""

    PRINTED = "printed_candidate"
    STUDENT = "student_candidate"
    TEACHER = "teacher_mark_risk_candidate"
    MIXED = "mixed_uncertain_candidate"
    BLANK = "blank_answer_candidate"


class EvidenceContextTag(StrEnum):
    """Safe visual-coverage metadata shown without asserting authorship."""

    SHORT_ANSWER = "short_answer"
    PARAGRAPH = "paragraph"
    DIFFICULT_HANDWRITING = "difficult_handwriting_risk"
    COLORED_INK = "colored_ink_present"
    DENSE = "dense_region"
    SPARSE = "sparse_region"
    WRITING_GUIDES = "writing_guides_present"
    MARGIN = "margin_region"


class EvidenceExpansionSample(BaseModel):
    """One immutable private crop awaiting a fresh human decision."""

    model_config = ConfigDict(frozen=True)

    sample_id: Annotated[str, Field(pattern=r"^evidence_v2_[0-9]{3}$")]
    paper_alias: Annotated[str, Field(pattern=r"^paper-[a-z]{1,2}$")]
    page_number: Annotated[int, Field(gt=0)]
    test_number: Annotated[int, Field(ge=1, le=99)] | None = None
    source_image_path: Path
    source_image_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    page_width: Annotated[int, Field(gt=0)]
    page_height: Annotated[int, Field(gt=0)]
    region: BoundingBox
    discovery_category: EvidenceCandidateCategory
    context_tags: tuple[EvidenceContextTag, ...] = ()
    discovery_reason: Annotated[str, Field(pattern=r"^[a-z0-9_+.-]{3,80}$")]

    @model_validator(mode="after")
    def region_is_page_local(self) -> Self:
        if (
            self.region.x + self.region.width > self.page_width
            or self.region.y + self.region.height > self.page_height
        ):
            raise ValueError("Evidence-v2 crop exceeds its canonical page")
        if len(self.context_tags) != len(set(self.context_tags)):
            raise ValueError("Evidence-v2 context tags must be unique")
        if not self.source_image_path.is_absolute():
            raise ValueError("Evidence-v2 source image path must be absolute")
        return self

    @property
    def categories(self) -> tuple[EvidenceContextTag, ...]:
        """Expose only non-label context tags to the generic local UI."""

        return self.context_tags


class EvidenceExpansionManifest(BaseModel):
    """Versioned candidates; annotations live in a separate private store."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["2.0"] = "2.0"
    benchmark_name: Literal["evidence_v2"] = "evidence_v2"
    samples: Annotated[tuple[EvidenceExpansionSample, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def identifiers_are_unique_and_ordered(self) -> Self:
        identifiers = [sample.sample_id for sample in self.samples]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Evidence-v2 sample IDs must be unique")
        if identifiers != sorted(identifiers):
            raise ValueError("Evidence-v2 samples must use deterministic ordering")
        return self

    @property
    def pending_count(self) -> int:
        return len(self.samples)

    @property
    def benchmark_ready(self) -> bool:
        return False

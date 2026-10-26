"""Immutable models for Test-owned student-answer extraction."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.evidence.safety_models import SafetyReason
from app.ocr.models import BoundingBox, OCRWarningCode
from app.structure.models import MarkerDetectionStrategy

UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
DurationMilliseconds = Annotated[float, Field(ge=0.0)]
PositiveTestNumber = Annotated[int, Field(ge=1, le=99)]


class AnswerExtractionStatus(StrEnum):
    """Trust state of one Test-owned answer."""

    EXTRACTED = "extracted"
    EMPTY = "empty"
    REVIEW_REQUIRED = "review_required"
    OCR_FAILED = "ocr_failed"
    STRUCTURE_INCOMPLETE = "structure_incomplete"


class AnswerFragmentStatus(StrEnum):
    """Outcome of OCR for one safe source region."""

    EXTRACTED = "extracted"
    OCR_FAILED = "ocr_failed"


class AnswerExtractionWarning(StrEnum):
    """Typed non-fatal extraction concerns safe to serialize and display."""

    REVIEW_EVIDENCE_PRESENT = "review_evidence_present"
    OCR_REGION_FAILED = "ocr_region_failed"
    OCR_RETURNED_EMPTY = "ocr_returned_empty"
    STRUCTURE_EVIDENCE_MISSING = "structure_evidence_missing"
    HISTORICAL_TEACHER_ANNOTATION_RISK = "historical_teacher_annotation_risk"


class AnswerSourceContext(StrEnum):
    """Whether source pages are normal runtime inputs or marked evaluation data."""

    RUNTIME_UNMARKED = "runtime_unmarked"
    HISTORICAL_MARKED_EVALUATION = "historical_marked_evaluation"


class AnswerSourceRegion(BaseModel):
    """Canonical geometry and derived crop provenance for safe evidence."""

    model_config = ConfigDict(frozen=True)

    region_index: Annotated[int, Field(ge=0)]
    paper_id: UUID
    test_number: PositiveTestNumber
    page_number: Annotated[int, Field(gt=0)]
    bbox: BoundingBox
    evidence_confidence: UnitInterval
    source_evidence_indices: tuple[Annotated[int, Field(ge=0)], ...] = ()
    canonical_image_path: Path
    canonical_image_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    derived_image_path: Path
    detection_strategy: Annotated[str, Field(min_length=1)]

    @field_validator("canonical_image_path", "derived_image_path")
    @classmethod
    def paths_are_absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("Answer source paths must be absolute")
        return value


class ReviewEvidenceRegion(BaseModel):
    """Quarantined evidence retained without entering trusted answer text."""

    model_config = ConfigDict(frozen=True)

    paper_id: UUID
    test_number: PositiveTestNumber
    page_number: Annotated[int, Field(gt=0)]
    bbox: BoundingBox
    reasons: Annotated[tuple[SafetyReason, ...], Field(min_length=1)]
    source_image_path: Path

    @field_validator("source_image_path")
    @classmethod
    def source_path_is_absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("Review evidence source path must be absolute")
        return value


class AnswerFragment(BaseModel):
    """One ordered OCR result tied to exactly one safe source region."""

    model_config = ConfigDict(frozen=True)

    fragment_index: Annotated[int, Field(ge=0)]
    source_region_index: Annotated[int, Field(ge=0)]
    paper_id: UUID
    test_number: PositiveTestNumber
    page_number: Annotated[int, Field(gt=0)]
    bbox: BoundingBox
    status: AnswerFragmentStatus
    raw_text: str = ""
    normalized_text: str = ""
    confidence: UnitInterval | None = None
    provider: Annotated[str, Field(min_length=1)]
    model_version: Annotated[str, Field(min_length=1)] | None = None
    processing_duration_ms: DurationMilliseconds
    warnings: tuple[OCRWarningCode, ...] = ()
    error_code: Annotated[str, Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def failure_has_no_fake_text(self) -> Self:
        if self.status is AnswerFragmentStatus.OCR_FAILED:
            if self.raw_text or self.normalized_text or self.error_code is None:
                raise ValueError("Failed OCR fragments require an error and no text")
        elif self.error_code is not None:
            raise ValueError("Successful OCR fragments cannot contain an error")
        return self


class AnswerExtractionProvenance(BaseModel):
    """Structure, safety, and OCR versions used for one answer."""

    model_config = ConfigDict(frozen=True)

    extraction_version: Annotated[str, Field(min_length=1)]
    test_label: Annotated[str, Field(pattern=r"^Test [0-9]{2}$")]
    marker_page_number: Annotated[int, Field(gt=0)]
    marker_bbox: BoundingBox
    marker_strategy: MarkerDetectionStrategy
    marker_confidence: UnitInterval
    marker_source_word_indices: Annotated[tuple[int, ...], Field(min_length=1)]
    safety_policy_versions: Annotated[tuple[str, ...], Field(min_length=1)]
    ocr_provider: Annotated[str, Field(min_length=1)]
    ocr_model_version: Annotated[str, Field(min_length=1)] | None = None
    source_context: AnswerSourceContext


class StructuredStudentAnswer(BaseModel):
    """Stable Test-level answer contract consumed by later marking phases."""

    model_config = ConfigDict(frozen=True)

    paper_id: UUID
    test_number: PositiveTestNumber
    status: AnswerExtractionStatus
    text: str
    pages: Annotated[tuple[int, ...], Field(min_length=1)]
    source_regions: tuple[AnswerSourceRegion, ...] = ()
    ocr_fragments: tuple[AnswerFragment, ...] = ()
    review_regions: tuple[ReviewEvidenceRegion, ...] = ()
    evidence_confidence: UnitInterval | None = None
    ocr_confidence: UnitInterval | None = None
    warnings: tuple[AnswerExtractionWarning, ...] = ()
    provenance: AnswerExtractionProvenance
    processing_duration_ms: DurationMilliseconds

    @model_validator(mode="after")
    def answer_state_is_consistent(self) -> Self:
        if list(self.pages) != sorted(set(self.pages)):
            raise ValueError("Answer pages must be unique and ordered")
        provenance = (
            tuple(
                (item.paper_id, item.test_number, item.page_number)
                for item in self.source_regions
            )
            + tuple(
                (item.paper_id, item.test_number, item.page_number)
                for item in self.ocr_fragments
            )
            + tuple(
                (item.paper_id, item.test_number, item.page_number)
                for item in self.review_regions
            )
        )
        if any(
            paper_id != self.paper_id or test_number != self.test_number
            for paper_id, test_number, _ in provenance
        ):
            raise ValueError("Answer children must match Test provenance")
        if any(page_number not in self.pages for _, _, page_number in provenance):
            raise ValueError("Answer children must belong to an answer page")
        if [item.fragment_index for item in self.ocr_fragments] != list(
            range(len(self.ocr_fragments))
        ):
            raise ValueError("Answer fragments must have contiguous ordered indices")
        if self.status is AnswerExtractionStatus.EMPTY and (
            self.text
            or self.source_regions
            or self.ocr_fragments
            or self.review_regions
        ):
            raise ValueError("Empty answers cannot contain extracted evidence")
        if self.status is AnswerExtractionStatus.EXTRACTED and not any(
            item.status is AnswerFragmentStatus.EXTRACTED and bool(item.normalized_text)
            for item in self.ocr_fragments
        ):
            raise ValueError("Extracted answers require non-empty OCR text")
        return self


class PaperExtractionProvenance(BaseModel):
    """Stable paper-level extraction configuration."""

    model_config = ConfigDict(frozen=True)

    extraction_version: Annotated[str, Field(min_length=1)]
    structure_test_count: Annotated[int, Field(ge=0)]
    source_page_count: Annotated[int, Field(gt=0)]
    ocr_provider: Annotated[str, Field(min_length=1)]
    ocr_model_version: Annotated[str, Field(min_length=1)] | None = None
    source_context: AnswerSourceContext


class StructuredStudentPaper(BaseModel):
    """Ordered answers and explicit paper-level incomplete states."""

    model_config = ConfigDict(frozen=True)

    paper_id: UUID
    answers: tuple[StructuredStudentAnswer, ...]
    detected_test_numbers: tuple[PositiveTestNumber, ...]
    missing_test_numbers: tuple[PositiveTestNumber, ...] = ()
    review_required_test_numbers: tuple[PositiveTestNumber, ...] = ()
    empty_test_numbers: tuple[PositiveTestNumber, ...] = ()
    ocr_failed_test_numbers: tuple[PositiveTestNumber, ...] = ()
    structure_incomplete_test_numbers: tuple[PositiveTestNumber, ...] = ()
    total_processing_duration_ms: DurationMilliseconds
    provenance: PaperExtractionProvenance

    @model_validator(mode="after")
    def paper_summary_matches_answers(self) -> Self:
        numbers = tuple(item.test_number for item in self.answers)
        if numbers != tuple(sorted(set(numbers))):
            raise ValueError("Paper answers must be unique and numerically ordered")
        if numbers != self.detected_test_numbers:
            raise ValueError("Detected Test summary must match answer objects")
        expected = {
            AnswerExtractionStatus.REVIEW_REQUIRED: self.review_required_test_numbers,
            AnswerExtractionStatus.EMPTY: self.empty_test_numbers,
            AnswerExtractionStatus.OCR_FAILED: self.ocr_failed_test_numbers,
            AnswerExtractionStatus.STRUCTURE_INCOMPLETE: (
                self.structure_incomplete_test_numbers
            ),
        }
        for status, summary in expected.items():
            actual = tuple(
                item.test_number for item in self.answers if item.status is status
            )
            if actual != summary:
                raise ValueError("Paper status summary does not match answers")
        if set(self.missing_test_numbers) & set(self.detected_test_numbers):
            raise ValueError("Missing Tests cannot also be detected")
        return self

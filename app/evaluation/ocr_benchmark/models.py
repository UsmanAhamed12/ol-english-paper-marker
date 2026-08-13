"""Typed OCR benchmark manifests, results, and summaries."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.ocr.models import OCRWarningCode

SafeIdentifier = Annotated[
    str,
    Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$"),
]
NonBlankString = Annotated[str, Field(min_length=1)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class BenchmarkDifficulty(StrEnum):
    """Human-assigned transcription difficulty for stratified reporting."""

    CLEAR = "clear"
    MEDIUM = "medium"
    DIFFICULT = "difficult"


class GroundTruthStatus(StrEnum):
    """Whether student text has been manually verified for scoring."""

    PENDING = "pending_manual_transcription"
    VERIFIED = "human_verified"


class BenchmarkStatus(StrEnum):
    """Outcome of one provider attempt."""

    SUCCESS = "success"
    FAILURE = "failure"


class BenchmarkRegion(BaseModel):
    """Optional pixel rectangle within the canonical page image."""

    model_config = ConfigDict(frozen=True)

    x: Annotated[int, Field(ge=0)]
    y: Annotated[int, Field(ge=0)]
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]


class OCRBenchmarkSample(BaseModel):
    """Private manifest entry targeting student answer text only."""

    model_config = ConfigDict(frozen=True)

    sample_id: SafeIdentifier
    paper_alias: SafeIdentifier
    page_number: Annotated[int, Field(gt=0)]
    image_path: Path
    image_width: Annotated[int, Field(gt=0)]
    image_height: Annotated[int, Field(gt=0)]
    region: BenchmarkRegion | None = None
    difficulty: BenchmarkDifficulty
    categories: Annotated[tuple[SafeIdentifier, ...], Field(min_length=1)]
    printed_content_present: bool
    teacher_annotations_present: bool
    transcription_target: Literal["student_answer_text"] = "student_answer_text"
    ground_truth_status: GroundTruthStatus
    ground_truth_student_text: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def validate_ground_truth_and_region(self) -> OCRBenchmarkSample:
        """Require human evidence for scoring and keep regions within the page."""

        if (
            self.ground_truth_status is GroundTruthStatus.VERIFIED
            and self.ground_truth_student_text is None
        ):
            raise ValueError("verified ground truth requires a manual transcription")
        if (
            self.ground_truth_status is GroundTruthStatus.PENDING
            and self.ground_truth_student_text is not None
        ):
            raise ValueError("pending ground truth must not contain a transcription")
        if self.region is not None and (
            self.region.x + self.region.width > self.image_width
            or self.region.y + self.region.height > self.image_height
        ):
            raise ValueError("benchmark region must fit within the page image")
        return self

    @property
    def is_ready(self) -> bool:
        """Return whether this sample may be used for metric scoring."""

        return (
            self.ground_truth_status is GroundTruthStatus.VERIFIED
            and self.ground_truth_student_text is not None
        )


class BenchmarkManifest(BaseModel):
    """Versioned collection of private benchmark sample definitions."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    samples: Annotated[tuple[OCRBenchmarkSample, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def sample_ids_must_be_unique(self) -> BenchmarkManifest:
        """Prevent results from becoming ambiguous across duplicate samples."""

        sample_ids = [sample.sample_id for sample in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("benchmark sample IDs must be unique")
        return self


class ErrorRate(BaseModel):
    """Edit count, denominator, and possibly undefined normalized rate."""

    model_config = ConfigDict(frozen=True)

    errors: Annotated[int, Field(ge=0)]
    reference_units: Annotated[int, Field(ge=0)]
    rate: NonNegativeFloat | None

    @model_validator(mode="after")
    def validate_empty_reference_semantics(self) -> ErrorRate:
        """Make zero-denominator behavior explicit and internally consistent."""

        if self.reference_units > 0 and self.rate is None:
            raise ValueError("rate is required when reference units are present")
        if self.reference_units == 0:
            expected_rate = 0.0 if self.errors == 0 else None
            if self.rate != expected_rate:
                raise ValueError("invalid empty-reference rate")
        return self


class OCRBenchmarkResult(BaseModel):
    """Validated outcome for one provider, prompt, and benchmark sample."""

    model_config = ConfigDict(frozen=True)

    sample_id: SafeIdentifier
    provider: NonBlankString
    model_version: str | None = None
    ocr_prompt_version: NonBlankString
    status: BenchmarkStatus
    prediction: str | None = None
    cer: ErrorRate | None = None
    wer: ErrorRate | None = None
    duration_ms: NonNegativeFloat | None = None
    warnings: tuple[OCRWarningCode, ...] = ()
    teacher_annotation_contamination: bool | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> OCRBenchmarkResult:
        """Keep successful measurements distinct from provider failures."""

        success_fields = (self.prediction, self.cer, self.wer, self.duration_ms)
        if self.status is BenchmarkStatus.SUCCESS:
            if any(value is None for value in success_fields) or self.error is not None:
                raise ValueError("successful results require metrics and no error")
        elif self.error is None or any(value is not None for value in success_fields):
            raise ValueError("failed results require an error and no metrics")
        return self


class OCRBenchmarkSummary(BaseModel):
    """Aggregate metrics that retain explicit success and failure counts."""

    model_config = ConfigDict(frozen=True)

    total_samples: Annotated[int, Field(ge=0)]
    successful_samples: Annotated[int, Field(ge=0)]
    failed_samples: Annotated[int, Field(ge=0)]
    mean_cer: NonNegativeFloat | None
    median_cer: NonNegativeFloat | None
    mean_wer: NonNegativeFloat | None
    median_wer: NonNegativeFloat | None
    mean_processing_duration_ms: NonNegativeFloat | None

    @model_validator(mode="after")
    def counts_must_balance(self) -> OCRBenchmarkSummary:
        """Ensure failures cannot disappear from aggregate reporting."""

        if self.successful_samples + self.failed_samples != self.total_samples:
            raise ValueError("successful and failed counts must equal total samples")
        return self

"""Typed immutable records for controlled V2 evidence experiments."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.evaluation.evidence_benchmark.evaluation import EvidenceBenchmarkEvaluation
from app.evaluation.evidence_benchmark.metrics import ClassificationMetrics
from app.evidence.v2_models import EvidenceSeparatorV2Config

DatasetId = Literal["phase_4c_4r", "evidence_v2", "evidence_teacher_v1"]


class EvidenceV2BenchmarkResult(BaseModel):
    """Safe aggregate result for one independently reported benchmark."""

    model_config = ConfigDict(frozen=True)

    dataset_id: DatasetId
    annotation_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    evaluation: EvidenceBenchmarkEvaluation
    teacher_as_student_count: Annotated[int, Field(ge=0)]
    teacher_as_student_rate: Annotated[float, Field(ge=0, le=1)] | None
    teacher_samples_with_student_regions: Annotated[int, Field(ge=0)]
    student_regions_on_teacher_samples: Annotated[int, Field(ge=0)]
    mean_predicted_region_area_ratio: Annotated[float, Field(ge=0, le=1)]


class EvidenceV2Experiment(BaseModel):
    """Non-overwriting reproducibility record for one controlled configuration."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    experiment_version: Literal["phase_4c_6"] = "phase_4c_6"
    configuration: EvidenceSeparatorV2Config
    configuration_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    code_version: Annotated[str, Field(min_length=1)]
    benchmark_fingerprints: dict[str, Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]]
    created_at_utc: datetime
    selected: bool
    benchmark_results: tuple[EvidenceV2BenchmarkResult, ...]
    combined_classification: ClassificationMetrics

    @model_validator(mode="after")
    def configuration_and_benchmarks_are_complete(self) -> EvidenceV2Experiment:
        if self.configuration.fingerprint() != self.configuration_hash:
            raise ValueError("V2 experiment configuration hash does not match")
        identifiers = [item.dataset_id for item in self.benchmark_results]
        if identifiers != ["phase_4c_4r", "evidence_v2", "evidence_teacher_v1"]:
            raise ValueError("V2 experiment must report all frozen benchmarks")
        if set(self.benchmark_fingerprints) != set(identifiers):
            raise ValueError("V2 experiment fingerprints must cover every benchmark")
        return self

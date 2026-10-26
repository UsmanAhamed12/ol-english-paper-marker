"""Typed aggregate records for the Phase 4C.6R safety experiment."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.evidence_benchmark.evaluation import EvidenceBenchmarkEvaluation
from app.evaluation.evidence_redesign.models import DatasetId


class SafetyFilterBenchmarkResult(BaseModel):
    """Privacy-safe result for one frozen evidence benchmark."""

    model_config = ConfigDict(frozen=True)

    dataset_id: DatasetId
    annotation_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    evaluation: EvidenceBenchmarkEvaluation
    safe_tests: Annotated[int, Field(ge=0)]
    review_required_tests: Annotated[int, Field(ge=0)]
    empty_tests: Annotated[int, Field(ge=0)]
    safe_candidates: Annotated[int, Field(ge=0)]
    quarantined_candidates: Annotated[int, Field(ge=0)]
    rejected_candidates: Annotated[int, Field(ge=0)]
    teacher_as_student_count: Annotated[int, Field(ge=0)]
    printed_as_student_count: Annotated[int, Field(ge=0)]
    teacher_samples_with_student_regions: Annotated[int, Field(ge=0)]
    student_regions_on_teacher_samples: Annotated[int, Field(ge=0)]
    v1_predicted_region_count: Annotated[int, Field(ge=0)]
    filtered_predicted_region_count: Annotated[int, Field(ge=0)]

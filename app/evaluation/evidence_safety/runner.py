"""Evaluate the fixed post-V1 extraction safety policy without tuning V1."""

from __future__ import annotations

from typing import cast

from app.evaluation.evidence_benchmark.evaluation import evaluate_evidence_benchmark
from app.evaluation.evidence_redesign.models import DatasetId
from app.evaluation.evidence_redesign.runner import FrozenEvidenceDataset
from app.evaluation.evidence_safety.models import SafetyFilterBenchmarkResult
from app.evidence.models import EvidenceType, TestEvidence
from app.evidence.safety_filter import (
    StudentCandidateSafetyFilter,
    as_filtered_test_evidence,
)
from app.evidence.safety_models import CandidateDisposition, SafeTestExtraction


def run_safety_filter(
    dataset: FrozenEvidenceDataset,
) -> tuple[
    SafetyFilterBenchmarkResult,
    dict[str, SafeTestExtraction],
    dict[str, TestEvidence],
]:
    """Apply one frozen policy to one complete immutable benchmark."""

    policy = StudentCandidateSafetyFilter()
    extractions = {
        sample_id: policy.apply(evidence)
        for sample_id, evidence in sorted(dataset.v1_predictions.items())
    }
    predictions = {
        sample_id: as_filtered_test_evidence(result)
        for sample_id, result in extractions.items()
    }
    evaluation = evaluate_evidence_benchmark(
        dataset.manifest,
        dataset.annotations,
        predictions,
    )
    teacher_samples = tuple(
        item
        for item in evaluation.samples
        if item.human_class is EvidenceType.TEACHER_CANDIDATE
    )
    printed_samples = tuple(
        item for item in evaluation.samples if item.human_class is EvidenceType.PRINTED
    )
    decisions = tuple(
        decision for result in extractions.values() for decision in result.decisions
    )
    return (
        SafetyFilterBenchmarkResult(
            dataset_id=cast(DatasetId, dataset.dataset_id),
            annotation_fingerprint=dataset.fingerprint,
            evaluation=evaluation,
            safe_tests=sum(
                result.status.value == "safe" for result in extractions.values()
            ),
            review_required_tests=sum(
                result.status.value == "review_required"
                for result in extractions.values()
            ),
            empty_tests=sum(
                result.status.value == "empty" for result in extractions.values()
            ),
            safe_candidates=sum(
                item.disposition is CandidateDisposition.SAFE_STUDENT
                for item in decisions
            ),
            quarantined_candidates=sum(
                item.disposition is CandidateDisposition.REVIEW_REQUIRED
                for item in decisions
            ),
            rejected_candidates=sum(
                item.disposition is CandidateDisposition.REJECTED_NON_STUDENT
                for item in decisions
            ),
            teacher_as_student_count=sum(
                item.predicted_class is EvidenceType.STUDENT_CANDIDATE
                for item in teacher_samples
            ),
            printed_as_student_count=sum(
                item.predicted_class is EvidenceType.STUDENT_CANDIDATE
                for item in printed_samples
            ),
            teacher_samples_with_student_regions=sum(
                bool(predictions[item.sample_id].answer_regions)
                for item in teacher_samples
            ),
            student_regions_on_teacher_samples=sum(
                len(predictions[item.sample_id].answer_regions)
                for item in teacher_samples
            ),
            v1_predicted_region_count=sum(
                len(item.answer_regions) for item in dataset.v1_predictions.values()
            ),
            filtered_predicted_region_count=sum(
                len(item.answer_regions) for item in predictions.values()
            ),
        ),
        extractions,
        predictions,
    )

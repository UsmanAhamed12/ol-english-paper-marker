"""Run controlled V2 variants against all frozen evidence benchmarks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import cv2

from app.core.exceptions import EvidenceSeparationError
from app.domain.models.paper import PaperPage
from app.evaluation.evidence_benchmark.annotations import EvidenceAnnotationStore
from app.evaluation.evidence_benchmark.evaluation import (
    EvidenceBenchmarkEvaluation,
    evaluate_evidence_benchmark,
)
from app.evaluation.evidence_benchmark.metrics import (
    ClassificationMetrics,
    classification_metrics,
)
from app.evaluation.evidence_benchmark.models import (
    EvidenceBenchmarkManifest,
    EvidenceBenchmarkSample,
)
from app.evaluation.evidence_expansion.models import (
    EvidenceExpansionManifest,
    EvidenceExpansionSample,
)
from app.evaluation.evidence_redesign.models import DatasetId, EvidenceV2BenchmarkResult
from app.evaluation.teacher_evidence.models import (
    TeacherEvidenceManifest,
    TeacherEvidenceSample,
)
from app.evidence.models import EvidenceType, TestEvidence
from app.evidence.separator_v2 import EvidenceSeparatorV2, as_test_evidence
from app.evidence.v2_models import EvidenceSeparatorV2Config, TestEvidenceV2

Manifest = (
    EvidenceBenchmarkManifest | EvidenceExpansionManifest | TeacherEvidenceManifest
)
Sample = EvidenceBenchmarkSample | EvidenceExpansionSample | TeacherEvidenceSample


@dataclass(frozen=True, slots=True)
class FrozenEvidenceDataset:
    """Validated private inputs for one immutable benchmark."""

    dataset_id: str
    fingerprint: str
    manifest: Manifest
    annotations: EvidenceAnnotationStore
    v1_predictions: dict[str, TestEvidence]


@dataclass(frozen=True, slots=True)
class VariantRun:
    """One in-memory controlled result including private V2 geometry."""

    result: EvidenceV2BenchmarkResult
    predictions: dict[str, TestEvidenceV2]


def controlled_variants() -> tuple[EvidenceSeparatorV2Config, ...]:
    """Predetermined small ablation set; no sample-specific configuration."""

    return (
        EvidenceSeparatorV2Config(
            variant="v2-a",
            enable_teacher_risk=False,
            enable_ambiguity_margin=False,
            enable_paragraph_merging=False,
            student_threshold=0.40,
            printed_threshold=0.56,
        ),
        EvidenceSeparatorV2Config(
            variant="v2-b",
            enable_teacher_risk=True,
            enable_ambiguity_margin=False,
            enable_paragraph_merging=False,
            student_threshold=0.40,
            printed_threshold=0.56,
            teacher_threshold=0.50,
            teacher_suppression_threshold=0.44,
        ),
        EvidenceSeparatorV2Config(
            variant="v2-c",
            enable_teacher_risk=True,
            enable_ambiguity_margin=True,
            enable_paragraph_merging=False,
            student_threshold=0.40,
            printed_threshold=0.56,
            teacher_threshold=0.50,
            teacher_suppression_threshold=0.44,
            ambiguity_margin=0.08,
        ),
        EvidenceSeparatorV2Config(
            variant="v2-d",
            enable_teacher_risk=True,
            enable_ambiguity_margin=True,
            enable_paragraph_merging=True,
            student_threshold=0.40,
            printed_threshold=0.56,
            teacher_threshold=0.50,
            teacher_suppression_threshold=0.44,
            ambiguity_margin=0.06,
            answer_padding_ratio=0.10,
            answer_merge_gap_ratio=0.45,
            maximum_answer_area_ratio=0.18,
        ),
    )


def run_variant(
    dataset: FrozenEvidenceDataset,
    config: EvidenceSeparatorV2Config,
) -> VariantRun:
    """Apply one configuration to every sample in an immutable dataset."""

    samples = cast(tuple[Sample, ...], dataset.manifest.samples)
    separator = EvidenceSeparatorV2(config)
    predictions: dict[str, TestEvidenceV2] = {}
    metric_predictions: dict[str, TestEvidence] = {}
    for sample in samples:
        base = dataset.v1_predictions.get(sample.sample_id)
        if base is None:
            raise EvidenceSeparationError("V2 input predictions are incomplete")
        page = _page(sample, base)
        result = separator.separate_existing(page, base)
        predictions[sample.sample_id] = result
        metric_predictions[sample.sample_id] = as_test_evidence(result)
    evaluation = evaluate_evidence_benchmark(
        dataset.manifest, dataset.annotations, metric_predictions
    )
    return VariantRun(
        result=_safe_result(dataset, samples, evaluation, predictions),
        predictions=predictions,
    )


def combined_classification(
    runs: Sequence[EvidenceV2BenchmarkResult],
) -> ClassificationMetrics:
    """Combine aligned sample labels without hiding per-dataset reporting."""

    human = tuple(
        sample.human_class for run in runs for sample in run.evaluation.samples
    )
    predicted = tuple(
        sample.predicted_class for run in runs for sample in run.evaluation.samples
    )
    return classification_metrics(human, predicted)


def select_variant(
    experiments: Sequence[
        tuple[EvidenceSeparatorV2Config, Sequence[EvidenceV2BenchmarkResult]]
    ],
) -> str | None:
    """Select only a candidate that does not exceed V1 teacher contamination."""

    def rank(
        item: tuple[EvidenceSeparatorV2Config, Sequence[EvidenceV2BenchmarkResult]],
    ) -> tuple[float, ...]:
        config, results = item
        teacher = next(
            result for result in results if result.dataset_id == "evidence_teacher_v1"
        )
        combined = combined_classification(results)
        by_type = {row.evidence_type: row for row in combined.per_class}
        student = by_type[EvidenceType.STUDENT_CANDIDATE]
        teacher_class = by_type[EvidenceType.TEACHER_CANDIDATE]
        printed = by_type[EvidenceType.PRINTED]
        localization = teacher.evaluation.answer_localization
        empty = _f1(
            localization.empty_prediction_precision,
            localization.empty_recall,
        )
        return (
            teacher.teacher_as_student_count,
            -(student.precision or 0),
            -(student.recall or 0),
            -(teacher_class.f1 or 0),
            -(printed.f1 or 0),
            combined.unknown_prediction_rate,
            -(localization.at_iou_50.f1 or 0),
            -(empty or 0),
            float(ord(config.variant[-1])),
        )

    best_config, best_results = min(experiments, key=rank)
    teacher = next(
        result for result in best_results if result.dataset_id == "evidence_teacher_v1"
    )
    return (
        best_config.variant
        if passes_teacher_safety_gate(teacher.teacher_as_student_count)
        else None
    )


def passes_teacher_safety_gate(
    teacher_as_student_count: int,
    *,
    v1_maximum: int = 3,
) -> bool:
    """Require a candidate to be no worse than frozen V1 teacher contamination."""

    if teacher_as_student_count < 0 or v1_maximum < 0:
        raise ValueError("Teacher contamination counts cannot be negative")
    return teacher_as_student_count <= v1_maximum


def _safe_result(
    dataset: FrozenEvidenceDataset,
    samples: tuple[Sample, ...],
    evaluation: EvidenceBenchmarkEvaluation,
    predictions: dict[str, TestEvidenceV2],
) -> EvidenceV2BenchmarkResult:
    teacher_samples = tuple(
        item
        for item in evaluation.samples
        if item.human_class is EvidenceType.TEACHER_CANDIDATE
    )
    teacher_as_student = sum(
        item.predicted_class is EvidenceType.STUDENT_CANDIDATE
        for item in teacher_samples
    )
    teacher_with_answers = sum(
        bool(predictions[item.sample_id].student_answer_regions)
        for item in teacher_samples
    )
    teacher_answer_boxes = sum(
        len(predictions[item.sample_id].student_answer_regions)
        for item in teacher_samples
    )
    area_ratios = [
        answer.bbox.width
        * answer.bbox.height
        / (sample.region.width * sample.region.height)
        for sample in samples
        for answer in predictions[sample.sample_id].student_answer_regions
    ]
    return EvidenceV2BenchmarkResult(
        dataset_id=cast(DatasetId, dataset.dataset_id),
        annotation_fingerprint=dataset.fingerprint,
        evaluation=evaluation,
        teacher_as_student_count=teacher_as_student,
        teacher_as_student_rate=(
            teacher_as_student / len(teacher_samples) if teacher_samples else None
        ),
        teacher_samples_with_student_regions=teacher_with_answers,
        student_regions_on_teacher_samples=teacher_answer_boxes,
        mean_predicted_region_area_ratio=(
            sum(area_ratios) / len(area_ratios) if area_ratios else 0.0
        ),
    )


def _page(sample: Sample, base: TestEvidence) -> PaperPage:
    source = sample.source_image_path.resolve(strict=True)
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim < 2:
        raise EvidenceSeparationError("Private V2 source image is invalid")
    height, width = image.shape[:2]
    if (width, height) != (sample.page_width, sample.page_height):
        raise EvidenceSeparationError("Private V2 source dimensions changed")
    return PaperPage(
        paper_id=base.paper_id,
        page_number=base.page_number,
        image_path=source,
        width=width,
        height=height,
    )


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

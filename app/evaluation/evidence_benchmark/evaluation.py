"""Evaluate frozen human evidence labels against deterministic predictions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from statistics import fmean
from typing import Annotated, cast

from pydantic import BaseModel, ConfigDict, Field

from app.core.exceptions import EvidenceSeparationError
from app.evaluation.evidence_benchmark.annotations import (
    EvidenceAnnotation,
    EvidenceAnnotationStore,
    annotation_fingerprint,
)
from app.evaluation.evidence_benchmark.metrics import (
    AnswerLocalizationMetrics,
    ClassificationMetrics,
    answer_localization_metrics,
    classification_metrics,
    dominant_evidence_type,
    intersection_over_union,
)
from app.evaluation.evidence_benchmark.models import (
    EvidenceBenchmarkManifest,
    EvidenceBenchmarkSample,
)
from app.evaluation.evidence_expansion.models import (
    EvidenceExpansionManifest,
    EvidenceExpansionSample,
)
from app.evidence.models import EvidenceType, TestEvidence
from app.ocr.models import BoundingBox


class SampleEvidenceEvaluation(BaseModel):
    """Privacy-safe per-sample facts used for aggregate error analysis."""

    model_config = ConfigDict(frozen=True)

    sample_id: str
    categories: tuple[str, ...]
    human_class: EvidenceType
    predicted_class: EvidenceType
    class_correct: bool
    human_answer_boxes: Annotated[int, Field(ge=0)]
    predicted_answer_boxes: Annotated[int, Field(ge=0)]
    matched_answer_boxes_at_50: Annotated[int, Field(ge=0)]
    mean_positive_iou: Annotated[float, Field(ge=0, le=1)] | None


class EvidenceBenchmarkEvaluation(BaseModel):
    """Complete safe-metric result for one frozen annotation dataset."""

    model_config = ConfigDict(frozen=True)

    annotation_fingerprint: str
    classification: ClassificationMetrics
    answer_localization: AnswerLocalizationMetrics
    category_classification_errors: dict[str, int]
    category_answer_misses: dict[str, int]
    category_answer_extras: dict[str, int]
    samples: tuple[SampleEvidenceEvaluation, ...]


def evaluate_evidence_benchmark(
    manifest: EvidenceBenchmarkManifest | EvidenceExpansionManifest,
    annotations: EvidenceAnnotationStore,
    predictions: dict[str, TestEvidence],
) -> EvidenceBenchmarkEvaluation:
    """Evaluate exact aligned sample IDs without exposing page content."""

    benchmark_samples = cast(
        tuple[EvidenceBenchmarkSample | EvidenceExpansionSample, ...],
        manifest.samples,
    )
    human_by_id = {item.sample_id: item for item in annotations.annotations}
    expected_ids = [sample.sample_id for sample in benchmark_samples]
    if set(human_by_id) != set(expected_ids) or set(predictions) != set(expected_ids):
        raise EvidenceSeparationError(
            "Evidence evaluation inputs do not match the benchmark samples"
        )
    ordered_annotations = tuple(human_by_id[sample_id] for sample_id in expected_ids)
    ordered_evidence = tuple(predictions[sample_id] for sample_id in expected_ids)
    human_classes = tuple(item.evidence_type for item in ordered_annotations)
    predicted_classes = tuple(dominant_evidence_type(item) for item in ordered_evidence)
    predicted_boxes = tuple(
        _crop_relative_answer_boxes(sample.region, evidence)
        for sample, evidence in zip(benchmark_samples, ordered_evidence, strict=True)
    )
    evaluation_samples = tuple(
        _sample_evaluation(sample, annotation, evidence, boxes)
        for sample, annotation, evidence, boxes in zip(
            benchmark_samples,
            ordered_annotations,
            ordered_evidence,
            predicted_boxes,
            strict=True,
        )
    )
    return EvidenceBenchmarkEvaluation(
        annotation_fingerprint=annotation_fingerprint(annotations),
        classification=classification_metrics(human_classes, predicted_classes),
        answer_localization=answer_localization_metrics(
            ordered_annotations, predicted_boxes
        ),
        category_classification_errors=_category_counts(
            evaluation_samples, lambda item: not item.class_correct
        ),
        category_answer_misses=_category_counts(
            evaluation_samples,
            lambda item: item.matched_answer_boxes_at_50 < item.human_answer_boxes,
        ),
        category_answer_extras=_category_counts(
            evaluation_samples,
            lambda item: item.matched_answer_boxes_at_50 < item.predicted_answer_boxes,
        ),
        samples=evaluation_samples,
    )


def _sample_evaluation(
    sample: EvidenceBenchmarkSample | EvidenceExpansionSample,
    annotation: EvidenceAnnotation,
    evidence: TestEvidence,
    predicted_boxes: tuple[BoundingBox, ...],
) -> SampleEvidenceEvaluation:
    human_boxes = tuple(item.bbox for item in annotation.answer_regions)
    scores = _matched_scores(human_boxes, predicted_boxes, threshold=0.0)
    matched_50 = len(_matched_scores(human_boxes, predicted_boxes, threshold=0.50))
    predicted_class = dominant_evidence_type(evidence)
    return SampleEvidenceEvaluation(
        sample_id=sample.sample_id,
        categories=_sample_categories(sample),
        human_class=annotation.evidence_type,
        predicted_class=predicted_class,
        class_correct=annotation.evidence_type is predicted_class,
        human_answer_boxes=len(human_boxes),
        predicted_answer_boxes=len(predicted_boxes),
        matched_answer_boxes_at_50=matched_50,
        mean_positive_iou=fmean(scores) if scores else None,
    )


def _sample_categories(
    sample: EvidenceBenchmarkSample | EvidenceExpansionSample,
) -> tuple[str, ...]:
    categories = tuple(category.value for category in sample.categories)
    if isinstance(sample, EvidenceExpansionSample):
        test_context = (
            "detected_test" if sample.test_number is not None else "no_detected_test"
        )
        return (sample.discovery_category.value, test_context, *categories)
    return categories


def _crop_relative_answer_boxes(
    crop: BoundingBox, evidence: TestEvidence
) -> tuple[BoundingBox, ...]:
    boxes = []
    for region in evidence.answer_regions:
        x = region.bbox.x - crop.x
        y = region.bbox.y - crop.y
        box = BoundingBox(
            x=x,
            y=y,
            width=region.bbox.width,
            height=region.bbox.height,
        )
        if x < 0 or y < 0 or x + box.width > crop.width or y + box.height > crop.height:
            raise EvidenceSeparationError(
                "Predicted answer region exceeds benchmark crop geometry"
            )
        boxes.append(box)
    return tuple(boxes)


def _matched_scores(
    human: tuple[BoundingBox, ...],
    predicted: tuple[BoundingBox, ...],
    *,
    threshold: float,
) -> tuple[float, ...]:
    candidates = sorted(
        (
            (intersection_over_union(actual, guess), actual_index, guess_index)
            for actual_index, actual in enumerate(human)
            for guess_index, guess in enumerate(predicted)
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    used_human: set[int] = set()
    used_predictions: set[int] = set()
    matches = []
    for score, human_index, predicted_index in candidates:
        if score <= 0 or score < threshold:
            continue
        if human_index in used_human or predicted_index in used_predictions:
            continue
        used_human.add(human_index)
        used_predictions.add(predicted_index)
        matches.append(score)
    return tuple(matches)


def _category_counts(
    samples: tuple[SampleEvidenceEvaluation, ...],
    predicate: Callable[[SampleEvidenceEvaluation], bool],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for sample in samples:
        if predicate(sample):
            counts.update(sample.categories)
    return dict(sorted(counts.items()))

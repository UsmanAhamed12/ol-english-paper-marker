"""Deterministic evidence classification and answer-localization metrics."""

from __future__ import annotations

from collections import Counter
from statistics import fmean, median
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.evidence_benchmark.annotations import (
    AnswerAnnotationStatus,
    EvidenceAnnotation,
)
from app.evidence.models import EvidenceType, TestEvidence
from app.ocr.models import BoundingBox


class EvidenceClassMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_type: EvidenceType
    support: Annotated[int, Field(ge=0)]
    predicted: Annotated[int, Field(ge=0)]
    true_positives: Annotated[int, Field(ge=0)]
    false_positives: Annotated[int, Field(ge=0)]
    false_negatives: Annotated[int, Field(ge=0)]
    precision: Annotated[float, Field(ge=0, le=1)] | None
    recall: Annotated[float, Field(ge=0, le=1)] | None
    f1: Annotated[float, Field(ge=0, le=1)] | None


class ClassificationMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_count: Annotated[int, Field(gt=0)]
    overall_accuracy: Annotated[float, Field(ge=0, le=1)]
    per_class: tuple[EvidenceClassMetrics, ...]
    macro_f1: Annotated[float, Field(ge=0, le=1)] | None
    weighted_f1: Annotated[float, Field(ge=0, le=1)] | None
    confusion_matrix: dict[str, dict[str, int]]
    unknown_prediction_rate: Annotated[float, Field(ge=0, le=1)]
    human_unknown_rate: Annotated[float, Field(ge=0, le=1)]


class LocalizationThresholdMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    iou_threshold: Annotated[float, Field(gt=0, le=1)]
    matched_box_count: Annotated[int, Field(ge=0)]
    precision: Annotated[float, Field(ge=0, le=1)] | None
    recall: Annotated[float, Field(ge=0, le=1)] | None
    f1: Annotated[float, Field(ge=0, le=1)] | None


class AnswerLocalizationMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    human_box_count: Annotated[int, Field(ge=0)]
    predicted_box_count: Annotated[int, Field(ge=0)]
    positive_overlap_match_count: Annotated[int, Field(ge=0)]
    mean_matched_iou: Annotated[float, Field(ge=0, le=1)] | None
    median_matched_iou: Annotated[float, Field(ge=0, le=1)] | None
    at_iou_50: LocalizationThresholdMetrics
    at_iou_25: LocalizationThresholdMetrics
    missed_human_boxes_at_50: Annotated[int, Field(ge=0)]
    extra_predicted_boxes_at_50: Annotated[int, Field(ge=0)]
    missed_human_boxes_at_25: Annotated[int, Field(ge=0)]
    extra_predicted_boxes_at_25: Annotated[int, Field(ge=0)]
    verified_empty_samples: Annotated[int, Field(ge=0)]
    predicted_empty_samples: Annotated[int, Field(ge=0)]
    correctly_predicted_empty: Annotated[int, Field(ge=0)]
    false_positive_empty_samples: Annotated[int, Field(ge=0)]
    empty_prediction_precision: Annotated[float, Field(ge=0, le=1)] | None
    empty_recall: Annotated[float, Field(ge=0, le=1)] | None


def dominant_evidence_type(evidence: TestEvidence) -> EvidenceType:
    """Select the unique largest total evidence-box area, otherwise UNKNOWN."""

    areas: Counter[EvidenceType] = Counter()
    for region in evidence.evidence_regions:
        areas[region.evidence_type] += region.bbox.width * region.bbox.height
    if not areas:
        return EvidenceType.UNKNOWN
    maximum = max(areas.values())
    winners = [kind for kind, area in areas.items() if area == maximum]
    return winners[0] if len(winners) == 1 else EvidenceType.UNKNOWN


def classification_metrics(
    human: tuple[EvidenceType, ...],
    predicted: tuple[EvidenceType, ...],
) -> ClassificationMetrics:
    """Calculate multiclass metrics while retaining undefined values as None."""

    if not human or len(human) != len(predicted):
        raise ValueError("Classification labels must be non-empty and aligned")
    classes = tuple(EvidenceType)
    matrix = {actual.value: {guess.value: 0 for guess in classes} for actual in classes}
    for actual, guess in zip(human, predicted, strict=True):
        matrix[actual.value][guess.value] += 1
    rows = tuple(_class_metrics(kind, human, predicted) for kind in classes)
    defined_f1 = [row.f1 for row in rows if row.f1 is not None]
    supported = [row for row in rows if row.support > 0 and row.f1 is not None]
    total_support = sum(row.support for row in supported)
    return ClassificationMetrics(
        sample_count=len(human),
        overall_accuracy=sum(a is b for a, b in zip(human, predicted, strict=True))
        / len(human),
        per_class=rows,
        macro_f1=fmean(defined_f1) if defined_f1 else None,
        weighted_f1=(
            sum(row.f1 * row.support for row in supported if row.f1 is not None)
            / total_support
            if total_support
            else None
        ),
        confusion_matrix=matrix,
        unknown_prediction_rate=sum(item is EvidenceType.UNKNOWN for item in predicted)
        / len(predicted),
        human_unknown_rate=sum(item is EvidenceType.UNKNOWN for item in human)
        / len(human),
    )


def answer_localization_metrics(
    annotations: tuple[EvidenceAnnotation, ...],
    predicted_boxes: tuple[tuple[BoundingBox, ...], ...],
) -> AnswerLocalizationMetrics:
    """Calculate sample-local deterministic one-to-one IoU matching metrics."""

    if not annotations or len(annotations) != len(predicted_boxes):
        raise ValueError("Answer annotations and predictions must be aligned")
    human_by_sample = tuple(
        tuple(region.bbox for region in annotation.answer_regions)
        for annotation in annotations
    )
    human_count = sum(len(items) for items in human_by_sample)
    predicted_count = sum(len(items) for items in predicted_boxes)
    positive = _all_matches(human_by_sample, predicted_boxes, threshold=0.0)
    at_50 = _threshold_metrics(
        human_by_sample, predicted_boxes, human_count, predicted_count, 0.50
    )
    at_25 = _threshold_metrics(
        human_by_sample, predicted_boxes, human_count, predicted_count, 0.25
    )
    empty_pairs = [
        (annotation, predictions)
        for annotation, predictions in zip(annotations, predicted_boxes, strict=True)
        if annotation.answer_status is AnswerAnnotationStatus.VERIFIED_EMPTY
    ]
    predicted_empty_count = sum(not boxes for boxes in predicted_boxes)
    correctly_empty = sum(not boxes for _, boxes in empty_pairs)
    return AnswerLocalizationMetrics(
        human_box_count=human_count,
        predicted_box_count=predicted_count,
        positive_overlap_match_count=len(positive),
        mean_matched_iou=fmean(positive) if positive else None,
        median_matched_iou=median(positive) if positive else None,
        at_iou_50=at_50,
        at_iou_25=at_25,
        missed_human_boxes_at_50=human_count - at_50.matched_box_count,
        extra_predicted_boxes_at_50=predicted_count - at_50.matched_box_count,
        missed_human_boxes_at_25=human_count - at_25.matched_box_count,
        extra_predicted_boxes_at_25=predicted_count - at_25.matched_box_count,
        verified_empty_samples=len(empty_pairs),
        predicted_empty_samples=predicted_empty_count,
        correctly_predicted_empty=correctly_empty,
        false_positive_empty_samples=sum(bool(boxes) for _, boxes in empty_pairs),
        empty_prediction_precision=(
            correctly_empty / predicted_empty_count if predicted_empty_count else None
        ),
        empty_recall=correctly_empty / len(empty_pairs) if empty_pairs else None,
    )


def intersection_over_union(left: BoundingBox, right: BoundingBox) -> float:
    width = max(
        0,
        min(left.x + left.width, right.x + right.width) - max(left.x, right.x),
    )
    height = max(
        0,
        min(left.y + left.height, right.y + right.height) - max(left.y, right.y),
    )
    intersection = width * height
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union


def _class_metrics(
    kind: EvidenceType,
    human: tuple[EvidenceType, ...],
    predicted: tuple[EvidenceType, ...],
) -> EvidenceClassMetrics:
    true_positive = sum(
        actual is kind and guess is kind
        for actual, guess in zip(human, predicted, strict=True)
    )
    false_positive = sum(
        actual is not kind and guess is kind
        for actual, guess in zip(human, predicted, strict=True)
    )
    false_negative = sum(
        actual is kind and guess is not kind
        for actual, guess in zip(human, predicted, strict=True)
    )
    support = true_positive + false_negative
    predicted_count = true_positive + false_positive
    precision = true_positive / predicted_count if predicted_count else None
    recall = true_positive / support if support else None
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / denominator if denominator else None
    return EvidenceClassMetrics(
        evidence_type=kind,
        support=support,
        predicted=predicted_count,
        true_positives=true_positive,
        false_positives=false_positive,
        false_negatives=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _threshold_metrics(
    human: tuple[tuple[BoundingBox, ...], ...],
    predicted: tuple[tuple[BoundingBox, ...], ...],
    human_count: int,
    predicted_count: int,
    threshold: float,
) -> LocalizationThresholdMetrics:
    matched = len(_all_matches(human, predicted, threshold=threshold))
    precision = matched / predicted_count if predicted_count else None
    recall = matched / human_count if human_count else None
    denominator = 2 * matched + (predicted_count - matched) + (human_count - matched)
    return LocalizationThresholdMetrics(
        iou_threshold=threshold,
        matched_box_count=matched,
        precision=precision,
        recall=recall,
        f1=2 * matched / denominator if denominator else None,
    )


def _all_matches(
    human: tuple[tuple[BoundingBox, ...], ...],
    predicted: tuple[tuple[BoundingBox, ...], ...],
    *,
    threshold: float,
) -> tuple[float, ...]:
    return tuple(
        score
        for actual, guesses in zip(human, predicted, strict=True)
        for score in _greedy_matches(actual, guesses, threshold=threshold)
    )


def _greedy_matches(
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
    used_predicted: set[int] = set()
    matches = []
    for score, actual_index, guess_index in candidates:
        if score <= 0 or score < threshold:
            continue
        if actual_index in used_human or guess_index in used_predicted:
            continue
        used_human.add(actual_index)
        used_predicted.add(guess_index)
        matches.append(score)
    return tuple(matches)

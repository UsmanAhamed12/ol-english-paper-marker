"""Deterministic evidence classification and localization metric tests."""

from __future__ import annotations

from app.evaluation.evidence_benchmark.annotations import (
    AnswerAnnotationStatus,
    EvidenceAnnotation,
)
from app.evaluation.evidence_benchmark.metrics import (
    answer_localization_metrics,
    classification_metrics,
    intersection_over_union,
)
from app.evaluation.evidence_benchmark.models import GroundTruthAnswerRegion
from app.evidence.models import EvidenceType
from app.ocr.models import BoundingBox


def _annotation(sample_id: str, boxes: tuple[BoundingBox, ...]) -> EvidenceAnnotation:
    return EvidenceAnnotation(
        sample_id=sample_id,
        evidence_type=EvidenceType.STUDENT_CANDIDATE,
        answer_status=(
            AnswerAnnotationStatus.ANNOTATED
            if boxes
            else AnswerAnnotationStatus.VERIFIED_EMPTY
        ),
        answer_regions=tuple(GroundTruthAnswerRegion(bbox=box) for box in boxes),
    )


def test_classification_metrics_and_confusion_matrix() -> None:
    human = (
        EvidenceType.PRINTED,
        EvidenceType.STUDENT_CANDIDATE,
        EvidenceType.TEACHER_CANDIDATE,
        EvidenceType.UNKNOWN,
    )
    predicted = (
        EvidenceType.PRINTED,
        EvidenceType.PRINTED,
        EvidenceType.STUDENT_CANDIDATE,
        EvidenceType.UNKNOWN,
    )

    report = classification_metrics(human, predicted)
    by_class = {item.evidence_type: item for item in report.per_class}

    assert report.overall_accuracy == 0.5
    assert report.confusion_matrix["student_candidate"]["printed"] == 1
    assert report.confusion_matrix["teacher_candidate"]["student_candidate"] == 1
    assert by_class[EvidenceType.STUDENT_CANDIDATE].precision == 0.0
    assert by_class[EvidenceType.STUDENT_CANDIDATE].recall == 0.0
    assert report.unknown_prediction_rate == 0.25
    assert report.human_unknown_rate == 0.25


def test_zero_support_metrics_remain_undefined() -> None:
    report = classification_metrics((EvidenceType.PRINTED,), (EvidenceType.PRINTED,))
    teacher = next(
        item
        for item in report.per_class
        if item.evidence_type is EvidenceType.TEACHER_CANDIDATE
    )
    assert teacher.support == 0
    assert teacher.precision is None
    assert teacher.recall is None
    assert teacher.f1 is None


def test_iou_and_one_to_one_threshold_metrics() -> None:
    human = (
        BoundingBox(x=0, y=0, width=100, height=100),
        BoundingBox(x=150, y=0, width=100, height=100),
    )
    predictions = (
        BoundingBox(x=0, y=0, width=100, height=100),
        BoundingBox(x=140, y=0, width=100, height=100),
        BoundingBox(x=400, y=0, width=50, height=50),
    )
    report = answer_localization_metrics(
        (_annotation("sample_001", human),), (predictions,)
    )

    assert intersection_over_union(human[0], predictions[0]) == 1.0
    assert report.human_box_count == 2
    assert report.predicted_box_count == 3
    assert report.at_iou_50.matched_box_count == 2
    assert report.at_iou_50.precision == 2 / 3
    assert report.at_iou_50.recall == 1.0
    assert report.extra_predicted_boxes_at_50 == 1
    assert report.mean_matched_iou is not None


def test_verified_empty_behavior_counts_false_positive_samples() -> None:
    empty = _annotation("sample_001", ())
    report = answer_localization_metrics(
        (empty, _annotation("sample_002", ())),
        (
            (),
            (BoundingBox(x=1, y=1, width=10, height=10),),
        ),
    )

    assert report.verified_empty_samples == 2
    assert report.correctly_predicted_empty == 1
    assert report.false_positive_empty_samples == 1
    assert report.predicted_empty_samples == 1
    assert report.empty_prediction_precision == 1.0
    assert report.empty_recall == 0.5
    assert report.at_iou_50.recall is None
    assert report.at_iou_50.precision == 0.0

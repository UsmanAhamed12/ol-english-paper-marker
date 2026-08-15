"""Synthetic end-to-end evidence report and comparison-overlay tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import UUID

import cv2
import numpy as np

from app.evaluation.evidence_benchmark.annotations import (
    AnswerAnnotationStatus,
    EvidenceAnnotation,
    EvidenceAnnotationStore,
)
from app.evaluation.evidence_benchmark.evaluation import (
    evaluate_evidence_benchmark,
)
from app.evaluation.evidence_benchmark.evaluation_overlay import (
    render_evaluation_overlay,
)
from app.evaluation.evidence_benchmark.metrics import dominant_evidence_type
from app.evaluation.evidence_benchmark.models import (
    EvidenceBenchmarkManifest,
    EvidenceBenchmarkSample,
    EvidenceSampleCategory,
    GroundTruthAnswerRegion,
)
from app.evidence.models import (
    AnswerRegionSignal,
    EvidenceRegion,
    EvidenceSignal,
    EvidenceType,
    StudentAnswerRegion,
)
from app.evidence.models import TestEvidence as EvidenceTestResult
from app.ocr.models import BoundingBox
from tests.unit.evidence.helpers import features

PAPER_ID = UUID("00000000-0000-0000-0000-000000000001")


def _sample(source: Path) -> EvidenceBenchmarkSample:
    return EvidenceBenchmarkSample(
        sample_id="sample_001",
        paper_alias="paper-a",
        page_number=1,
        test_number=1,
        source_image_path=source,
        page_width=500,
        page_height=400,
        region=BoundingBox(x=100, y=100, width=300, height=200),
        categories=(EvidenceSampleCategory.SHORT_ANSWER,),
    )


def _evidence(source: Path) -> EvidenceTestResult:
    region = EvidenceRegion(
        paper_id=PAPER_ID,
        page_number=1,
        test_number=1,
        bbox=BoundingBox(x=120, y=120, width=160, height=50),
        evidence_type=EvidenceType.STUDENT_CANDIDATE,
        confidence=0.8,
        signals=(EvidenceSignal.IRREGULAR_BASELINE,),
        features=features(irregularity=0.8),
        source_image_path=source,
        classification_strategy="synthetic",
    )
    answer = StudentAnswerRegion(
        paper_id=PAPER_ID,
        page_number=1,
        test_number=1,
        bbox=BoundingBox(x=120, y=120, width=160, height=50),
        confidence=0.8,
        signals=(AnswerRegionSignal.STUDENT_EVIDENCE_CLUSTER,),
        source_image_path=source,
        detection_strategy="synthetic",
    )
    return EvidenceTestResult(
        paper_id=PAPER_ID,
        page_number=1,
        test_number=1,
        region_bbox=BoundingBox(x=100, y=100, width=300, height=200),
        evidence_regions=(region,),
        answer_regions=(answer,),
    )


def _annotation() -> EvidenceAnnotation:
    return EvidenceAnnotation(
        sample_id="sample_001",
        evidence_type=EvidenceType.STUDENT_CANDIDATE,
        answer_status=AnswerAnnotationStatus.ANNOTATED,
        answer_regions=(
            GroundTruthAnswerRegion(bbox=BoundingBox(x=20, y=20, width=160, height=50)),
        ),
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dominant_class_and_aligned_evaluation(tmp_path: Path) -> None:
    source = (tmp_path / "source.png").resolve()
    source.write_bytes(b"synthetic")
    manifest = EvidenceBenchmarkManifest(samples=(_sample(source),))
    evidence = _evidence(source)

    report = evaluate_evidence_benchmark(
        manifest,
        EvidenceAnnotationStore(annotations=(_annotation(),)),
        {"sample_001": evidence},
    )

    assert dominant_evidence_type(evidence) is EvidenceType.STUDENT_CANDIDATE
    assert report.classification.overall_accuracy == 1.0
    assert report.answer_localization.at_iou_50.matched_box_count == 1
    assert report.category_classification_errors == {}
    assert report.category_answer_misses == {}
    assert report.category_answer_extras == {}


def test_empty_evidence_dominance_is_unknown(tmp_path: Path) -> None:
    source = (tmp_path / "source.png").resolve()
    source.write_bytes(b"synthetic")
    evidence = EvidenceTestResult(
        paper_id=PAPER_ID,
        page_number=1,
        test_number=1,
        region_bbox=BoundingBox(x=100, y=100, width=300, height=200),
    )
    assert dominant_evidence_type(evidence) is EvidenceType.UNKNOWN


def test_comparison_overlay_preserves_private_sample(tmp_path: Path) -> None:
    source = (tmp_path / "sample_001.png").resolve()
    image = np.full((200, 300, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(source), image)
    before = _hash(source)

    output = render_evaluation_overlay(
        _sample(source),
        _annotation(),
        _evidence(source),
        source,
        tmp_path / "overlays" / "sample_001.png",
    )

    assert output.is_file()
    assert _hash(source) == before
    rendered = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert rendered is not None
    assert rendered.shape[:2] == image.shape[:2]

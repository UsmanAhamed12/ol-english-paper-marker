"""Synthetic freeze, recovery, and teacher-contamination tests."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.exceptions import EvidenceSeparationError
from app.evaluation.evidence_benchmark.annotations import (
    AnswerAnnotationStatus,
    EvidenceAnnotation,
    EvidenceAnnotationStore,
    annotation_fingerprint,
)
from app.evaluation.evidence_benchmark.evaluation import SampleEvidenceEvaluation
from app.evaluation.evidence_benchmark.metrics import classification_metrics
from app.evaluation.teacher_evidence.freezing import (
    freeze_teacher_annotations,
    teacher_manifest_fingerprint,
    verify_frozen_teacher_annotations,
)
from app.evaluation.teacher_evidence.models import (
    TeacherCandidateFeatures,
    TeacherDiscoveryCategory,
    TeacherDiscoverySignal,
    TeacherEvidenceManifest,
    TeacherEvidenceSample,
)
from app.evidence.models import EvidenceType
from app.ocr.models import BoundingBox
from scripts.finalize_teacher_evidence import (
    _binary_f1,
    _stratified_teacher_analysis,
    _teacher_localization_contamination,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


def _manifest(tmp_path: Path) -> TeacherEvidenceManifest:
    source = (tmp_path / "canonical.png").resolve()
    assert cv2.imwrite(str(source), np.full((100, 200, 3), 255, dtype=np.uint8))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    sample = TeacherEvidenceSample(
        sample_id="evidence_teacher_v1_001",
        paper_alias="paper-a",
        page_number=1,
        source_image_path=source,
        source_image_sha256=digest,
        page_width=200,
        page_height=100,
        region=BoundingBox(x=0, y=0, width=200, height=100),
        candidate_component=BoundingBox(x=20, y=20, width=20, height=20),
        discovery_category=TeacherDiscoveryCategory.MARGIN_SCORE,
        discovery_signals=(TeacherDiscoverySignal.MARGIN_ACTIVITY,),
        features=TeacherCandidateFeatures(
            component_area_ratio=0.01,
            chromatic_foreground_ratio=0.2,
            mean_saturation=0.2,
            foreground_ratio=0.1,
            edge_density=0.1,
            local_whitespace_ratio=0.8,
            margin_proximity=0.9,
            ocr_proximity=0.2,
            nearby_ocr_words=0,
            angled_line_count=1,
        ),
        selection_rank=1,
        discovery_reason="margin_score_risk+multi_signal_context",
    )
    samples = tmp_path / "samples"
    samples.mkdir()
    assert cv2.imwrite(
        str(samples / "evidence_teacher_v1_001.png"),
        np.full((100, 200, 3), 255, dtype=np.uint8),
    )
    return TeacherEvidenceManifest(samples=(sample,))


def _store() -> EvidenceAnnotationStore:
    return EvidenceAnnotationStore(
        annotations=(
            EvidenceAnnotation(
                sample_id="evidence_teacher_v1_001",
                evidence_type=EvidenceType.TEACHER_CANDIDATE,
                answer_status=AnswerAnnotationStatus.VERIFIED_EMPTY,
            ),
        )
    )


def test_teacher_snapshot_is_recoverable_and_refuses_overwrite(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    store = _store()
    frozen = tmp_path / "frozen"
    snapshot, provenance_path, provenance = freeze_teacher_annotations(
        manifest,
        store,
        samples_root=tmp_path / "samples",
        frozen_root=frozen,
        private_root=tmp_path,
        phase_4c_4r_fingerprint=HASH_A,
        evidence_v2_fingerprint=HASH_B,
        ocr_benchmark_fingerprint=HASH_C,
        corpus_sha256=HASH_D,
        canonical_snapshot_sha256=HASH_E,
        timestamp_factory=lambda: datetime(2026, 8, 21, tzinfo=UTC),
    )
    recovered, recovered_provenance = verify_frozen_teacher_annotations(
        manifest,
        snapshot,
        provenance_path,
        samples_root=tmp_path / "samples",
        private_root=tmp_path,
    )
    assert annotation_fingerprint(recovered) == annotation_fingerprint(store)
    assert recovered_provenance == provenance
    assert recovered_provenance.evidence_v2_fingerprint == HASH_B
    assert teacher_manifest_fingerprint(manifest) == provenance.manifest_fingerprint
    with pytest.raises(EvidenceSeparationError, match="already exists"):
        freeze_teacher_annotations(
            manifest,
            store,
            samples_root=tmp_path / "samples",
            frozen_root=frozen,
            private_root=tmp_path,
            phase_4c_4r_fingerprint=HASH_A,
            evidence_v2_fingerprint=HASH_B,
            ocr_benchmark_fingerprint=HASH_C,
            corpus_sha256=HASH_D,
            canonical_snapshot_sha256=HASH_E,
        )


def test_teacher_to_student_and_answer_contamination_are_explicit() -> None:
    metrics = classification_metrics(
        (EvidenceType.TEACHER_CANDIDATE, EvidenceType.STUDENT_CANDIDATE),
        (EvidenceType.STUDENT_CANDIDATE, EvidenceType.TEACHER_CANDIDATE),
    )
    assert (
        metrics.confusion_matrix[EvidenceType.TEACHER_CANDIDATE.value][
            EvidenceType.STUDENT_CANDIDATE.value
        ]
        == 1
    )
    samples = (
        SampleEvidenceEvaluation(
            sample_id="synthetic_teacher",
            categories=("margin_score_risk",),
            human_class=EvidenceType.TEACHER_CANDIDATE,
            predicted_class=EvidenceType.UNKNOWN,
            class_correct=False,
            human_answer_boxes=0,
            predicted_answer_boxes=3,
            matched_answer_boxes_at_50=0,
            mean_positive_iou=None,
        ),
    )
    contamination = _teacher_localization_contamination(samples)
    assert contamination["teacher_samples_with_predicted_answer_regions"] == 1
    assert contamination["predicted_student_answer_boxes_on_teacher_samples"] == 3
    assert (
        _stratified_teacher_analysis(samples)["margin_score_risk"][
            "predicted_answer_boxes_on_teacher_samples"
        ]
        == 3
    )


def test_empty_f1_retains_undefined_and_zero_semantics() -> None:
    assert _binary_f1(0, 0, 0) is None
    assert _binary_f1(0, 2, 3) == 0.0
    assert _binary_f1(2, 1, 1) == pytest.approx(2 / 3)

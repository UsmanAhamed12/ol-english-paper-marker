"""Synthetic evidence-v2 freezing, recovery, and evaluation tests."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

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
from app.evaluation.evidence_benchmark.evaluation import evaluate_evidence_benchmark
from app.evaluation.evidence_benchmark.models import GroundTruthAnswerRegion
from app.evaluation.evidence_expansion.freezing import (
    expansion_manifest_fingerprint,
    freeze_expansion_annotations,
    verify_frozen_expansion_annotations,
)
from app.evaluation.evidence_expansion.models import (
    EvidenceCandidateCategory,
    EvidenceContextTag,
    EvidenceExpansionManifest,
    EvidenceExpansionSample,
)
from app.evidence.models import EvidenceType
from app.evidence.models import TestEvidence as EvidenceTestResult
from app.ocr.models import BoundingBox

PAPER_ID = UUID("00000000-0000-0000-0000-000000000001")


def _manifest(root: Path) -> EvidenceExpansionManifest:
    source = (root / "canonical.png").resolve()
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(source), image)
    return EvidenceExpansionManifest(
        samples=(
            EvidenceExpansionSample(
                sample_id="evidence_v2_001",
                paper_alias="paper-a",
                page_number=1,
                test_number=2,
                source_image_path=source,
                source_image_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                page_width=200,
                page_height=100,
                region=BoundingBox(x=10, y=10, width=100, height=60),
                discovery_category=EvidenceCandidateCategory.STUDENT,
                context_tags=(EvidenceContextTag.SHORT_ANSWER,),
                discovery_reason="synthetic_candidate",
            ),
        )
    )


def _annotations() -> EvidenceAnnotationStore:
    return EvidenceAnnotationStore(
        annotations=(
            EvidenceAnnotation(
                sample_id="evidence_v2_001",
                evidence_type=EvidenceType.STUDENT_CANDIDATE,
                answer_status=AnswerAnnotationStatus.ANNOTATED,
                answer_regions=(
                    GroundTruthAnswerRegion(
                        bbox=BoundingBox(x=5, y=5, width=20, height=10)
                    ),
                ),
            ),
        )
    )


def _freeze(
    root: Path,
) -> tuple[EvidenceExpansionManifest, Path, Path, str]:
    manifest = _manifest(root)
    samples = root / "samples"
    samples.mkdir()
    (samples / "evidence_v2_001.png").write_bytes(b"synthetic-private-image")
    annotations = _annotations()
    snapshot, provenance, safe = freeze_expansion_annotations(
        manifest,
        annotations,
        samples_root=samples,
        frozen_root=root / "frozen",
        private_root=root,
        phase_4c_4r_fingerprint="1" * 64,
        ocr_benchmark_fingerprint="2" * 64,
        corpus_sha256="3" * 64,
        canonical_snapshot_sha256="4" * 64,
        timestamp_factory=lambda: datetime(2026, 8, 15, 12, 0, tzinfo=UTC),
    )
    return manifest, snapshot, provenance, safe.annotation_fingerprint


def test_v2_fingerprint_is_deterministic_and_preserves_rectangle_order(
    tmp_path: Path,
) -> None:
    first = _annotations()
    changed = EvidenceAnnotationStore(
        annotations=(
            first.annotations[0].model_copy(
                update={
                    "answer_regions": (
                        GroundTruthAnswerRegion(
                            bbox=BoundingBox(x=6, y=5, width=20, height=10)
                        ),
                    )
                }
            ),
        )
    )
    manifest = _manifest(tmp_path)

    assert annotation_fingerprint(first) == annotation_fingerprint(first)
    assert annotation_fingerprint(first) != annotation_fingerprint(changed)
    assert expansion_manifest_fingerprint(manifest) == (
        expansion_manifest_fingerprint(manifest)
    )


def test_complete_v2_snapshot_recovers_geometry_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    manifest, snapshot, provenance, fingerprint = _freeze(tmp_path)

    recovered, safe = verify_frozen_expansion_annotations(
        manifest,
        snapshot,
        provenance,
        samples_root=tmp_path / "samples",
        private_root=tmp_path,
    )

    assert annotation_fingerprint(recovered) == fingerprint
    assert safe.annotation_count == 1
    assert safe.human_answer_box_count == 1
    assert recovered.annotations[0].answer_regions == (
        GroundTruthAnswerRegion(bbox=BoundingBox(x=5, y=5, width=20, height=10)),
    )
    with pytest.raises(EvidenceSeparationError, match="already exists"):
        freeze_expansion_annotations(
            manifest,
            _annotations(),
            samples_root=tmp_path / "samples",
            frozen_root=tmp_path / "frozen",
            private_root=tmp_path,
            phase_4c_4r_fingerprint="1" * 64,
            ocr_benchmark_fingerprint="2" * 64,
            corpus_sha256="3" * 64,
            canonical_snapshot_sha256="4" * 64,
        )


def test_v2_evaluation_uses_discovery_and_context_only_for_strata(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    sample = manifest.samples[0]
    prediction = EvidenceTestResult(
        paper_id=PAPER_ID,
        page_number=sample.page_number,
        test_number=sample.test_number or 1,
        region_bbox=sample.region,
    )

    report = evaluate_evidence_benchmark(
        manifest,
        _annotations(),
        {sample.sample_id: prediction},
    )

    assert report.samples[0].predicted_class is EvidenceType.UNKNOWN
    assert report.samples[0].categories == (
        EvidenceCandidateCategory.STUDENT.value,
        "detected_test",
        EvidenceContextTag.SHORT_ANSWER.value,
    )
    student = next(
        row
        for row in report.classification.per_class
        if row.evidence_type is EvidenceType.STUDENT_CANDIDATE
    )
    teacher = next(
        row
        for row in report.classification.per_class
        if row.evidence_type is EvidenceType.TEACHER_CANDIDATE
    )
    assert student.false_negatives == 1
    assert teacher.precision is None
    assert teacher.recall is None
    assert teacher.f1 is None

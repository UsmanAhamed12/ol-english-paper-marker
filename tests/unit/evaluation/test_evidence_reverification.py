"""Synthetic tests for session-scoped human annotation re-verification."""

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from app.core.exceptions import EvidenceSeparationError
from app.evaluation.evidence_benchmark.annotations import (
    AnswerAnnotationStatus,
    EvidenceAnnotation,
    EvidenceAnnotationStore,
)
from app.evaluation.evidence_benchmark.models import (
    EvidenceBenchmarkManifest,
    EvidenceBenchmarkSample,
    EvidenceSampleCategory,
)
from app.evaluation.evidence_benchmark.reverification import (
    EvidenceReverificationRepository,
    freeze_reverified_annotations,
    validate_reverification_dataset,
    verify_frozen_annotations,
)
from app.evidence.models import EvidenceType
from app.ocr.models import BoundingBox


def _manifest(root: Path) -> EvidenceBenchmarkManifest:
    return EvidenceBenchmarkManifest(
        samples=(
            EvidenceBenchmarkSample(
                sample_id="sample_001",
                paper_alias="paper-a",
                page_number=1,
                test_number=1,
                source_image_path=root / "source.png",
                page_width=200,
                page_height=100,
                region=BoundingBox(x=0, y=0, width=200, height=100),
                categories=(EvidenceSampleCategory.MIXED,),
            ),
        )
    )


def _annotation(kind: EvidenceType = EvidenceType.PRINTED) -> EvidenceAnnotation:
    return EvidenceAnnotation(
        sample_id="sample_001",
        evidence_type=kind,
        answer_status=AnswerAnnotationStatus.VERIFIED_EMPTY,
    )


def test_old_human_verified_state_does_not_preapprove_new_session(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    annotations = EvidenceAnnotationStore(annotations=(_annotation(),))
    repository = EvidenceReverificationRepository(
        root / "reverification_session.json", private_root=root
    )
    session = repository.initialize(
        annotations,
        session_id=UUID("00000000-0000-0000-0000-000000000001"),
    )

    assert session.reverified_annotations == ()
    assert repository.current_ids(annotations) == frozenset()


def test_reverification_tracks_exact_saved_semantics_and_validates_completion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    (root / "samples").mkdir(parents=True)
    (root / "samples" / "sample_001.png").write_bytes(b"synthetic")
    annotations = EvidenceAnnotationStore(annotations=(_annotation(),))
    repository = EvidenceReverificationRepository(
        root / "reverification_session.json", private_root=root
    )
    repository.initialize(annotations)
    session = repository.mark_reverified(_annotation())

    assert repository.current_ids(annotations) == frozenset({"sample_001"})
    validate_reverification_dataset(
        _manifest(root), annotations, session, root / "samples"
    )

    changed = EvidenceAnnotationStore(annotations=(_annotation(EvidenceType.UNKNOWN),))
    assert repository.current_ids(changed) == frozenset()
    with pytest.raises(EvidenceSeparationError, match="changed"):
        validate_reverification_dataset(
            _manifest(root), changed, session, root / "samples"
        )


def test_incomplete_reverification_is_not_ready(tmp_path: Path) -> None:
    root = tmp_path / "private"
    (root / "samples").mkdir(parents=True)
    (root / "samples" / "sample_001.png").write_bytes(b"synthetic")
    annotations = EvidenceAnnotationStore(annotations=(_annotation(),))
    repository = EvidenceReverificationRepository(
        root / "reverification_session.json", private_root=root
    )
    session = repository.initialize(annotations)

    with pytest.raises(EvidenceSeparationError, match="every expected"):
        validate_reverification_dataset(
            _manifest(root), annotations, session, root / "samples"
        )


def test_complete_snapshot_is_recoverable_verified_and_never_overwritten(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    (root / "samples").mkdir(parents=True)
    (root / "samples" / "sample_001.png").write_bytes(b"synthetic")
    manifest = _manifest(root)
    annotations = EvidenceAnnotationStore(annotations=(_annotation(),))
    repository = EvidenceReverificationRepository(
        root / "reverification_session.json", private_root=root
    )
    repository.initialize(annotations)
    session = repository.mark_reverified(_annotation())

    snapshot, provenance, safe = freeze_reverified_annotations(
        manifest,
        annotations,
        session,
        samples_root=root / "samples",
        frozen_root=root / "frozen",
        private_root=root,
        timestamp_factory=lambda: datetime(2026, 8, 14, 12, 0, tzinfo=UTC),
    )

    assert safe["annotation_count"] == 1
    assert safe["reverification_count"] == 1
    assert safe["verified_empty_count"] == 1
    assert safe["human_answer_box_count"] == 0
    assert safe["created_at_utc"] == "2026-08-14T12:00:00Z"
    assert json.loads(snapshot.read_text())["annotations"][0]["sample_id"] == (
        "sample_001"
    )
    assert (
        verify_frozen_annotations(
            manifest,
            snapshot,
            provenance,
            samples_root=root / "samples",
            private_root=root,
        )
        == safe["annotation_fingerprint"]
    )

    with pytest.raises(EvidenceSeparationError, match="already exists"):
        freeze_reverified_annotations(
            manifest,
            annotations,
            session,
            samples_root=root / "samples",
            frozen_root=root / "frozen",
            private_root=root,
        )

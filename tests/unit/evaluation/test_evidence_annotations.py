"""Synthetic tests for private human visual annotations."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.exceptions import EvidenceSeparationError
from app.evaluation.evidence_benchmark.annotations import (
    AnswerAnnotationStatus,
    EvidenceAnnotation,
    EvidenceAnnotationRepository,
    EvidenceAnnotationStore,
    annotation_fingerprint,
    freeze_annotation_provenance,
    validate_annotation_dataset,
)
from app.evaluation.evidence_benchmark.models import (
    EvidenceBenchmarkManifest,
    EvidenceBenchmarkSample,
    EvidenceSampleCategory,
    GroundTruthAnswerRegion,
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
                source_image_path=root / "synthetic.png",
                page_width=500,
                page_height=400,
                region=BoundingBox(x=50, y=60, width=300, height=200),
                categories=(EvidenceSampleCategory.MIXED,),
            ),
        )
    )


def _rectangle() -> GroundTruthAnswerRegion:
    return GroundTruthAnswerRegion(bbox=BoundingBox(x=10, y=20, width=100, height=50))


def test_annotation_requires_explicit_nonempty_or_verified_empty_state() -> None:
    with pytest.raises(ValidationError, match="at least one rectangle"):
        EvidenceAnnotation(
            sample_id="sample_001",
            evidence_type=EvidenceType.UNKNOWN,
            answer_status=AnswerAnnotationStatus.ANNOTATED,
        )

    verified_empty = EvidenceAnnotation(
        sample_id="sample_001",
        evidence_type=EvidenceType.PRINTED,
        answer_status=AnswerAnnotationStatus.VERIFIED_EMPTY,
    )
    assert verified_empty.answer_regions == ()

    with pytest.raises(ValidationError, match="cannot contain rectangles"):
        EvidenceAnnotation(
            sample_id="sample_001",
            evidence_type=EvidenceType.STUDENT_CANDIDATE,
            answer_status=AnswerAnnotationStatus.VERIFIED_EMPTY,
            answer_regions=(_rectangle(),),
        )


def test_repository_saves_and_replaces_only_explicit_human_labels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    repository = EvidenceAnnotationRepository(
        _manifest(root), root / "annotations.json", private_root=root
    )
    first = EvidenceAnnotation(
        sample_id="sample_001",
        evidence_type=EvidenceType.STUDENT_CANDIDATE,
        answer_status=AnswerAnnotationStatus.ANNOTATED,
        answer_regions=(_rectangle(),),
    )
    store = repository.save(first)
    second = EvidenceAnnotation(
        sample_id="sample_001",
        evidence_type=EvidenceType.UNKNOWN,
        answer_status=AnswerAnnotationStatus.VERIFIED_EMPTY,
    )
    replaced = repository.save(second)

    assert store.annotations == (first,)
    assert replaced.annotations == (second,)
    assert repository.completion() == (1, 1)
    payload = json.loads(repository.path.read_text(encoding="utf-8"))
    assert "transcription" not in json.dumps(payload)


def test_repository_backs_up_every_semantic_replacement_byte_for_byte(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    timestamps = iter(
        (
            datetime(2026, 8, 14, 12, 0, tzinfo=UTC),
            datetime(2026, 8, 14, 12, 1, tzinfo=UTC),
        )
    )
    repository = EvidenceAnnotationRepository(
        _manifest(root),
        root / "annotations.json",
        private_root=root,
        timestamp_factory=lambda: next(timestamps),
    )
    first = EvidenceAnnotation(
        sample_id="sample_001",
        evidence_type=EvidenceType.STUDENT_CANDIDATE,
        answer_status=AnswerAnnotationStatus.ANNOTATED,
        answer_regions=(_rectangle(),),
    )
    repository.save(first)
    first_bytes = repository.path.read_bytes()

    repository.save(first)
    assert list((root / "backups").glob("*.json")) == []

    second = first.model_copy(update={"evidence_type": EvidenceType.UNKNOWN})
    repository.save(second)
    first_backup = list((root / "backups").glob("*.json"))
    assert len(first_backup) == 1
    assert first_backup[0].read_bytes() == first_bytes
    assert (
        annotation_fingerprint(EvidenceAnnotationStore(annotations=(first,)))
        in first_backup[0].name
    )
    assert annotation_fingerprint(
        EvidenceAnnotationRepository(
            _manifest(root), first_backup[0], private_root=root
        ).load()
    ) == annotation_fingerprint(EvidenceAnnotationStore(annotations=(first,)))

    repository.save(first)
    assert len(list((root / "backups").glob("*.json"))) == 2


def test_repository_rejects_unknown_samples_and_out_of_crop_boxes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    repository = EvidenceAnnotationRepository(
        _manifest(root), root / "annotations.json", private_root=root
    )
    with pytest.raises(EvidenceSeparationError, match="unknown"):
        repository.save(
            EvidenceAnnotation(
                sample_id="sample_999",
                evidence_type=EvidenceType.UNKNOWN,
                answer_status=AnswerAnnotationStatus.VERIFIED_EMPTY,
            )
        )
    with pytest.raises(EvidenceSeparationError, match="exceeds"):
        repository.save(
            EvidenceAnnotation(
                sample_id="sample_001",
                evidence_type=EvidenceType.STUDENT_CANDIDATE,
                answer_status=AnswerAnnotationStatus.ANNOTATED,
                answer_regions=(
                    GroundTruthAnswerRegion(
                        bbox=BoundingBox(x=290, y=190, width=20, height=20)
                    ),
                ),
            )
        )


def test_repository_rejects_public_output_and_invalid_store(tmp_path: Path) -> None:
    root = tmp_path / "private"
    with pytest.raises(EvidenceSeparationError, match="private"):
        EvidenceAnnotationRepository(
            _manifest(root), tmp_path / "public.json", private_root=root
        )
    root.mkdir()
    path = root / "annotations.json"
    path.write_text("not json", encoding="utf-8")
    repository = EvidenceAnnotationRepository(_manifest(root), path, private_root=root)
    with pytest.raises(EvidenceSeparationError, match="invalid"):
        repository.load()


@pytest.mark.parametrize("coordinate", [1.5, "student text"])
def test_repository_rejects_noninteger_or_textual_geometry(
    tmp_path: Path, coordinate: object
) -> None:
    root = tmp_path / "private"
    root.mkdir()
    path = root / "annotations.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "annotations": [
                    {
                        "sample_id": "sample_001",
                        "evidence_type": "student_candidate",
                        "answer_status": "annotated",
                        "answer_regions": [
                            {
                                "bbox": {
                                    "x": coordinate,
                                    "y": 0,
                                    "width": 10,
                                    "height": 10,
                                }
                            }
                        ],
                        "human_verified": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    repository = EvidenceAnnotationRepository(_manifest(root), path, private_root=root)

    with pytest.raises(EvidenceSeparationError, match="invalid"):
        repository.load()


def test_annotation_fingerprint_is_deterministic_and_label_sensitive(
    tmp_path: Path,
) -> None:
    first = EvidenceAnnotation(
        sample_id="sample_001",
        evidence_type=EvidenceType.PRINTED,
        answer_status=AnswerAnnotationStatus.VERIFIED_EMPTY,
    )
    changed = EvidenceAnnotation(
        sample_id="sample_001",
        evidence_type=EvidenceType.UNKNOWN,
        answer_status=AnswerAnnotationStatus.VERIFIED_EMPTY,
    )
    store = EvidenceAnnotationStore(annotations=(first,))

    assert annotation_fingerprint(store) == annotation_fingerprint(store)
    assert annotation_fingerprint(store) != annotation_fingerprint(
        EvidenceAnnotationStore(annotations=(changed,))
    )


def test_validation_and_freeze_record_only_safe_aggregate_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    (root / "samples").mkdir(parents=True)
    (root / "samples" / "sample_001.png").write_bytes(b"synthetic")
    store = EvidenceAnnotationStore(
        annotations=(
            EvidenceAnnotation(
                sample_id="sample_001",
                evidence_type=EvidenceType.STUDENT_CANDIDATE,
                answer_status=AnswerAnnotationStatus.ANNOTATED,
                answer_regions=(_rectangle(),),
            ),
        )
    )
    validate_annotation_dataset(_manifest(root), store, root / "samples")

    payload = freeze_annotation_provenance(
        store, root / "provenance.json", private_root=root
    )

    assert payload["sample_count"] == 1
    assert payload["samples_with_answer_regions"] == 1
    assert payload["human_answer_box_count"] == 1
    assert payload["verified_empty_samples"] == 0
    assert payload["class_distribution"] == {"student_candidate": 1}
    assert "source_image_path" not in json.dumps(payload)


def test_validation_rejects_missing_annotation_and_sample_image(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    with pytest.raises(EvidenceSeparationError, match="missing expected sample"):
        validate_annotation_dataset(
            _manifest(root), EvidenceAnnotationStore(), root / "samples"
        )

    store = EvidenceAnnotationStore(
        annotations=(
            EvidenceAnnotation(
                sample_id="sample_001",
                evidence_type=EvidenceType.UNKNOWN,
                answer_status=AnswerAnnotationStatus.VERIFIED_EMPTY,
            ),
        )
    )
    with pytest.raises(EvidenceSeparationError, match="image is missing"):
        validate_annotation_dataset(_manifest(root), store, root / "samples")

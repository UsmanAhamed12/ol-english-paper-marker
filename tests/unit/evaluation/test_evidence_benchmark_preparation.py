"""Synthetic evidence benchmark preparation and validation tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from app.evaluation.evidence_benchmark.models import (
    EvidenceBenchmarkManifest,
    EvidenceBenchmarkSample,
    EvidenceSampleCategory,
    GroundTruthAnswerRegion,
    HumanEvidenceStatus,
)
from app.evaluation.evidence_benchmark.preparation import (
    prepare_evidence_benchmark,
    safe_sample_filename,
)
from app.evidence.models import EvidenceType
from app.ocr.models import BoundingBox


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample(source: Path) -> EvidenceBenchmarkSample:
    return EvidenceBenchmarkSample(
        sample_id="sample_001",
        paper_alias="paper-a",
        page_number=1,
        test_number=1,
        source_image_path=source,
        page_width=400,
        page_height=300,
        region=BoundingBox(x=50, y=60, width=200, height=120),
        categories=(EvidenceSampleCategory.MIXED,),
    )


def test_safe_filename_and_crop_are_deterministic_and_private(tmp_path: Path) -> None:
    source = tmp_path / "private-source-name.png"
    image = np.full((300, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (50, 60), (249, 179), (10, 20, 30), -1)
    assert cv2.imwrite(str(source), image)
    sample = _sample(source)
    manifest = EvidenceBenchmarkManifest(samples=(sample,))
    before = _hash(source)

    first, worksheet = prepare_evidence_benchmark(manifest, tmp_path / "private")
    first_hash = _hash(first[0])
    second, _ = prepare_evidence_benchmark(manifest, tmp_path / "private")

    assert safe_sample_filename(sample) == "sample_001.png"
    assert first[0].name == "sample_001.png"
    assert "private-source-name" not in first[0].name
    assert _hash(second[0]) == first_hash
    assert _hash(source) == before
    assert worksheet.is_file()


def test_existing_worksheet_is_not_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    assert cv2.imwrite(str(source), np.full((300, 400, 3), 255, dtype=np.uint8))
    manifest = EvidenceBenchmarkManifest(samples=(_sample(source),))
    root = tmp_path / "private"
    _, worksheet = prepare_evidence_benchmark(manifest, root)
    worksheet.write_text("human work\n", encoding="utf-8")

    prepare_evidence_benchmark(manifest, root)

    assert worksheet.read_text(encoding="utf-8") == "human work\n"


def test_pending_sample_rejects_automatic_ground_truth(tmp_path: Path) -> None:
    payload = _sample(tmp_path / "private.png").model_dump()
    payload["ground_truth_evidence_type"] = EvidenceType.PRINTED

    with pytest.raises(ValidationError, match="Pending"):
        EvidenceBenchmarkSample.model_validate(payload)


def test_human_verified_sample_requires_explicit_answer_verification(
    tmp_path: Path,
) -> None:
    payload = _sample(tmp_path / "private.png").model_dump()
    payload["human_status"] = HumanEvidenceStatus.HUMAN_VERIFIED
    payload["ground_truth_evidence_type"] = EvidenceType.UNKNOWN

    with pytest.raises(ValidationError, match="requires both"):
        EvidenceBenchmarkSample.model_validate(payload)


def test_verified_empty_answer_list_is_explicitly_accepted(tmp_path: Path) -> None:
    payload = _sample(tmp_path / "private.png").model_dump()
    payload.update(
        human_status=HumanEvidenceStatus.HUMAN_VERIFIED,
        ground_truth_evidence_type=EvidenceType.PRINTED,
        answer_regions_verified=True,
        ground_truth_answer_regions=(),
    )

    verified = EvidenceBenchmarkSample.model_validate(payload)
    manifest = EvidenceBenchmarkManifest(samples=(verified,))

    assert manifest.benchmark_ready is True
    assert manifest.pending_count == 0


def test_answer_box_and_sample_region_validation(tmp_path: Path) -> None:
    payload = _sample(tmp_path / "private.png").model_dump()
    payload.update(
        human_status=HumanEvidenceStatus.HUMAN_VERIFIED,
        ground_truth_evidence_type=EvidenceType.STUDENT_CANDIDATE,
        answer_regions_verified=True,
        ground_truth_answer_regions=(
            GroundTruthAnswerRegion(
                bbox=BoundingBox(x=190, y=100, width=20, height=30)
            ),
        ),
    )

    with pytest.raises(ValidationError, match="answer box exceeds"):
        EvidenceBenchmarkSample.model_validate(payload)

    payload = _sample(tmp_path / "private.png").model_dump()
    payload["region"] = BoundingBox(x=350, y=250, width=100, height=100)
    with pytest.raises(ValidationError, match="region exceeds"):
        EvidenceBenchmarkSample.model_validate(payload)

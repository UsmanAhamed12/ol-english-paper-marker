"""Synthetic tests for teacher-focused candidate discovery and preparation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from app.core.exceptions import EvidenceSeparationError
from app.domain.models.paper import PaperPage
from app.evaluation.evidence_benchmark.annotation_web import (
    _benchmark_payload,
)
from app.evaluation.evidence_benchmark.annotations import EvidenceAnnotationRepository
from app.evaluation.teacher_evidence.discovery import (
    TeacherEvidenceProposal,
    contextual_crop,
    discover_teacher_evidence_candidates,
    select_teacher_evidence_candidates,
    suppress_teacher_candidate_duplicates,
)
from app.evaluation.teacher_evidence.manifest import load_teacher_evidence_manifest
from app.evaluation.teacher_evidence.models import (
    TeacherCandidateFeatures,
    TeacherDiscoveryCategory,
    TeacherDiscoverySignal,
    TeacherEvidenceManifest,
    TeacherEvidenceSample,
)
from app.evaluation.teacher_evidence.preparation import (
    prepare_teacher_artifacts,
    write_json_atomic,
)
from app.ocr.models import BoundingBox, OCRPageResult, OCRStructuredEvidence, OCRWord


def _features(*, chromatic: float = 0.0) -> TeacherCandidateFeatures:
    return TeacherCandidateFeatures(
        component_area_ratio=0.001,
        chromatic_foreground_ratio=chromatic,
        mean_saturation=chromatic,
        foreground_ratio=0.2,
        edge_density=0.1,
        local_whitespace_ratio=0.8,
        margin_proximity=0.7,
        ocr_proximity=0.5,
        nearby_ocr_words=1,
        angled_line_count=2,
    )


def _proposal(
    *, alias: str = "paper-a", x: int = 20, score: float = 0.8
) -> TeacherEvidenceProposal:
    return TeacherEvidenceProposal(
        paper_alias=alias,
        page_number=1,
        test_number=None,
        region=BoundingBox(x=x, y=10, width=620, height=380),
        candidate_component=BoundingBox(x=x + 200, y=150, width=40, height=30),
        category=TeacherDiscoveryCategory.COMPACT_GEOMETRY,
        signals=(TeacherDiscoverySignal.TICK_CROSS_GEOMETRY,),
        features=_features(),
        reason="tick_cross_correction_risk+multi_signal_context",
        score=score,
    )


def _sample(source: Path, digest: str) -> TeacherEvidenceSample:
    return TeacherEvidenceSample(
        sample_id="evidence_teacher_v1_001",
        paper_alias="paper-a",
        page_number=1,
        source_image_path=source.resolve(),
        source_image_sha256=digest,
        page_width=900,
        page_height=500,
        region=BoundingBox(x=10, y=10, width=620, height=380),
        candidate_component=BoundingBox(x=200, y=140, width=40, height=30),
        discovery_category=TeacherDiscoveryCategory.CHROMATIC,
        discovery_signals=(TeacherDiscoverySignal.CHROMATIC_INK,),
        features=_features(chromatic=0.8),
        selection_rank=1,
        discovery_reason="chromatic_ink_risk+multi_signal_context",
    )


def test_models_are_pending_and_enforce_geometry(tmp_path: Path) -> None:
    source = (tmp_path / "canonical.png").resolve()
    source.write_bytes(b"synthetic")
    sample = _sample(source, hashlib.sha256(b"synthetic").hexdigest())
    manifest = TeacherEvidenceManifest(samples=(sample,))
    assert manifest.pending_count == 1
    assert manifest.benchmark_ready is False
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    assert load_teacher_evidence_manifest(manifest_path) == manifest
    with pytest.raises(ValidationError, match="component"):
        TeacherEvidenceSample.model_validate(
            {
                **sample.model_dump(),
                "candidate_component": {"x": 700, "y": 20, "width": 20, "height": 20},
            }
        )


def test_context_crop_is_bounded_and_deterministic() -> None:
    component = BoundingBox(x=0, y=0, width=20, height=20)
    first = contextual_crop(component, 800, 500)
    assert first == contextual_crop(component, 800, 500)
    assert first.x == 0 and first.y == 0
    assert first.x + first.width <= 800
    assert first.y + first.height <= 500


def test_discovery_uses_color_as_hint_not_ground_truth(tmp_path: Path) -> None:
    image = np.full((500, 900, 3), 255, dtype=np.uint8)
    cv2.line(image, (50, 200), (95, 245), (0, 0, 200), 7)
    cv2.line(image, (95, 200), (50, 245), (0, 0, 200), 7)
    source = tmp_path / "page.png"
    assert cv2.imwrite(str(source), image)
    page = PaperPage(
        paper_id=uuid4(),
        page_number=1,
        image_path=source.resolve(),
        width=900,
        height=500,
    )
    word = OCRWord(
        text="Printed",
        confidence=0.95,
        bbox=BoundingBox(x=200, y=180, width=100, height=30),
        block_number=1,
        paragraph_number=1,
        line_number=1,
        word_number=1,
    )
    ocr = OCRPageResult(
        paper_id=page.paper_id,
        page_number=1,
        source_image_path=page.image_path,
        raw_text="Printed",
        normalized_text="Printed",
        confidence=None,
        provider="fake",
        model_version="test",
        processing_duration_ms=1,
        evidence=OCRStructuredEvidence(words=(word,), layout_text="Printed"),
    )
    proposals = discover_teacher_evidence_candidates(
        page, ocr, image, paper_alias="paper-a"
    )
    assert proposals
    assert any(
        TeacherDiscoverySignal.CHROMATIC_INK in proposal.signals
        for proposal in proposals
    )
    assert all(not hasattr(proposal, "evidence_type") for proposal in proposals)


def test_overlap_suppression_and_selection_are_deterministic() -> None:
    duplicate = (_proposal(score=0.9), _proposal(x=24, score=0.7))
    assert len(suppress_teacher_candidate_duplicates(duplicate)) == 1
    pool = tuple(
        _proposal(alias=f"paper-{chr(97 + index)}", x=20, score=0.8 - index / 100)
        for index in range(12)
    )
    first = select_teacher_evidence_candidates(
        pool, target_count=10, maximum_per_paper=1
    )
    assert first == select_teacher_evidence_candidates(
        pool, target_count=10, maximum_per_paper=1
    )
    assert len(first) == 10


def test_private_materialization_preserves_source_and_starts_unlabeled(
    tmp_path: Path,
) -> None:
    image = np.full((500, 900, 3), 245, dtype=np.uint8)
    source = tmp_path / "canonical.png"
    assert cv2.imwrite(str(source), image)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = TeacherEvidenceManifest(samples=(_sample(source, before),))
    private = tmp_path / "private"
    assert prepare_teacher_artifacts(manifest, private) == (1, 0)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert (private / "samples" / "evidence_teacher_v1_001.png").is_file()
    repository = EvidenceAnnotationRepository(
        manifest, private / "annotations.json", private_root=private
    )
    server = SimpleNamespace(
        manifest=manifest,
        private_root=private.resolve(),
        repository=repository,
        reverification_repository=None,
    )
    payload = _benchmark_payload(cast(Any, server))
    assert payload["completed"] == 0
    assert payload["samples"][0]["annotation"] is None
    assert "source_image_path" not in str(payload)


def test_duplicate_crop_rejected_and_metadata_write_is_atomic(tmp_path: Path) -> None:
    image = np.full((500, 900, 3), 245, dtype=np.uint8)
    source = tmp_path / "canonical.png"
    assert cv2.imwrite(str(source), image)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    first = _sample(source, digest)
    second = first.model_copy(
        update={"sample_id": "evidence_teacher_v1_002", "selection_rank": 2}
    )
    manifest = TeacherEvidenceManifest(samples=(first, second))
    with pytest.raises(EvidenceSeparationError, match="duplicate crop"):
        prepare_teacher_artifacts(manifest, tmp_path / "private")
    metadata = tmp_path / "private" / "metadata.json"
    write_json_atomic(metadata, {"status": "pending"})
    assert metadata.read_text(encoding="utf-8").endswith("\n")
    assert not (metadata.parent / ".metadata.json.tmp").exists()

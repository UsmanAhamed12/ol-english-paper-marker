"""Synthetic tests for Phase 4C.5A candidate discovery and preparation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from app.domain.models.paper import PaperPage
from app.evaluation.evidence_benchmark.annotation_web import (
    _benchmark_payload,
    create_annotation_server,
)
from app.evaluation.evidence_benchmark.annotations import (
    EvidenceAnnotationRepository,
)
from app.evaluation.evidence_expansion.discovery import (
    DiscoveredEvidenceCandidate,
    discover_page_candidates,
    reject_overlapping_candidates,
    select_balanced_candidates,
)
from app.evaluation.evidence_expansion.models import (
    EvidenceCandidateCategory,
    EvidenceContextTag,
    EvidenceExpansionManifest,
    EvidenceExpansionSample,
)
from app.evaluation.evidence_expansion.preparation import (
    prepare_expansion_artifacts,
    write_json_atomic,
)
from app.ocr.models import (
    BoundingBox,
    OCRPageResult,
    OCRStructuredEvidence,
    OCRWord,
)


def _page(path: Path, *, width: int = 1800, height: int = 700) -> PaperPage:
    return PaperPage(
        paper_id=uuid4(),
        page_number=2,
        image_path=path.resolve(),
        width=width,
        height=height,
    )


def _ocr(page: PaperPage) -> OCRPageResult:
    words = (
        OCRWord(
            text="Printed",
            confidence=0.96,
            bbox=BoundingBox(x=80, y=100, width=110, height=35),
            block_number=1,
            paragraph_number=1,
            line_number=1,
            word_number=1,
        ),
        OCRWord(
            text="question",
            confidence=0.94,
            bbox=BoundingBox(x=210, y=100, width=130, height=35),
            block_number=1,
            paragraph_number=1,
            line_number=1,
            word_number=2,
        ),
        OCRWord(
            text="unclear",
            confidence=0.18,
            bbox=BoundingBox(x=1320, y=350, width=160, height=70),
            block_number=2,
            paragraph_number=1,
            line_number=1,
            word_number=1,
        ),
    )
    return OCRPageResult(
        paper_id=page.paper_id,
        page_number=page.page_number,
        source_image_path=page.image_path,
        raw_text="Printed question\nunclear",
        normalized_text="Printed question\nunclear",
        confidence=None,
        provider="fake",
        model_version="test",
        processing_duration_ms=1,
        evidence=OCRStructuredEvidence(words=words, layout_text="Printed question"),
    )


def _candidate(
    category: EvidenceCandidateCategory,
    *,
    alias: str = "paper-a",
    x: int = 0,
    score: float = 0.8,
    page_number: int = 1,
) -> DiscoveredEvidenceCandidate:
    return DiscoveredEvidenceCandidate(
        paper_alias=alias,
        page_number=page_number,
        test_number=None,
        bbox=BoundingBox(x=x, y=20, width=200, height=100),
        category=category,
        context_tags=(),
        reason="synthetic_candidate",
        score=score,
    )


def _sample(source: Path, source_hash: str) -> EvidenceExpansionSample:
    return EvidenceExpansionSample(
        sample_id="evidence_v2_001",
        paper_alias="paper-a",
        page_number=1,
        source_image_path=source.resolve(),
        source_image_sha256=source_hash,
        page_width=120,
        page_height=80,
        region=BoundingBox(x=10, y=10, width=70, height=50),
        discovery_category=EvidenceCandidateCategory.TEACHER,
        context_tags=(EvidenceContextTag.COLORED_INK,),
        discovery_reason="synthetic_candidate",
    )


def test_candidate_discovery_uses_local_evidence_and_valid_crop_bounds(
    tmp_path: Path,
) -> None:
    image = np.full((700, 1800, 3), 255, dtype=np.uint8)
    cv2.putText(image, "PRINTED", (80, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.line(image, (30, 500), (80, 540), (0, 0, 180), 8)
    source = tmp_path / "page.png"
    assert cv2.imwrite(str(source), image)
    page = _page(source)

    first = discover_page_candidates(page, _ocr(page), image, paper_alias="paper-a")
    second = discover_page_candidates(page, _ocr(page), image, paper_alias="paper-a")

    assert first == second
    assert {item.category for item in first} >= {
        EvidenceCandidateCategory.STUDENT,
        EvidenceCandidateCategory.TEACHER,
        EvidenceCandidateCategory.BLANK,
    }
    assert all(item.bbox.x + item.bbox.width <= page.width for item in first)
    assert all(item.bbox.y + item.bbox.height <= page.height for item in first)


def test_candidate_discovery_rejects_invalid_image_geometry(tmp_path: Path) -> None:
    page = _page(tmp_path / "missing.png")
    with pytest.raises(ValueError, match="does not match"):
        discover_page_candidates(
            page,
            _ocr(page),
            np.zeros((50, 50, 3), dtype=np.uint8),
            paper_alias="paper-a",
        )


def test_overlap_rejection_and_balanced_sampling_are_deterministic() -> None:
    overlapping = (
        _candidate(EvidenceCandidateCategory.PRINTED, score=0.9),
        _candidate(EvidenceCandidateCategory.STUDENT, x=5, score=0.7),
    )
    assert len(reject_overlapping_candidates(overlapping)) == 1
    pool = tuple(
        _candidate(
            category,
            alias=f"paper-{letter}",
            x=index * 210,
            page_number=category_index + 1,
        )
        for category_index, category in enumerate(EvidenceCandidateCategory)
        for index, letter in enumerate("abc")
    )
    quotas = {category: 2 for category in EvidenceCandidateCategory}
    assert select_balanced_candidates(pool, quotas=quotas) == (
        select_balanced_candidates(pool, quotas=quotas)
    )
    assert len(select_balanced_candidates(pool, quotas=quotas)) == 10


def test_expansion_models_enforce_schema_bounds_and_pending_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "canonical.png"
    source.write_bytes(b"synthetic")
    sample = _sample(source, hashlib.sha256(b"synthetic").hexdigest())
    manifest = EvidenceExpansionManifest(samples=(sample,))
    assert manifest.schema_version == "2.0"
    assert manifest.pending_count == 1
    assert manifest.benchmark_ready is False
    with pytest.raises(ValidationError, match="exceeds"):
        EvidenceExpansionSample.model_validate(
            {
                **sample.model_dump(),
                "region": {"x": 100, "y": 0, "width": 30, "height": 20},
            }
        )


def test_private_preparation_is_atomic_and_preserves_source_hash(
    tmp_path: Path,
) -> None:
    image = np.full((80, 120, 3), 240, dtype=np.uint8)
    source = tmp_path / "canonical.png"
    assert cv2.imwrite(str(source), image)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = EvidenceExpansionManifest(samples=(_sample(source, before),))
    private = tmp_path / "private"

    written, duplicates = prepare_expansion_artifacts(manifest, private)

    assert (written, duplicates) == (1, 0)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert (private / "samples" / "evidence_v2_001.png").is_file()
    assert (private / "overlays" / "evidence_v2_001.png").is_file()
    write_json_atomic(private / "metadata.json", {"pending": True})
    assert not tuple(private.glob(".*.tmp"))


def test_preparation_rejects_changed_canonical_source(tmp_path: Path) -> None:
    image = np.full((80, 120, 3), 240, dtype=np.uint8)
    source = tmp_path / "canonical.png"
    assert cv2.imwrite(str(source), image)
    manifest = EvidenceExpansionManifest(samples=(_sample(source, "0" * 64),))
    with pytest.raises(Exception, match="source hash changed"):
        prepare_expansion_artifacts(manifest, tmp_path / "private")


def test_v2_labeler_does_not_preselect_discovery_class(tmp_path: Path) -> None:
    private = tmp_path / "private"
    samples = private / "samples"
    samples.mkdir(parents=True)
    source = tmp_path / "canonical.png"
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(source), image)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    sample = _sample(source, source_hash)
    assert cv2.imwrite(str(samples / "evidence_v2_001.png"), image[10:60, 10:80])
    manifest = EvidenceExpansionManifest(samples=(sample,))
    repository = EvidenceAnnotationRepository(
        manifest,
        private / "annotations.json",
        private_root=private,
    )
    server = create_annotation_server(manifest, private, repository, port=0)
    try:
        payload = _benchmark_payload(server)
    finally:
        server.server_close()
    record = payload["samples"][0]
    assert record["annotation"] is None
    assert "teacher_mark_risk_candidate" not in record["categories"]
    assert payload["completed"] == 0


def test_private_repository_rejects_path_outside_dataset(tmp_path: Path) -> None:
    source = tmp_path / "canonical.png"
    source.write_bytes(b"synthetic")
    manifest = EvidenceExpansionManifest(
        samples=(_sample(source, hashlib.sha256(b"synthetic").hexdigest()),)
    )
    with pytest.raises(Exception, match="private evaluation storage"):
        EvidenceAnnotationRepository(
            manifest,
            tmp_path / "outside.json",
            private_root=tmp_path / "private",
        )

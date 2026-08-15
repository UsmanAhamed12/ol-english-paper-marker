"""Local HTTP boundary tests for the private visual labeler."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.evaluation.evidence_benchmark.annotation_web import (
    ANNOTATION_HTML,
    LOOPBACK_HOST,
    EvidenceAnnotationServer,
    create_annotation_server,
)
from app.evaluation.evidence_benchmark.annotations import (
    AnswerAnnotationStatus,
    EvidenceAnnotation,
    EvidenceAnnotationRepository,
)
from app.evaluation.evidence_benchmark.models import (
    EvidenceBenchmarkManifest,
    EvidenceBenchmarkSample,
    EvidenceSampleCategory,
)
from app.evaluation.evidence_benchmark.reverification import (
    EvidenceReverificationRepository,
)
from app.evidence.models import EvidenceType
from app.ocr.models import BoundingBox


def _manifest(root: Path) -> EvidenceBenchmarkManifest:
    return EvidenceBenchmarkManifest(
        samples=(
            EvidenceBenchmarkSample(
                sample_id="sample_001",
                paper_alias="paper-a",
                page_number=2,
                test_number=3,
                source_image_path=root / "source.png",
                page_width=300,
                page_height=200,
                region=BoundingBox(x=10, y=20, width=180, height=100),
                categories=(EvidenceSampleCategory.SHORT_ANSWER,),
            ),
        )
    )


@contextmanager
def _running_server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    root = tmp_path / "private"
    (root / "samples").mkdir(parents=True)
    (root / "overlays").mkdir()
    image = np.full((100, 180, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(root / "samples" / "sample_001.png"), image)
    assert cv2.imwrite(str(root / "overlays" / "sample_001.png"), image)
    manifest = _manifest(root)
    repository = EvidenceAnnotationRepository(
        manifest, root / "annotations.json", private_root=root
    )
    server = create_annotation_server(manifest, root, repository, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{LOOPBACK_HOST}:{server.server_port}", root
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _get(url: str) -> tuple[int, bytes, dict[str, str]]:
    with urllib.request.urlopen(url) as response:
        return response.status, response.read(), dict(response.headers)


def test_interface_contains_required_local_annotation_controls() -> None:
    assert "Original crop" in ANNOTATION_HTML
    assert "Existing prediction overlay" in ANNOTATION_HTML
    assert "Explicitly verified empty" in ANNOTATION_HTML
    assert "Previous sample" in ANNOTATION_HTML
    assert "Save and next" in ANNOTATION_HTML
    assert "Delete" in ANNOTATION_HTML
    assert "Clear rectangles" in ANNOTATION_HTML
    assert "student_candidate" in ANNOTATION_HTML
    assert "Re-verified / Save" in ANNOTATION_HTML
    assert "Current saved class" in ANNOTATION_HTML
    assert "Re-verified in this session: YES" in ANNOTATION_HTML


def test_server_is_loopback_only_and_api_omits_private_source_path(
    tmp_path: Path,
) -> None:
    with _running_server(tmp_path) as (url, _):
        status, body, headers = _get(f"{url}/api/benchmark")
        payload = json.loads(body)
        assert status == 200
        assert payload["local_only"] is True
        assert payload["completed"] == 0
        assert payload["samples"][0]["sample_id"] == "sample_001"
        assert "source_image_path" not in body.decode()
        assert headers["Cache-Control"] == "no-store"
        status, image, _ = _get(f"{url}/assets/samples/sample_001.png")
        assert status == 200
        assert image.startswith(b"\x89PNG")


def test_server_persists_valid_rectangles_and_rejects_invalid_requests(
    tmp_path: Path,
) -> None:
    with _running_server(tmp_path) as (url, root):
        payload = json.dumps(
            {
                "evidence_type": "student_candidate",
                "answer_status": "annotated",
                "answer_regions": [
                    {"bbox": {"x": 10, "y": 12, "width": 80, "height": 30}}
                ],
                "human_verified": True,
            }
        ).encode()
        request = urllib.request.Request(
            f"{url}/api/annotations/sample_001",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read())
        assert result["completed"] == 1
        assert (root / "annotations.json").is_file()

        invalid = urllib.request.Request(
            f"{url}/api/annotations/sample_001",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(invalid)
        except urllib.error.HTTPError as error:
            assert error.code == 400
        else:
            raise AssertionError("Invalid annotation request unexpectedly succeeded")


def test_asset_path_traversal_is_rejected(tmp_path: Path) -> None:
    with _running_server(tmp_path) as (url, _):
        try:
            urllib.request.urlopen(f"{url}/assets/samples/%2e%2e")
        except urllib.error.HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError("Traversal request unexpectedly succeeded")


def test_server_type_is_explicit() -> None:
    assert issubclass(EvidenceAnnotationServer, object)


def test_reverification_mode_starts_unapproved_and_requires_explicit_action(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    (root / "samples").mkdir(parents=True)
    image = np.full((100, 180, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(root / "samples" / "sample_001.png"), image)
    manifest = _manifest(root)
    annotations = EvidenceAnnotationRepository(
        manifest, root / "annotations.json", private_root=root
    )
    annotations.save(
        EvidenceAnnotation(
            sample_id="sample_001",
            evidence_type=EvidenceType.PRINTED,
            answer_status=AnswerAnnotationStatus.VERIFIED_EMPTY,
        )
    )
    reverification = EvidenceReverificationRepository(
        root / "reverification_session.json", private_root=root
    )
    reverification.initialize(annotations.load())
    server = create_annotation_server(
        manifest,
        root,
        annotations,
        port=0,
        reverification_repository=reverification,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://{LOOPBACK_HOST}:{server.server_port}"
    try:
        _, body, _ = _get(f"{url}/api/benchmark")
        state = json.loads(body)
        assert state["mode"] == "reverification"
        assert state["completed"] == 0
        assert state["samples"][0]["reverified"] is False

        missing_confirmation = urllib.request.Request(
            f"{url}/api/annotations/sample_001",
            data=json.dumps(
                {
                    "evidence_type": "printed",
                    "answer_status": "verified_empty",
                    "answer_regions": [],
                    "human_verified": True,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as captured:
            urllib.request.urlopen(missing_confirmation)
        assert captured.value.code == 400

        confirmed = urllib.request.Request(
            f"{url}/api/annotations/sample_001",
            data=json.dumps(
                {
                    "evidence_type": "printed",
                    "answer_status": "verified_empty",
                    "answer_regions": [],
                    "human_verified": True,
                    "reverified": True,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(confirmed) as response:
            saved = json.loads(response.read())
        assert saved["completed"] == 1
        assert saved["ready"] is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

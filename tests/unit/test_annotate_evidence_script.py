"""Tests for the private evidence-labeling launcher."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import annotate_evidence


class _FakeServer:
    server_port = 4321
    served = False
    closed = False

    def serve_forever(self) -> None:
        self.served = True
        raise KeyboardInterrupt

    def server_close(self) -> None:
        self.closed = True


def test_launcher_reports_only_local_url_and_safe_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_name = "private-student-source.png"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "samples": [
                    {
                        "sample_id": "sample_001",
                        "paper_alias": "paper-a",
                        "page_number": 1,
                        "test_number": 1,
                        "source_image_path": str(tmp_path / private_name),
                        "page_width": 200,
                        "page_height": 100,
                        "region": {"x": 0, "y": 0, "width": 200, "height": 100},
                        "categories": ["mixed_evidence_candidate"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    fake = _FakeServer()
    monkeypatch.setattr(
        annotate_evidence,
        "create_annotation_server",
        lambda *args, **kwargs: fake,
    )

    assert annotate_evidence.main(("--manifest", str(manifest))) == 0

    output = capsys.readouterr().out
    assert "http://127.0.0.1:4321" in output
    assert "Progress: 0/1" in output
    assert private_name not in output
    assert fake.served is True
    assert fake.closed is True


def test_launcher_starts_a_separate_reverification_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "samples": [
                    {
                        "sample_id": "sample_001",
                        "paper_alias": "paper-a",
                        "page_number": 1,
                        "test_number": 1,
                        "source_image_path": str(tmp_path / "source.png"),
                        "page_width": 200,
                        "page_height": 100,
                        "region": {"x": 0, "y": 0, "width": 200, "height": 100},
                        "categories": ["mixed_evidence_candidate"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    fake = _FakeServer()
    monkeypatch.setattr(
        annotate_evidence,
        "create_annotation_server",
        lambda *args, **kwargs: fake,
    )

    assert annotate_evidence.main(("--manifest", str(manifest), "--reverify")) == 0

    output = capsys.readouterr().out
    assert "Re-verified: 0/1" in output
    assert (tmp_path / "reverification_session.json").is_file()


def test_launcher_supports_pending_teacher_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = (tmp_path / "canonical.png").resolve()
    assert cv2.imwrite(str(source), np.full((100, 200, 3), 255, dtype=np.uint8))
    import hashlib

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": "evidence-teacher-v1",
                "candidate_generation_version": "teacher-risk-discovery-v1",
                "samples": [
                    {
                        "sample_id": "evidence_teacher_v1_001",
                        "paper_alias": "paper-a",
                        "page_number": 1,
                        "source_image_path": str(source),
                        "source_image_sha256": hashlib.sha256(
                            source.read_bytes()
                        ).hexdigest(),
                        "page_width": 200,
                        "page_height": 100,
                        "region": {"x": 0, "y": 0, "width": 200, "height": 100},
                        "candidate_component": {
                            "x": 20,
                            "y": 20,
                            "width": 20,
                            "height": 20,
                        },
                        "discovery_category": "chromatic_ink_risk",
                        "discovery_signals": ["chromatic_ink"],
                        "features": {
                            "component_area_ratio": 0.01,
                            "chromatic_foreground_ratio": 0.5,
                            "mean_saturation": 0.5,
                            "foreground_ratio": 0.1,
                            "edge_density": 0.1,
                            "local_whitespace_ratio": 0.8,
                            "margin_proximity": 0.5,
                            "ocr_proximity": 0.2,
                            "nearby_ocr_words": 0,
                            "angled_line_count": 1,
                        },
                        "selection_rank": 1,
                        "discovery_reason": "chromatic_ink_risk+multi_signal_context",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    fake = _FakeServer()
    monkeypatch.setattr(
        annotate_evidence, "create_annotation_server", lambda *a, **k: fake
    )
    assert (
        annotate_evidence.main(
            ("--dataset", "evidence-teacher-v1", "--manifest", str(manifest))
        )
        == 0
    )
    assert "Progress: 0/1" in capsys.readouterr().out

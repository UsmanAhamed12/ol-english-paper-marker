"""Tests for privacy-safe evidence benchmark CLI validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare_evidence_benchmark import main


def test_validation_never_prints_private_source_paths(
    tmp_path: Path,
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
                        "page_width": 400,
                        "page_height": 300,
                        "region": {"x": 10, "y": 20, "width": 100, "height": 80},
                        "categories": ["mixed_evidence_candidate"],
                        "human_status": "pending",
                        "ground_truth_evidence_type": None,
                        "answer_regions_verified": False,
                        "ground_truth_answer_regions": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert main(("validate", "--manifest", str(manifest))) == 0

    output = capsys.readouterr().out
    assert private_name not in output
    assert "ground_truth_evidence_type" not in output
    assert '"pending_count": 1' in output
    assert '"benchmark_ready": false' in output

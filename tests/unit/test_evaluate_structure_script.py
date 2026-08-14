"""Tests for safe structure-benchmark CLI behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_structure import main


def test_validate_prints_only_safe_manifest_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    private_source = tmp_path / "private-student-source.pdf"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "human_verified": True,
                "papers": [
                    {
                        "paper_alias": "paper-a",
                        "source_path": str(private_source),
                        "expected_page_count": 1,
                        "expected_markers": [
                            {
                                "test_number": 1,
                                "page_number": 1,
                                "bbox": {
                                    "x": 10,
                                    "y": 20,
                                    "width": 30,
                                    "height": 40,
                                },
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert main(("validate", "--manifest", str(manifest))) == 0

    output = capsys.readouterr().out
    assert "paper-a" in output
    assert "private-student-source" not in output
    assert "ground_truth" not in output

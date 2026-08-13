"""Tests for the validation-only OCR benchmark command."""

from __future__ import annotations

import json
from pathlib import Path

from pytest import CaptureFixture, MonkeyPatch

from app.evaluation.ocr_benchmark.models import GroundTruthStatus
from scripts import benchmark_ocr


def test_validate_command_reports_pending_manifest_without_private_text(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "ocr_benchmark_manifest.json"
    manifest_payload = json.loads(fixture.read_text(encoding="utf-8"))
    manifest_payload["samples"][0]["ground_truth_status"] = (
        GroundTruthStatus.PENDING.value
    )
    manifest_payload["samples"][0]["ground_truth_student_text"] = None
    pending_manifest = tmp_path / "pending.json"
    pending_manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["benchmark_ocr", "validate", str(pending_manifest)],
    )

    assert benchmark_ocr.main() == 0

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload == {
        "benchmark_ready": False,
        "ground_truth_fingerprint": None,
        "human_verified_count": 0,
        "human_verified_empty_count": 0,
        "ocr_executed": False,
        "pending_count": 1,
        "sample_count": 1,
        "schema_version": "1.0",
    }
    assert "This is synthetic text" not in output


def test_validate_ready_manifest_never_prints_private_text(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "ocr_benchmark_manifest.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    private_text = "synthetic-private-transcription"
    payload["samples"][0]["ground_truth_status"] = GroundTruthStatus.VERIFIED.value
    payload["samples"][0]["ground_truth_student_text"] = private_text
    manifest = tmp_path / "private.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["benchmark_ocr", "validate", str(manifest)])

    assert benchmark_ocr.main() == 0

    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["benchmark_ready"] is True
    assert report["ground_truth_fingerprint"] is not None
    assert private_text not in output

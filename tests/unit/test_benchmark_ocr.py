"""Tests for the validation-only OCR benchmark command."""

from __future__ import annotations

import json
from pathlib import Path

from pytest import CaptureFixture, MonkeyPatch

from scripts import benchmark_ocr


def test_validate_command_reports_only_safe_counts(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "ocr_benchmark_manifest.json"
    monkeypatch.setattr(
        "sys.argv",
        ["benchmark_ocr", "validate", str(fixture)],
    )

    assert benchmark_ocr.main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ocr_executed": False,
        "pending_samples": 0,
        "ready_samples": 1,
        "sample_count": 1,
        "schema_version": "1.0",
    }

"""Tests for the validation-only OCR benchmark command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import CaptureFixture, MonkeyPatch

from app.core.exceptions import ConfigurationError
from app.evaluation.ocr_benchmark.manifest import load_manifest
from app.evaluation.ocr_benchmark.models import GroundTruthStatus
from app.ocr.preprocessing.models import PreprocessingVariant
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


def test_run_cli_rejects_unsupported_provider() -> None:
    with pytest.raises(SystemExit):
        benchmark_ocr.parse_args(
            ["run", "--provider", "cloud", "--model", "synthetic:1b"]
        )


def test_run_cli_requires_a_model() -> None:
    args = benchmark_ocr.parse_args(["run", "--provider", "ollama"])
    assert args.model is None


def test_run_cli_accepts_tesseract_without_model() -> None:
    args = benchmark_ocr.parse_args(["run", "--provider", "tesseract", "--smoke"])
    assert args.provider == "tesseract"
    assert args.model is None
    assert args.smoke is True
    assert args.preprocessing == "none"


def test_run_cli_accepts_fixed_preprocessing_variant() -> None:
    args = benchmark_ocr.parse_args(
        ["run", "--provider", "tesseract", "--preprocessing", "grayscale-denoise"]
    )
    assert args.preprocessing == "grayscale-denoise"
    assert (
        benchmark_ocr.tesseract_result_name(PreprocessingVariant.GRAYSCALE_DENOISE)
        == "tesseract-grayscale-denoise"
    )


def test_run_cli_rejects_unknown_preprocessing_variant() -> None:
    with pytest.raises(SystemExit):
        benchmark_ocr.parse_args(
            ["run", "--provider", "tesseract", "--preprocessing", "aggressive"]
        )


def test_cli_does_not_print_private_validation_errors(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    private_text = "synthetic-private-marker"
    invalid = tmp_path / "private-invalid.json"
    invalid.write_text(private_text, encoding="utf-8")

    assert benchmark_ocr.main(["validate", str(invalid)]) == 2

    output = capsys.readouterr().out
    assert private_text not in output
    assert "validation or execution failed" in output


def test_frozen_benchmark_requires_all_eight_samples() -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "ocr_benchmark_manifest.json"
    manifest = load_manifest(fixture)

    with pytest.raises(ConfigurationError, match="not ready"):
        benchmark_ocr._require_frozen_benchmark(
            manifest,
            benchmark_ocr.FROZEN_BENCHMARK_FINGERPRINT,
        )


def test_empty_behavior_counts_only_executed_results() -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "ocr_benchmark_manifest.json"
    manifest = load_manifest(fixture)

    assert benchmark_ocr._verified_empty_behavior(manifest, ()) == {
        "sample_count": 0,
        "empty_predictions": 0,
        "nonempty_predictions": 0,
        "failed": 0,
    }

"""Tests for loading synthetic OCR benchmark manifests."""

from __future__ import annotations

from pathlib import Path

from app.evaluation.ocr_benchmark.manifest import load_manifest


def test_loads_synthetic_manifest_fixture() -> None:
    fixture = Path(__file__).parents[2] / "fixtures" / "ocr_benchmark_manifest.json"

    manifest = load_manifest(fixture)

    assert manifest.schema_version == "1.0"
    assert len(manifest.samples) == 1
    assert manifest.samples[0].paper_alias == "synthetic-paper-a"
    assert manifest.samples[0].is_ready

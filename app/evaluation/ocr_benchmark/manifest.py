"""Load and validate private OCR benchmark manifests."""

from __future__ import annotations

from pathlib import Path

from app.evaluation.ocr_benchmark.models import BenchmarkManifest


def load_manifest(path: Path) -> BenchmarkManifest:
    """Load a UTF-8 JSON manifest through its typed schema."""

    return BenchmarkManifest.model_validate_json(path.read_text(encoding="utf-8"))

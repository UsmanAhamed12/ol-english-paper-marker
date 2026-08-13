"""Validate a private OCR benchmark manifest without running OCR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.ocr_benchmark.manifest import load_manifest


def parse_args() -> argparse.Namespace:
    """Parse the validation-only Phase 4A command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate",))
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path("data/evaluation/ocr/benchmark_manifest.json"),
    )
    return parser.parse_args()


def main() -> int:
    """Validate the manifest and print only non-sensitive aggregate counts."""

    args = parse_args()
    manifest = load_manifest(args.manifest)
    ready_samples = sum(sample.is_ready for sample in manifest.samples)
    payload = {
        "schema_version": manifest.schema_version,
        "sample_count": len(manifest.samples),
        "ready_samples": ready_samples,
        "pending_samples": len(manifest.samples) - ready_samples,
        "ocr_executed": False,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

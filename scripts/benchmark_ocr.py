"""Validate a private OCR benchmark manifest without running OCR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.ocr_benchmark.ground_truth import ground_truth_fingerprint
from app.evaluation.ocr_benchmark.manifest import load_manifest
from app.evaluation.ocr_benchmark.models import GroundTruthStatus


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
    human_verified_count = sum(
        sample.ground_truth_status is GroundTruthStatus.VERIFIED
        for sample in manifest.samples
    )
    human_verified_empty_count = sum(
        sample.ground_truth_status is GroundTruthStatus.VERIFIED_EMPTY
        for sample in manifest.samples
    )
    pending_count = sum(not sample.is_ready for sample in manifest.samples)
    payload = {
        "schema_version": manifest.schema_version,
        "sample_count": len(manifest.samples),
        "human_verified_count": human_verified_count,
        "human_verified_empty_count": human_verified_empty_count,
        "pending_count": pending_count,
        "benchmark_ready": manifest.is_ready,
        "ground_truth_fingerprint": (
            ground_truth_fingerprint(manifest) if manifest.is_ready else None
        ),
        "ocr_executed": False,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Freeze human-entered OCR ground truth into the private manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.ocr_benchmark.ground_truth import (
    freeze_ground_truth,
    ground_truth_fingerprint,
)
from app.evaluation.ocr_benchmark.manifest import load_manifest

_VERIFIED_EMPTY_IDS = frozenset(
    {
        "teacher-annotation-risk",
        "sparse-answer-page",
    }
)
_NON_TEXT_NOTES = {
    "paragraph-writing": (
        "Human review confirmed a graphical correct/tick mark opposite the textual "
        "student x; the graphical mark is excluded from textual ground truth."
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse private worksheet and manifest paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worksheet",
        type=Path,
        default=Path("data/evaluation/ocr/transcription_worksheet.md"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/evaluation/ocr/benchmark_manifest.json"),
    )
    return parser.parse_args()


def main() -> int:
    """Freeze authoritative text and print safe metadata only."""

    args = parse_args()
    manifest = load_manifest(args.manifest)
    worksheet = args.worksheet.read_text(encoding="utf-8")
    frozen = freeze_ground_truth(
        manifest,
        worksheet,
        verified_empty_sample_ids=_VERIFIED_EMPTY_IDS,
        notes_by_sample_id=_NON_TEXT_NOTES,
    )
    args.manifest.write_text(
        frozen.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    payload = {
        "sample_count": len(frozen.samples),
        "human_verified_count": sum(
            sample.ground_truth_status.value == "human_verified"
            for sample in frozen.samples
        ),
        "human_verified_empty_count": sum(
            sample.ground_truth_status.value == "human_verified_empty"
            for sample in frozen.samples
        ),
        "pending_count": sum(not sample.is_ready for sample in frozen.samples),
        "benchmark_ready": frozen.is_ready,
        "ground_truth_fingerprint": ground_truth_fingerprint(frozen),
        "ocr_executed": False,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

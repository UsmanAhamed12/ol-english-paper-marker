"""Thin CLI for the local end-to-end paper-marking service."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from app.core.config import get_settings
from app.orchestration.factory import create_local_paper_marking_service
from app.orchestration.models import FinalPaperStatus, PaperMarkingRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--scheme", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path beneath configured ignored runtime storage",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    result = create_local_paper_marking_service(settings).mark_paper(
        PaperMarkingRequest(
            student_pdf=args.student,
            marking_scheme_pdf=args.scheme,
            output_path=args.output,
        )
    )
    print(
        json.dumps(
            {
                "aligned_tests": len(result.aligned_tests),
                "confirmed_marks": str(result.total_confirmed_marks),
                "final_score_available": result.final_score is not None,
                "known_maximum": str(result.known_maximum_marks),
                "review_items": len(result.review_items),
                "run_id": str(result.run_id),
                "status": result.status.value,
                "total_duration_ms": round(result.total_duration_ms, 3),
                "unresolved_marks": str(result.unresolved_marks),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if result.status is FinalPaperStatus.FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

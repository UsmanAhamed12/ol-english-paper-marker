"""Canonical private serialization for subjective and combined results."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.exceptions import MarkingEngineError
from app.marking.subjective.models import (
    CombinedPaperMarkingResult,
    SubjectivePaperMarkingResult,
)


def subjective_result_json(
    result: SubjectivePaperMarkingResult | CombinedPaperMarkingResult,
) -> str:
    return json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_private_subjective_result(
    result: SubjectivePaperMarkingResult | CombinedPaperMarkingResult,
    output_path: Path,
    *,
    private_root: Path,
) -> Path:
    root = private_root.expanduser().resolve()
    target = output_path.expanduser().resolve()
    if not target.is_relative_to(root):
        raise MarkingEngineError("Subjective marking results must remain private")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(subjective_result_json(result) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target

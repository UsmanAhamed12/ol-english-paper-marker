"""Deterministic private serialization for marking results."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.exceptions import MarkingEngineError
from app.marking.models import PaperMarkingResult, TestMarkResult


def marking_result_json(result: PaperMarkingResult | TestMarkResult) -> str:
    """Return canonical JSON while preserving original student evidence."""

    return json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_private_marking_result(
    result: PaperMarkingResult,
    output_path: Path,
    *,
    private_root: Path,
) -> Path:
    """Atomically write a private result only beneath approved storage."""

    root = private_root.expanduser().resolve()
    target = output_path.expanduser().resolve()
    if not target.is_relative_to(root):
        raise MarkingEngineError("Marking results must remain private")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(marking_result_json(result) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target

"""Deterministic private serialization for final orchestration results."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.exceptions import OrchestrationError
from app.orchestration.models import FinalPaperMarkingResult


def final_result_json(result: FinalPaperMarkingResult) -> str:
    """Return canonical JSON without mutating internal evidence."""

    return json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_private_final_result(
    result: FinalPaperMarkingResult,
    output_path: Path,
    *,
    private_root: Path,
) -> Path:
    """Atomically write only beneath an explicitly approved private root."""

    root = private_root.expanduser().resolve()
    target = output_path.expanduser().resolve()
    if not target.is_relative_to(root):
        raise OrchestrationError("Final marking results must remain private")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(final_result_json(result) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target

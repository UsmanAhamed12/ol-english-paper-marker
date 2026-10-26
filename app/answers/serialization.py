"""Deterministic private JSON serialization for extracted papers."""

from __future__ import annotations

import json
from pathlib import Path

from app.answers.models import StructuredStudentPaper
from app.core.exceptions import AnswerExtractionError


def structured_paper_json(paper: StructuredStudentPaper) -> str:
    """Return canonical deterministic JSON without altering OCR evidence."""

    return json.dumps(
        paper.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def write_structured_paper(
    paper: StructuredStudentPaper,
    output_path: Path,
    *,
    private_root: Path,
) -> Path:
    """Atomically persist runtime JSON only beneath an approved private root."""

    root = private_root.expanduser().resolve()
    target = output_path.expanduser().resolve()
    if not target.is_relative_to(root):
        raise AnswerExtractionError("Structured answer output must remain private")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(structured_paper_json(paper) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target

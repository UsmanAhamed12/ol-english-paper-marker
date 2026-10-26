"""Deterministic private serialization for marking-scheme contracts."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from app.core.exceptions import MarkingSchemeIngestionError
from app.marking_scheme.models import AlignedPaper, StructuredMarkingScheme


def marking_scheme_json(scheme: StructuredMarkingScheme) -> str:
    """Return canonical semantic JSON for one structured marking scheme."""

    return _canonical_json(scheme)


def aligned_paper_json(alignment: AlignedPaper) -> str:
    """Return canonical semantic JSON for one exact-number alignment."""

    return _canonical_json(alignment)


def write_private_json(
    payload: StructuredMarkingScheme | AlignedPaper,
    output_path: Path,
    *,
    private_root: Path,
) -> Path:
    """Atomically write marking data only beneath approved private storage."""

    root = private_root.expanduser().resolve()
    target = output_path.expanduser().resolve()
    if not target.is_relative_to(root):
        raise MarkingSchemeIngestionError("Marking-scheme output must remain private")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(_canonical_json(payload) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def _canonical_json(value: BaseModel) -> str:
    return json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

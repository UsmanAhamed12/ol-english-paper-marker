"""Load and fingerprint frozen historical reference datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.evaluation.models import HistoricalEvaluationDataset


def load_dataset(path: Path) -> HistoricalEvaluationDataset:
    """Load UTF-8 JSON and validate all teacher reference marks."""

    return HistoricalEvaluationDataset.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def dataset_fingerprint(dataset: HistoricalEvaluationDataset) -> str:
    """Return a stable SHA-256 digest of the complete reference dataset."""

    canonical = json.dumps(
        dataset.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

"""Deterministic serialization for safe aggregate evaluation reports."""

from __future__ import annotations

import json

from app.evaluation.models import HistoricalEvaluationReport


def serialize_report(report: HistoricalEvaluationReport) -> str:
    """Serialize report fields in stable key order without source evidence."""

    return json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )

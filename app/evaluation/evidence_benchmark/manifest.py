"""Load private evidence benchmark candidates through typed validation."""

from pathlib import Path

from app.evaluation.evidence_benchmark.models import EvidenceBenchmarkManifest


def load_evidence_manifest(path: Path) -> EvidenceBenchmarkManifest:
    """Load a UTF-8 private manifest without logging sensitive paths."""

    return EvidenceBenchmarkManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )

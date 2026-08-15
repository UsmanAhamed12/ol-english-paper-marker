"""Private evidence-v2 manifest loading."""

from pathlib import Path

from app.evaluation.evidence_expansion.models import EvidenceExpansionManifest


def load_evidence_expansion_manifest(path: Path) -> EvidenceExpansionManifest:
    """Load the private versioned manifest without logging source paths."""

    return EvidenceExpansionManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )

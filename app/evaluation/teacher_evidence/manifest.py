"""Private teacher-evidence manifest loading."""

from pathlib import Path

from app.evaluation.teacher_evidence.models import TeacherEvidenceManifest


def load_teacher_evidence_manifest(path: Path) -> TeacherEvidenceManifest:
    """Load teacher-risk candidates without logging private source paths."""

    return TeacherEvidenceManifest.model_validate_json(path.read_text(encoding="utf-8"))

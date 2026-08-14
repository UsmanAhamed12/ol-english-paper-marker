"""Load private structure ground truth through strict typed validation."""

from pathlib import Path

from app.evaluation.structure_benchmark.models import StructureBenchmarkManifest


def load_structure_manifest(path: Path) -> StructureBenchmarkManifest:
    """Load a UTF-8 JSON structure manifest."""

    return StructureBenchmarkManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )

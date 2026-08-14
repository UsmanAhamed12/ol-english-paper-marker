"""Private benchmark contracts and metrics for exam structure detection."""

from app.evaluation.structure_benchmark.models import StructureBenchmarkManifest
from app.evaluation.structure_benchmark.runner import evaluate_structure

__all__ = ["StructureBenchmarkManifest", "evaluate_structure"]

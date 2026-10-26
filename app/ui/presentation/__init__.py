"""Typed view models and deterministic orchestration-result mapping."""

from app.ui.presentation.mapper import map_final_result
from app.ui.presentation.models import ResultViewModel

__all__ = ["ResultViewModel", "map_final_result"]

"""Deterministic conditional routing for the paper-marking graph."""

from __future__ import annotations

from typing import Literal

from app.graph.state import PaperMarkingState


def route_after_aggregation(
    state: PaperMarkingState,
) -> Literal["completed", "review_required"]:
    """Route unresolved combined scores to the review-required completion node."""

    result = state.get("combined_result")
    if result is None:
        raise ValueError("Combined result is required before workflow routing")
    return "completed" if result.final_score is not None else "review_required"

"""Deterministic reading order for safe student answer regions."""

from __future__ import annotations

from app.evidence.models import StudentAnswerRegion


def order_answer_regions(
    regions: tuple[StudentAnswerRegion, ...],
) -> tuple[StudentAnswerRegion, ...]:
    """Order regions by page, vertical position, then horizontal position."""

    return tuple(
        sorted(
            regions,
            key=lambda item: (
                item.page_number,
                item.bbox.y,
                item.bbox.x,
                item.bbox.height,
                item.bbox.width,
            ),
        )
    )


def combine_fragment_text(values: tuple[str, ...]) -> str:
    """Join non-empty normalized fragments without rewriting their contents."""

    return "\n".join(value for value in values if value)

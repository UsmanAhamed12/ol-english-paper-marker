"""Small deterministic formatting helpers with no score inference."""

from __future__ import annotations

from decimal import Decimal


def format_marks(value: Decimal) -> str:
    """Format Decimal marks without binary floating-point conversion."""

    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal("1")))
    return format(normalized, "f")


def humanize_code(code: str) -> str:
    """Convert a stable safe code to a readable label."""

    base, separator, count = code.partition(":")
    label = base.replace("_", " ").strip().capitalize()
    return f"{label} ({count})" if separator and count.isdigit() else label


def percentage(confirmed: Decimal, maximum: Decimal) -> str | None:
    """Return a percentage only for a legitimate non-zero final maximum."""

    if maximum <= 0:
        return None
    value = (confirmed / maximum) * Decimal("100")
    return f"{value.quantize(Decimal('0.1'))}%"

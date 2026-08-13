"""Small, identity-free PDF fixtures generated during tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pymupdf
import pytest


@pytest.fixture
def make_pdf() -> Callable[[Path, int], Path]:
    """Return a factory for deterministic, tiny PDFs with visible page labels."""

    def factory(path: Path, page_count: int) -> Path:
        document = pymupdf.open()  # type: ignore[no-untyped-call]
        for page_number in range(1, page_count + 1):
            page = document.new_page(width=144, height=216)
            page.insert_text((18, 36), f"Test page {page_number}")
        document.save(str(path), garbage=4, deflate=True)  # type: ignore[no-untyped-call]
        document.close()  # type: ignore[no-untyped-call]
        return path

    return factory

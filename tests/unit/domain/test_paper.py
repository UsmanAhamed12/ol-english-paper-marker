"""Tests for paper and rendered-page domain invariants."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.domain.models.paper import PaperDocument, PaperPage


def _document_data(tmp_path: Path) -> dict[str, object]:
    return {
        "paper_id": uuid4(),
        "source_path": (tmp_path / "source.pdf").resolve(),
        "original_filename": "source.pdf",
        "page_count": 2,
        "file_size_bytes": 100,
        "sha256": "a" * 64,
    }


def test_document_requires_absolute_source_path(tmp_path: Path) -> None:
    data = _document_data(tmp_path)
    data["source_path"] = Path("relative.pdf")

    with pytest.raises(ValidationError, match="source_path must be absolute"):
        PaperDocument.model_validate(data)


def test_rendered_pages_must_be_complete_ordered_and_owned(tmp_path: Path) -> None:
    data = _document_data(tmp_path)
    paper_id = data["paper_id"]
    assert isinstance(paper_id, UUID)
    page = PaperPage(
        paper_id=paper_id,
        page_number=2,
        image_path=(tmp_path / "page_0002.png").resolve(),
        width=100,
        height=200,
    )
    data["pages"] = (page,)

    with pytest.raises(ValidationError, match="page count"):
        PaperDocument.model_validate(data)


def test_page_requires_absolute_image_path() -> None:
    with pytest.raises(ValidationError, match="image_path must be absolute"):
        PaperPage(
            paper_id=uuid4(),
            page_number=1,
            image_path=Path("page_0001.png"),
            width=100,
            height=200,
        )

"""Behavioral tests for validation, loading, and page rendering."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

import pymupdf
import pytest

from app.core.exceptions import (
    InvalidPDFError,
    PDFPageLimitError,
    PDFRenderingError,
    PDFTooLargeError,
)
from app.domain.models.paper import PaperDocument
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.pdf_renderer import PDFRenderer
from app.ingestion.validators import PDFValidator

PdfFactory = Callable[[Path, int], Path]


def _validator(
    *,
    max_file_size_bytes: int = 1_000_000,
    max_pages: int = 10,
) -> PDFValidator:
    return PDFValidator(
        max_file_size_bytes=max_file_size_bytes,
        max_pages=max_pages,
    )


def test_valid_pdf_is_accepted_and_page_count_is_extracted(
    tmp_path: Path,
    make_pdf: PdfFactory,
) -> None:
    source = make_pdf(tmp_path / "paper.pdf", 2)

    result = _validator().validate(source)

    assert result.source_path == source.resolve()
    assert result.page_count == 2
    assert result.file_size_bytes == source.stat().st_size


@pytest.mark.parametrize("source_name", ["missing.pdf", "missing.PDF"])
def test_missing_pdf_is_rejected(tmp_path: Path, source_name: str) -> None:
    with pytest.raises(InvalidPDFError, match="does not exist"):
        _validator().validate(tmp_path / source_name)


def test_directory_and_unsupported_extension_are_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "directory.pdf"
    directory.mkdir()
    text_file = tmp_path / "paper.txt"
    text_file.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(InvalidPDFError, match="regular file"):
        _validator().validate(directory)
    with pytest.raises(InvalidPDFError, match="extension"):
        _validator().validate(text_file)


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "empty.pdf"
    source.touch()

    with pytest.raises(InvalidPDFError, match="empty"):
        _validator().validate(source)


@pytest.mark.parametrize(
    "content",
    [b"this is not a PDF", b"%PDF-1.7\ncorrupt and incomplete"],
    ids=["fake", "corrupt"],
)
def test_unreadable_pdf_content_is_rejected(tmp_path: Path, content: bytes) -> None:
    source = tmp_path / "unreadable.pdf"
    source.write_bytes(content)

    with pytest.raises(InvalidPDFError, match="readable PDF"):
        _validator().validate(source)


def test_password_protected_pdf_is_rejected(
    tmp_path: Path,
    make_pdf: PdfFactory,
) -> None:
    source = make_pdf(tmp_path / "plain.pdf", 1)
    protected = tmp_path / "protected.pdf"
    aes_256 = pymupdf.PDF_ENCRYPT_AES_256  # type: ignore[attr-defined]
    document = pymupdf.open(str(source))  # type: ignore[no-untyped-call]
    document.save(  # type: ignore[no-untyped-call]
        str(protected),
        encryption=aes_256,
        owner_pw="owner-test-password",
        user_pw="user-test-password",
    )
    document.close()  # type: ignore[no-untyped-call]

    with pytest.raises(InvalidPDFError, match="readable PDF"):
        _validator().validate(protected)


def test_zero_page_pdf_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "zero-pages.pdf"
    source.write_bytes(b"%PDF placeholder for parser boundary test")

    class ZeroPageDocument:
        is_pdf = True
        needs_pass = False
        page_count = 0

        def __enter__(self) -> ZeroPageDocument:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(pymupdf, "open", lambda *_args, **_kwargs: ZeroPageDocument())

    with pytest.raises(InvalidPDFError, match="at least one page"):
        _validator().validate(source)


def test_page_limit_is_enforced(tmp_path: Path, make_pdf: PdfFactory) -> None:
    source = make_pdf(tmp_path / "two-pages.pdf", 2)

    with pytest.raises(PDFPageLimitError, match="page-count"):
        _validator(max_pages=1).validate(source)


def test_file_size_limit_is_enforced(tmp_path: Path, make_pdf: PdfFactory) -> None:
    source = make_pdf(tmp_path / "paper.pdf", 1)

    with pytest.raises(PDFTooLargeError, match="file-size"):
        _validator(max_file_size_bytes=source.stat().st_size - 1).validate(source)


def test_loader_builds_identity_safe_deterministic_metadata(
    tmp_path: Path,
    make_pdf: PdfFactory,
) -> None:
    source = make_pdf(tmp_path / "student name.pdf", 2)
    loader = PDFLoader(_validator())

    first = loader.load(source)
    second = loader.load(source)
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    assert isinstance(first.paper_id, UUID)
    assert first.paper_id != second.paper_id
    assert first.paper_id.hex not in first.original_filename
    assert first.page_count == 2
    assert first.file_size_bytes == source.stat().st_size
    assert first.sha256 == second.sha256 == expected_hash
    assert first.pages == ()


def test_renderer_produces_complete_valid_page_metadata(
    tmp_path: Path,
    make_pdf: PdfFactory,
) -> None:
    source = make_pdf(tmp_path / "unsafe student name.pdf", 2)
    document = PDFLoader(_validator()).load(source)
    runtime_dir = tmp_path / "runtime"

    rendered = PDFRenderer(
        runtime_data_dir=runtime_dir,
        render_dpi=150,
    ).render(document)

    assert isinstance(rendered, PaperDocument)
    assert len(rendered.pages) == document.page_count == 2
    assert [page.page_number for page in rendered.pages] == [1, 2]
    assert [page.image_path.name for page in rendered.pages] == [
        "page_0001.png",
        "page_0002.png",
    ]
    assert rendered.pages[0].image_path.parent == (
        runtime_dir.resolve() / document.paper_id.hex / "pages"
    )
    assert "unsafe student name.pdf" not in str(rendered.pages[0].image_path)
    for page in rendered.pages:
        assert page.image_format == "PNG"
        assert page.width > 0
        assert page.height > 0
        assert page.image_path.is_file()
        reopened = pymupdf.Pixmap(  # type: ignore[no-untyped-call]
            str(page.image_path)
        )
        assert (reopened.width, reopened.height) == (page.width, page.height)


def test_renderer_rejects_source_changed_after_loading(
    tmp_path: Path,
    make_pdf: PdfFactory,
) -> None:
    source = make_pdf(tmp_path / "paper.pdf", 1)
    document = PDFLoader(_validator()).load(source)
    source.write_bytes(source.read_bytes() + b"changed")

    with pytest.raises(PDFRenderingError, match="changed"):
        PDFRenderer(
            runtime_data_dir=tmp_path / "runtime",
            render_dpi=150,
        ).render(document)


def test_invalid_limits_are_rejected() -> None:
    with pytest.raises(ValueError, match="limits"):
        PDFValidator(max_file_size_bytes=0, max_pages=1)
    with pytest.raises(ValueError, match="render_dpi"):
        PDFRenderer(runtime_data_dir=Path("runtime"), render_dpi=700)

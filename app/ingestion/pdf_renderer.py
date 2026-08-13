"""Render validated PDF pages to stable, identity-safe PNG files."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pymupdf

from app.core.exceptions import PDFRenderingError
from app.domain.models.paper import PaperDocument, PaperPage
from app.ingestion.pdf_loader import _sha256


class PDFRenderer:
    """Render every page without OCR preprocessing or text recognition."""

    def __init__(self, *, runtime_data_dir: Path, render_dpi: int) -> None:
        if render_dpi < 72 or render_dpi > 600:
            raise ValueError("render_dpi must be between 72 and 600")
        self._runtime_data_dir = runtime_data_dir
        self._render_dpi = render_dpi

    def render(self, document: PaperDocument) -> PaperDocument:
        """Return the immutable document with its complete rendered page set."""

        try:
            if not document.source_path.is_file():
                raise PDFRenderingError("PDF source is no longer available")
            if _sha256(document.source_path) != document.sha256:
                raise PDFRenderingError("PDF source changed after loading")
        except PDFRenderingError:
            raise
        except OSError as error:
            raise PDFRenderingError("PDF source could not be read") from error

        runtime_root = self._prepare_runtime_root()
        paper_dir = runtime_root / document.paper_id.hex
        pages_dir = paper_dir / "pages"
        if pages_dir.exists():
            raise PDFRenderingError("Rendered page output already exists")

        paper_dir.mkdir(parents=True, exist_ok=True)
        if not paper_dir.resolve().is_relative_to(runtime_root):
            raise PDFRenderingError("Unsafe runtime output path")

        try:
            rendered_dimensions = self._render_to_temporary_directory(
                document,
                runtime_root,
                pages_dir,
            )
        except PDFRenderingError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise PDFRenderingError("PDF pages could not be rendered") from error

        pages = tuple(
            PaperPage(
                paper_id=document.paper_id,
                page_number=page_number,
                image_path=(pages_dir / _page_filename(page_number)).resolve(),
                width=width,
                height=height,
            )
            for page_number, (width, height) in enumerate(
                rendered_dimensions,
                start=1,
            )
        )
        document_data = document.model_dump(exclude={"pages"})
        return PaperDocument.model_validate({**document_data, "pages": pages})

    def _prepare_runtime_root(self) -> Path:
        self._runtime_data_dir.mkdir(parents=True, exist_ok=True)
        return self._runtime_data_dir.resolve(strict=True)

    def _render_to_temporary_directory(
        self,
        paper: PaperDocument,
        runtime_root: Path,
        final_pages_dir: Path,
    ) -> list[tuple[int, int]]:
        dimensions: list[tuple[int, int]] = []
        with tempfile.TemporaryDirectory(
            dir=runtime_root,
            prefix=f".{paper.paper_id.hex}-",
        ) as temporary_directory:
            temporary_pages = Path(temporary_directory)
            # PyMuPDF's public constructor has no complete type annotation.
            with pymupdf.open(str(paper.source_path)) as source:  # type: ignore[no-untyped-call]
                if not source.is_pdf or source.needs_pass:
                    raise PDFRenderingError("PDF source is no longer readable")
                if source.page_count != paper.page_count:
                    raise PDFRenderingError("PDF page count changed after loading")

                for page_index, source_page in enumerate(source):
                    page_number = page_index + 1
                    output_path = temporary_pages / _page_filename(page_number)
                    pixmap = source_page.get_pixmap(
                        dpi=self._render_dpi,
                        alpha=False,
                        annots=True,
                    )
                    pixmap.save(str(output_path))
                    rendered_image = pymupdf.Pixmap(  # type: ignore[no-untyped-call]
                        str(output_path)
                    )
                    if rendered_image.width <= 0 or rendered_image.height <= 0:
                        raise PDFRenderingError("Rendered page has invalid dimensions")
                    dimensions.append((rendered_image.width, rendered_image.height))

            temporary_pages.replace(final_pages_dir)
        return dimensions


def _page_filename(page_number: int) -> str:
    return f"page_{page_number:04d}.png"

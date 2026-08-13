"""Prepare private benchmark images and a blank transcription worksheet."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping
from pathlib import Path

import pymupdf
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from app.core.exceptions import OCRBenchmarkPreparationError
from app.domain.models.paper import PaperPage
from app.evaluation.ocr_benchmark.models import (
    BenchmarkManifest,
    BenchmarkRegion,
    SafeIdentifier,
)


class BenchmarkPaperSource(BaseModel):
    """Private mapping from a safe paper alias to a local source PDF."""

    model_config = ConfigDict(frozen=True)

    paper_alias: SafeIdentifier
    source_pdf_path: Path


class BenchmarkSourceMap(BaseModel):
    """Validated private source mappings required by a benchmark manifest."""

    model_config = ConfigDict(frozen=True)

    sources: tuple[BenchmarkPaperSource, ...]

    @model_validator(mode="after")
    def aliases_must_be_unique(self) -> BenchmarkSourceMap:
        aliases = [source.paper_alias for source in self.sources]
        if len(aliases) != len(set(aliases)):
            raise ValueError("benchmark paper aliases must be unique")
        return self


class PreparedBenchmarkSample(BaseModel):
    """Safe metadata for one materialized private benchmark image."""

    model_config = ConfigDict(frozen=True)

    sample_id: SafeIdentifier
    output_path: Path
    width: int
    height: int

    @field_validator("width", "height")
    @classmethod
    def dimensions_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("prepared image dimensions must be positive")
        return value


class BenchmarkPreparer:
    """Materialize safe benchmark images without changing canonical pages."""

    def __init__(
        self,
        *,
        evaluation_root: Path,
        samples_dir: Path,
        worksheet_path: Path,
    ) -> None:
        self._evaluation_root = evaluation_root.resolve()
        self._samples_dir = samples_dir.resolve()
        self._worksheet_path = worksheet_path.resolve()
        self._require_private_path(self._samples_dir)
        self._require_private_path(self._worksheet_path)

    def prepare(
        self,
        manifest: BenchmarkManifest,
        canonical_pages: Mapping[str, tuple[PaperPage, ...]],
    ) -> tuple[PreparedBenchmarkSample, ...]:
        """Prepare every sample deterministically in manifest order."""

        self._samples_dir.mkdir(parents=True, exist_ok=True)
        prepared: list[PreparedBenchmarkSample] = []
        canonical_hashes: dict[Path, str] = {}

        for index, sample in enumerate(manifest.samples, start=1):
            page = self._find_page(
                canonical_pages,
                sample.paper_alias,
                sample.page_number,
            )
            canonical_hashes.setdefault(page.image_path, _file_sha256(page.image_path))
            output_path = self._samples_dir / safe_sample_filename(index)
            width, height = _materialize_image(
                page.image_path,
                output_path,
                sample.region,
            )
            prepared.append(
                PreparedBenchmarkSample(
                    sample_id=sample.sample_id,
                    output_path=output_path,
                    width=width,
                    height=height,
                )
            )

        for path, original_hash in canonical_hashes.items():
            if _file_sha256(path) != original_hash:
                raise OCRBenchmarkPreparationError(
                    "Canonical rendered page changed during benchmark preparation"
                )

        if not self._worksheet_path.exists():
            self._worksheet_path.parent.mkdir(parents=True, exist_ok=True)
            self._worksheet_path.write_text(
                _build_worksheet(manifest),
                encoding="utf-8",
            )
        return tuple(prepared)

    def _require_private_path(self, path: Path) -> None:
        if not path.is_relative_to(self._evaluation_root):
            raise OCRBenchmarkPreparationError(
                "Benchmark output must remain under the private evaluation root"
            )

    @staticmethod
    def _find_page(
        canonical_pages: Mapping[str, tuple[PaperPage, ...]],
        paper_alias: str,
        page_number: int,
    ) -> PaperPage:
        pages = canonical_pages.get(paper_alias)
        if pages is None:
            raise OCRBenchmarkPreparationError(
                f"Canonical pages are unavailable for alias {paper_alias}"
            )
        for page in pages:
            if page.page_number == page_number:
                if not page.image_path.is_file():
                    raise OCRBenchmarkPreparationError(
                        f"Canonical page is unavailable for alias {paper_alias}"
                    )
                return page
        raise OCRBenchmarkPreparationError(
            f"Page {page_number} is unavailable for alias {paper_alias}"
        )


def safe_sample_filename(index: int) -> str:
    """Return a deterministic identity-safe filename for one-based index."""

    if index <= 0:
        raise ValueError("sample index must be positive")
    return f"sample_{index:03d}.png"


def _materialize_image(
    source_path: Path,
    output_path: Path,
    region: BenchmarkRegion | None,
) -> tuple[int, int]:
    """Copy a full page or create a derived crop through PyMuPDF."""

    try:
        source = pymupdf.Pixmap(str(source_path))  # type: ignore[no-untyped-call]
        source_width, source_height = source.width, source.height
        if region is not None and (
            region.x + region.width > source_width
            or region.y + region.height > source_height
        ):
            raise OCRBenchmarkPreparationError(
                "Benchmark region exceeds the canonical page dimensions"
            )

        temporary_path = output_path.with_name(f".{output_path.stem}.tmp.png")
        if region is None:
            shutil.copyfile(source_path, temporary_path)
            expected_dimensions = (source_width, source_height)
        else:
            _save_crop(source_path, temporary_path, region, source_width, source_height)
            expected_dimensions = (region.width, region.height)

        rendered = pymupdf.Pixmap(str(temporary_path))  # type: ignore[no-untyped-call]
        actual_dimensions = (rendered.width, rendered.height)
        if actual_dimensions != expected_dimensions:
            raise OCRBenchmarkPreparationError(
                "Prepared benchmark image has unexpected dimensions"
            )
        temporary_path.replace(output_path)
        return actual_dimensions
    except OCRBenchmarkPreparationError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise OCRBenchmarkPreparationError(
            "Benchmark image could not be prepared"
        ) from error


def _save_crop(
    source_path: Path,
    output_path: Path,
    region: BenchmarkRegion,
    source_width: int,
    source_height: int,
) -> None:
    """Render an exact pixel crop without touching the canonical image."""

    document = pymupdf.open()  # type: ignore[no-untyped-call]
    try:
        page = document.new_page(width=source_width, height=source_height)
        page.insert_image(  # type: ignore[no-untyped-call]
            page.rect,
            filename=str(source_path),
        )
        clip = pymupdf.Rect(  # type: ignore[no-untyped-call]
            region.x,
            region.y,
            region.x + region.width,
            region.y + region.height,
        )
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(1, 1),  # type: ignore[no-untyped-call]
            clip=clip,
            alpha=False,
        )
        pixmap.save(str(output_path))  # type: ignore[no-untyped-call]
    finally:
        document.close()  # type: ignore[no-untyped-call]


def _build_worksheet(manifest: BenchmarkManifest) -> str:
    """Create the deterministic blank human-transcription worksheet."""

    sections = [
        "# Private OCR benchmark transcription worksheet",
        "",
        "Transcribe student answer text only. Exclude printed examination content",
        "and teacher annotations. Preserve student spelling and grammar exactly.",
        "",
    ]
    for index, sample in enumerate(manifest.samples, start=1):
        region = (
            "Full page"
            if sample.region is None
            else (
                f"x={sample.region.x}, y={sample.region.y}, "
                f"width={sample.region.width}, height={sample.region.height}"
            )
        )
        sections.extend(
            [
                f"## Sample {index:03d}",
                "",
                f"Sample ID: {sample.sample_id}",
                f"Paper alias: {sample.paper_alias}",
                f"Page: {sample.page_number}",
                f"Difficulty: {sample.difficulty.value}",
                f"Category: {', '.join(sample.categories)}",
                "Teacher annotations present/expected: "
                + ("Yes" if sample.teacher_annotations_present else "No"),
                f"Region: {region}",
                "Ground truth student text:",
                "",
                "",
            ]
        )
    return "\n".join(sections)


def _file_sha256(path: Path) -> str:
    """Hash one image without loading it entirely into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

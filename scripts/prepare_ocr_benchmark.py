"""Materialize private OCR samples and a blank human worksheet."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from uuid import UUID, uuid5

import pymupdf

from app.core.config import Settings
from app.core.exceptions import OCRBenchmarkPreparationError
from app.domain.models.paper import PaperDocument, PaperPage
from app.evaluation.ocr_benchmark.manifest import load_manifest
from app.evaluation.ocr_benchmark.models import (
    BenchmarkManifest,
    OCRBenchmarkSample,
)
from app.evaluation.ocr_benchmark.preparation import (
    BenchmarkPreparer,
    BenchmarkSourceMap,
)
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.pdf_renderer import PDFRenderer
from app.ingestion.validators import PDFValidator

_BENCHMARK_NAMESPACE = UUID("6d3fe01b-519e-47d6-8ee4-ae992739c083")


def parse_args() -> argparse.Namespace:
    """Parse private preparation paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/evaluation/ocr/benchmark_manifest.json"),
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("data/evaluation/ocr/paper_sources.json"),
    )
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("data/evaluation/ocr"),
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path("data/runtime/ocr-benchmark"),
    )
    return parser.parse_args()


def main() -> int:
    """Render missing canonical pages, prepare samples, and report safe facts."""

    args = parse_args()
    manifest = load_manifest(args.manifest)
    source_map = BenchmarkSourceMap.model_validate_json(
        args.sources.read_text(encoding="utf-8")
    )
    canonical_pages = _canonical_pages(manifest, source_map, args.runtime_root)
    updated_manifest = _with_canonical_metadata(manifest, canonical_pages)
    args.manifest.write_text(
        updated_manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    preparer = BenchmarkPreparer(
        evaluation_root=args.evaluation_root,
        samples_dir=args.evaluation_root / "samples",
        worksheet_path=args.evaluation_root / "transcription_worksheet.md",
    )
    prepared = preparer.prepare(updated_manifest, canonical_pages)

    for sample in updated_manifest.samples:
        region = (
            "full-page"
            if sample.region is None
            else (
                f"{sample.region.x},{sample.region.y},"
                f"{sample.region.width},{sample.region.height}"
            )
        )
        print(
            f"sample_id={sample.sample_id} alias={sample.paper_alias} "
            f"page={sample.page_number} region={region} "
            f"difficulty={sample.difficulty.value} "
            f"categories={','.join(sample.categories)} "
            f"teacher_annotations={sample.teacher_annotations_present} "
            f"ready={sample.is_ready}"
        )
    print(f"prepared_samples={len(prepared)} ocr_executed=false")
    return 0


def _canonical_pages(
    manifest: BenchmarkManifest,
    source_map: BenchmarkSourceMap,
    runtime_root: Path,
) -> dict[str, tuple[PaperPage, ...]]:
    """Resolve pages through the Phase 2 validator, loader, and renderer."""

    sources = {source.paper_alias: source for source in source_map.sources}
    required_aliases = {sample.paper_alias for sample in manifest.samples}
    if missing := required_aliases - sources.keys():
        raise OCRBenchmarkPreparationError(
            f"Private source mappings are missing for aliases: {sorted(missing)}"
        )

    settings = Settings()
    validator = PDFValidator(
        max_file_size_bytes=settings.max_pdf_size_mb * 1024 * 1024,
        max_pages=settings.max_pdf_pages,
    )
    renderer = PDFRenderer(
        runtime_data_dir=runtime_root,
        render_dpi=settings.pdf_render_dpi,
    )
    pages_by_alias: dict[str, tuple[PaperPage, ...]] = {}
    for alias in sorted(required_aliases):
        paper_id = uuid5(_BENCHMARK_NAMESPACE, alias)
        loader = PDFLoader(
            validator,
            paper_id_factory=partial(_fixed_paper_id, paper_id),
        )
        document = loader.load(sources[alias].source_pdf_path)
        pages_by_alias[alias] = _render_or_load(document, renderer, runtime_root)
    return pages_by_alias


def _render_or_load(
    document: PaperDocument,
    renderer: PDFRenderer,
    runtime_root: Path,
) -> tuple[PaperPage, ...]:
    """Render once, then safely reuse the deterministic Phase 2 output."""

    paper_dir = runtime_root.resolve() / document.paper_id.hex
    pages_dir = paper_dir / "pages"
    digest_path = paper_dir / "source.sha256"
    if not pages_dir.exists():
        rendered = renderer.render(document)
        digest_path.write_text(document.sha256 + "\n", encoding="ascii")
        return rendered.pages
    if (
        not digest_path.is_file()
        or digest_path.read_text(encoding="ascii").strip() != document.sha256
    ):
        raise OCRBenchmarkPreparationError(
            "Existing canonical pages do not match the configured source"
        )

    expected_paths = [
        pages_dir / f"page_{page_number:04d}.png"
        for page_number in range(1, document.page_count + 1)
    ]
    if any(not path.is_file() for path in expected_paths):
        raise OCRBenchmarkPreparationError("Canonical rendered page set is incomplete")
    pages: list[PaperPage] = []
    for page_number, path in enumerate(expected_paths, start=1):
        width, height = _image_dimensions(path)
        pages.append(
            PaperPage(
                paper_id=document.paper_id,
                page_number=page_number,
                image_path=path.resolve(),
                width=width,
                height=height,
            )
        )
    return tuple(pages)


def _image_dimensions(path: Path) -> tuple[int, int]:
    pixmap = pymupdf.Pixmap(str(path))  # type: ignore[no-untyped-call]
    return pixmap.width, pixmap.height


def _fixed_paper_id(paper_id: UUID) -> UUID:
    return paper_id


def _with_canonical_metadata(
    manifest: BenchmarkManifest,
    canonical_pages: dict[str, tuple[PaperPage, ...]],
) -> BenchmarkManifest:
    """Refresh private image references and dimensions after Phase 2 rendering."""

    samples: list[OCRBenchmarkSample] = []
    for sample in manifest.samples:
        page = next(
            (
                candidate
                for candidate in canonical_pages[sample.paper_alias]
                if candidate.page_number == sample.page_number
            ),
            None,
        )
        if page is None:
            raise OCRBenchmarkPreparationError(
                f"Page {sample.page_number} is unavailable for alias "
                f"{sample.paper_alias}"
            )
        data = sample.model_dump()
        data.update(
            image_path=page.image_path,
            image_width=page.width,
            image_height=page.height,
        )
        samples.append(OCRBenchmarkSample.model_validate(data))
    return BenchmarkManifest(samples=tuple(samples))


if __name__ == "__main__":
    raise SystemExit(main())

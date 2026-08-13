"""Synthetic-only tests for private OCR benchmark preparation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

import pymupdf
import pytest

from app.core.exceptions import OCRBenchmarkPreparationError
from app.domain.models.paper import PaperPage
from app.evaluation.ocr_benchmark.models import (
    BenchmarkDifficulty,
    BenchmarkManifest,
    BenchmarkRegion,
    GroundTruthStatus,
    OCRBenchmarkSample,
)
from app.evaluation.ocr_benchmark.preparation import (
    BenchmarkPreparer,
    safe_sample_filename,
)


def _canonical_page(tmp_path: Path, *, page_number: int = 1) -> PaperPage:
    image_path = (tmp_path / f"canonical_{page_number:04d}.png").resolve()
    document = pymupdf.open()  # type: ignore[no-untyped-call]
    try:
        page = document.new_page(width=100, height=80)
        page.draw_rect(
            pymupdf.Rect(0, 0, 50, 80),  # type: ignore[no-untyped-call]
            color=(1, 0, 0),
            fill=(1, 0, 0),
        )
        page.draw_rect(
            pymupdf.Rect(50, 0, 100, 80),  # type: ignore[no-untyped-call]
            color=(0, 0, 1),
            fill=(0, 0, 1),
        )
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(1, 1),  # type: ignore[no-untyped-call]
            alpha=False,
        )
        pixmap.save(str(image_path))  # type: ignore[no-untyped-call]
    finally:
        document.close()  # type: ignore[no-untyped-call]
    return PaperPage(
        paper_id=uuid4(),
        page_number=page_number,
        image_path=image_path,
        width=100,
        height=80,
    )


def _sample(
    *,
    sample_id: str = "synthetic-sample-01",
    page_number: int = 1,
    region: BenchmarkRegion | None = None,
    declared_width: int = 100,
    declared_height: int = 80,
) -> OCRBenchmarkSample:
    return OCRBenchmarkSample(
        sample_id=sample_id,
        paper_alias="synthetic-paper-a",
        page_number=page_number,
        image_path=Path("private/canonical.png"),
        image_width=declared_width,
        image_height=declared_height,
        region=region,
        difficulty=BenchmarkDifficulty.MEDIUM,
        categories=("synthetic-category",),
        printed_content_present=True,
        teacher_annotations_present=True,
        ground_truth_status=GroundTruthStatus.PENDING,
        ground_truth_student_text=None,
    )


def _preparer(tmp_path: Path) -> BenchmarkPreparer:
    root = tmp_path / "data" / "evaluation" / "ocr"
    return BenchmarkPreparer(
        evaluation_root=root,
        samples_dir=root / "samples",
        worksheet_path=root / "transcription_worksheet.md",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_crop_uses_configured_pixel_coordinates(tmp_path: Path) -> None:
    page = _canonical_page(tmp_path)
    sample = _sample(region=BenchmarkRegion(x=50, y=10, width=40, height=30))

    prepared = _preparer(tmp_path).prepare(
        BenchmarkManifest(samples=(sample,)),
        {"synthetic-paper-a": (page,)},
    )

    output = pymupdf.Pixmap(str(prepared[0].output_path))  # type: ignore[no-untyped-call]
    assert (output.width, output.height) == (40, 30)
    red, green, blue = output.pixel(20, 15)[:3]  # type: ignore[no-untyped-call]
    assert blue > 240
    assert red < 15
    assert green < 15


@pytest.mark.parametrize(
    ("index", "expected"),
    [(1, "sample_001.png"), (8, "sample_008.png"), (125, "sample_125.png")],
)
def test_safe_sample_filenames_are_deterministic(index: int, expected: str) -> None:
    assert safe_sample_filename(index) == expected


def test_preparation_is_idempotent_and_preserves_human_worksheet(
    tmp_path: Path,
) -> None:
    page = _canonical_page(tmp_path)
    manifest = BenchmarkManifest(samples=(_sample(),))
    preparer = _preparer(tmp_path)

    first = preparer.prepare(manifest, {"synthetic-paper-a": (page,)})
    first_hash = _sha256(first[0].output_path)
    worksheet = tmp_path / "data/evaluation/ocr/transcription_worksheet.md"
    human_content = worksheet.read_text(encoding="utf-8") + "manual text\n"
    worksheet.write_text(human_content, encoding="utf-8")

    second = preparer.prepare(manifest, {"synthetic-paper-a": (page,)})

    assert second[0].output_path == first[0].output_path
    assert _sha256(second[0].output_path) == first_hash
    assert worksheet.read_text(encoding="utf-8") == human_content


def test_region_is_checked_against_actual_canonical_dimensions(tmp_path: Path) -> None:
    page = _canonical_page(tmp_path)
    sample = _sample(
        region=BenchmarkRegion(x=80, y=0, width=30, height=20),
        declared_width=200,
        declared_height=200,
    )

    with pytest.raises(OCRBenchmarkPreparationError, match="exceeds"):
        _preparer(tmp_path).prepare(
            BenchmarkManifest(samples=(sample,)),
            {"synthetic-paper-a": (page,)},
        )


@pytest.mark.parametrize("missing_kind", ["alias", "page", "file"])
def test_missing_canonical_source_page_is_rejected(
    tmp_path: Path,
    missing_kind: str,
) -> None:
    page = _canonical_page(tmp_path)
    sample = _sample(page_number=2 if missing_kind == "page" else 1)
    pages = {} if missing_kind == "alias" else {"synthetic-paper-a": (page,)}
    if missing_kind == "file":
        page.image_path.unlink()

    with pytest.raises(OCRBenchmarkPreparationError, match="unavailable"):
        _preparer(tmp_path).prepare(BenchmarkManifest(samples=(sample,)), pages)


def test_canonical_image_is_unchanged(tmp_path: Path) -> None:
    page = _canonical_page(tmp_path)
    original_hash = _sha256(page.image_path)
    sample = _sample(region=BenchmarkRegion(x=10, y=10, width=20, height=20))

    _preparer(tmp_path).prepare(
        BenchmarkManifest(samples=(sample,)),
        {"synthetic-paper-a": (page,)},
    )

    assert _sha256(page.image_path) == original_hash


def test_outputs_must_remain_under_private_evaluation_root(tmp_path: Path) -> None:
    root = tmp_path / "data/evaluation/ocr"

    with pytest.raises(OCRBenchmarkPreparationError, match="private"):
        BenchmarkPreparer(
            evaluation_root=root,
            samples_dir=tmp_path / "public-samples",
            worksheet_path=root / "worksheet.md",
        )


def test_worksheet_contains_blank_ground_truth_field(tmp_path: Path) -> None:
    page = _canonical_page(tmp_path)

    _preparer(tmp_path).prepare(
        BenchmarkManifest(samples=(_sample(),)),
        {"synthetic-paper-a": (page,)},
    )

    worksheet = (tmp_path / "data/evaluation/ocr/transcription_worksheet.md").read_text(
        encoding="utf-8"
    )
    assert "Sample ID: synthetic-sample-01" in worksheet
    assert "Paper alias: synthetic-paper-a" in worksheet
    assert "Ground truth student text:\n\n" in worksheet

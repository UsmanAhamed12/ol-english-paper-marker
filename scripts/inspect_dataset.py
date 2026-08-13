"""Read-only PDF dataset inventory for Phase 0.

The script uses Poppler command-line tools already available on the development
machine. It does not alter source PDFs or create derived dataset artifacts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median


@dataclass(frozen=True)
class ImageRecord:
    """Metadata for one image reported by ``pdfimages -list``."""

    page: int
    width: int
    height: int
    encoding: str
    x_ppi: int
    y_ppi: int


@dataclass(frozen=True)
class PdfRecord:
    """Read-only structural metadata for one PDF."""

    file: str
    size_bytes: int
    pages: int
    encrypted: bool
    extractable_text_characters: int
    images: tuple[ImageRecord, ...]


@dataclass(frozen=True)
class DatasetSummary:
    """Aggregate measurements for the inspected PDF corpus."""

    pdf_count: int
    total_pages: int
    total_bytes: int
    page_count_distribution: dict[int, int]
    minimum_pages: int
    median_pages: float
    maximum_pages: int
    textless_pdfs: int
    encrypted_pdfs: int
    image_count: int
    jpeg_image_count: int
    minimum_image_width: int
    median_image_width: float
    maximum_image_width: int
    minimum_image_height: int
    median_image_height: float
    maximum_image_height: int


def parse_pdfinfo(output: str) -> dict[str, str]:
    """Parse Poppler ``pdfinfo`` key/value output."""

    fields: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip()] = value.strip()
    return fields


def parse_pdfimages(output: str) -> tuple[ImageRecord, ...]:
    """Parse image rows emitted by Poppler ``pdfimages -list``."""

    records: list[ImageRecord] = []
    for line in output.splitlines():
        columns = line.split()
        if len(columns) < 14 or not columns[0].isdigit():
            continue
        records.append(
            ImageRecord(
                page=int(columns[0]),
                width=int(columns[3]),
                height=int(columns[4]),
                encoding=columns[8],
                x_ppi=int(columns[12]),
                y_ppi=int(columns[13]),
            )
        )
    return tuple(records)


def run_command(arguments: Sequence[str]) -> str:
    """Run a read-only inspection command and return its standard output."""

    completed = subprocess.run(
        arguments,
        check=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout


def inspect_pdf(path: Path) -> PdfRecord:
    """Collect structural and text-layer metadata for one PDF."""

    info = parse_pdfinfo(run_command(("pdfinfo", str(path))))
    text = run_command(("pdftotext", str(path), "-"))
    images = parse_pdfimages(run_command(("pdfimages", "-list", str(path))))
    return PdfRecord(
        file=path.name,
        size_bytes=path.stat().st_size,
        pages=int(info["Pages"]),
        encrypted=info.get("Encrypted", "no").lower().startswith("yes"),
        extractable_text_characters=len(text.strip()),
        images=images,
    )


def summarize(records: Sequence[PdfRecord]) -> DatasetSummary:
    """Calculate aggregate measurements from PDF inspection records."""

    if not records:
        raise ValueError("No PDF records were supplied")

    pages = [record.pages for record in records]
    images = [image for record in records for image in record.images]
    if not images:
        raise ValueError("No embedded images were found")

    widths = [image.width for image in images]
    heights = [image.height for image in images]
    return DatasetSummary(
        pdf_count=len(records),
        total_pages=sum(pages),
        total_bytes=sum(record.size_bytes for record in records),
        page_count_distribution=dict(sorted(Counter(pages).items())),
        minimum_pages=min(pages),
        median_pages=float(median(pages)),
        maximum_pages=max(pages),
        textless_pdfs=sum(
            record.extractable_text_characters == 0 for record in records
        ),
        encrypted_pdfs=sum(record.encrypted for record in records),
        image_count=len(images),
        jpeg_image_count=sum(image.encoding == "jpeg" for image in images),
        minimum_image_width=min(widths),
        median_image_width=float(median(widths)),
        maximum_image_width=max(widths),
        minimum_image_height=min(heights),
        median_image_height=float(median(heights)),
        maximum_image_height=max(heights),
    )


def inspect_dataset(dataset_dir: Path) -> tuple[DatasetSummary, list[PdfRecord]]:
    """Inspect every PDF directly under ``dataset_dir``."""

    paths = sorted(dataset_dir.glob("*.pdf"))
    if not paths:
        raise ValueError(f"No PDFs found in {dataset_dir}")
    records = [inspect_pdf(path) for path in paths]
    return summarize(records), records


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        type=Path,
        default=Path("data/raw/marked_papers"),
        help="directory containing the raw PDFs",
    )
    return parser.parse_args()


def main() -> int:
    """Run the inventory and print JSON to standard output."""

    missing = [
        tool for tool in ("pdfinfo", "pdftotext", "pdfimages") if not shutil.which(tool)
    ]
    if missing:
        raise RuntimeError(f"Required Poppler tools not found: {', '.join(missing)}")

    args = parse_args()
    summary, records = inspect_dataset(args.dataset_dir)
    payload = {
        "dataset_dir": str(args.dataset_dir),
        "summary": asdict(summary),
        "pdfs": [asdict(record) for record in records],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

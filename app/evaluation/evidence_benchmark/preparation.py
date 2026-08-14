"""Materialize private crops and a blank, idempotent labeling worksheet."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2

from app.core.exceptions import EvidenceSeparationError
from app.evaluation.evidence_benchmark.models import (
    EvidenceBenchmarkManifest,
    EvidenceBenchmarkSample,
)


def prepare_evidence_benchmark(
    manifest: EvidenceBenchmarkManifest,
    output_root: Path,
) -> tuple[tuple[Path, ...], Path]:
    """Refresh safe crops while preserving any existing human worksheet."""

    root = output_root.resolve()
    samples_root = root / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    outputs = tuple(
        _materialize_sample(sample, samples_root / safe_sample_filename(sample))
        for sample in manifest.samples
    )
    worksheet = root / "labeling_worksheet.md"
    if not worksheet.exists():
        worksheet.write_text(_worksheet(manifest), encoding="utf-8")
    return outputs, worksheet


def safe_sample_filename(sample: EvidenceBenchmarkSample) -> str:
    """Return a deterministic filename containing no source identity."""

    return f"{sample.sample_id}.png"


def _materialize_sample(sample: EvidenceBenchmarkSample, output: Path) -> Path:
    source = sample.source_image_path.expanduser().resolve(strict=True)
    if output.resolve() == source:
        raise EvidenceSeparationError("Evidence sample cannot overwrite its source")
    source_hash = _sha256(source)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3:
        raise EvidenceSeparationError("Evidence benchmark source image is invalid")
    height, width = image.shape[:2]
    if (width, height) != (sample.page_width, sample.page_height):
        raise EvidenceSeparationError("Evidence benchmark page dimensions changed")
    box = sample.region
    crop = image[box.y : box.y + box.height, box.x : box.x + box.width]
    temporary = output.with_name(f".{output.stem}.tmp.png")
    if not cv2.imwrite(str(temporary), crop):
        raise EvidenceSeparationError("Evidence benchmark crop could not be written")
    temporary.replace(output)
    if _sha256(source) != source_hash:
        raise EvidenceSeparationError(
            "Canonical page changed during evidence benchmark preparation"
        )
    return output.resolve()


def _worksheet(manifest: EvidenceBenchmarkManifest) -> str:
    lines = [
        "# Private evidence-separation labeling worksheet",
        "",
        "Do not transcribe student text. Inspect each original crop and the derived",
        "overlay. Select exactly one dominant evidence class: PRINTED,",
        "STUDENT_CANDIDATE, TEACHER_CANDIDATE, or UNKNOWN. Use UNKNOWN whenever",
        "attribution is mixed or uncertain. Record zero or more student-answer boxes",
        "as crop-relative `x,y,width,height`; an explicitly verified empty list means",
        "the crop contains no student-answer region.",
        "",
    ]
    for sample in manifest.samples:
        lines.extend(
            (
                f"## {sample.sample_id}",
                "",
                f"Paper alias: {sample.paper_alias}",
                f"Page: {sample.page_number}",
                f"Test: {sample.test_number:02d}",
                "Selection categories: "
                + ", ".join(item.value for item in sample.categories),
                (
                    "Page region: "
                    f"{sample.region.x},{sample.region.y},"
                    f"{sample.region.width},{sample.region.height}"
                ),
                "Ground-truth evidence class:",
                "Answer regions (crop-relative x,y,width,height; use [] if none):",
                "Human verification state: pending",
                "Notes:",
                "",
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

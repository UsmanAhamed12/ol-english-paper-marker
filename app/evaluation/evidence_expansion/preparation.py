"""Materialize private evidence-v2 crops without creating human labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from app.core.exceptions import EvidenceSeparationError
from app.evaluation.evidence_expansion.models import EvidenceExpansionManifest


def prepare_expansion_artifacts(
    manifest: EvidenceExpansionManifest,
    output_root: Path,
) -> tuple[int, int]:
    """Write deterministic safe crops/overlays and return duplicate counts."""

    root = output_root.resolve()
    samples_root = root / "samples"
    overlays_root = root / "overlays"
    samples_root.mkdir(parents=True, exist_ok=True)
    overlays_root.mkdir(parents=True, exist_ok=True)
    exact_hashes: set[str] = set()
    written = 0
    duplicates = 0
    for sample in manifest.samples:
        source = sample.source_image_path.resolve(strict=True)
        before = _sha256(source)
        if before != sample.source_image_sha256:
            raise EvidenceSeparationError("Evidence-v2 canonical source hash changed")
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (sample.page_height, sample.page_width):
            raise EvidenceSeparationError("Evidence-v2 canonical source is invalid")
        box = sample.region
        crop = image[box.y : box.y + box.height, box.x : box.x + box.width].copy()
        crop_hash = hashlib.sha256(crop.tobytes()).hexdigest()
        if crop_hash in exact_hashes:
            duplicates += 1
            raise EvidenceSeparationError(
                "Evidence-v2 contains an exact duplicate crop"
            )
        exact_hashes.add(crop_hash)
        sample_path = samples_root / f"{sample.sample_id}.png"
        overlay_path = overlays_root / f"{sample.sample_id}.png"
        _write_image_atomic(sample_path, crop)
        overlay = crop.copy()
        cv2.rectangle(
            overlay,
            (1, 1),
            (max(1, crop.shape[1] - 2), max(1, crop.shape[0] - 2)),
            (0, 165, 255),
            max(2, crop.shape[1] // 500),
        )
        cv2.putText(
            overlay,
            "CANDIDATE - HUMAN LABEL REQUIRED",
            (12, min(38, max(20, crop.shape[0] - 8))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 90, 220),
            2,
            cv2.LINE_AA,
        )
        _write_image_atomic(overlay_path, overlay)
        if _sha256(source) != before:
            raise EvidenceSeparationError(
                "Canonical image changed during evidence-v2 preparation"
            )
        written += 1
    return written, duplicates


def write_json_atomic(path: Path, payload: object) -> None:
    """Write private metadata atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError as error:
        raise EvidenceSeparationError(
            "Private evidence-v2 metadata could not be written"
        ) from error


def _write_image_atomic(path: Path, image: np.ndarray) -> None:
    temporary = path.with_name(f".{path.stem}.tmp.png")
    try:
        if not cv2.imwrite(str(temporary), image):
            raise EvidenceSeparationError("Evidence-v2 image could not be written")
        temporary.replace(path)
    except cv2.error as error:
        raise EvidenceSeparationError(
            "Evidence-v2 image could not be written"
        ) from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

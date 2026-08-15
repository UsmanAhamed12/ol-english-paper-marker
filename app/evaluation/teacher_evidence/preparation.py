"""Private crop materialization for the teacher-focused candidate set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from app.core.exceptions import EvidenceSeparationError
from app.evaluation.teacher_evidence.models import TeacherEvidenceManifest


def prepare_teacher_artifacts(
    manifest: TeacherEvidenceManifest, output_root: Path
) -> tuple[int, int]:
    """Write neutral derived crops and overlays while preserving canonical bytes."""

    root = output_root.resolve()
    samples_root = root / "samples"
    overlays_root = root / "overlays"
    samples_root.mkdir(parents=True, exist_ok=True)
    overlays_root.mkdir(parents=True, exist_ok=True)
    crop_hashes: set[str] = set()
    written = 0
    duplicates = 0
    for sample in manifest.samples:
        source = sample.source_image_path.resolve(strict=True)
        before = sha256_file(source)
        if before != sample.source_image_sha256:
            raise EvidenceSeparationError("Teacher candidate canonical hash changed")
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (sample.page_height, sample.page_width):
            raise EvidenceSeparationError(
                "Teacher candidate canonical image is invalid"
            )
        box = sample.region
        crop = image[box.y : box.y + box.height, box.x : box.x + box.width].copy()
        crop_hash = hashlib.sha256(crop.tobytes()).hexdigest()
        if crop_hash in crop_hashes:
            duplicates += 1
            raise EvidenceSeparationError("Teacher candidate contains a duplicate crop")
        crop_hashes.add(crop_hash)
        write_image_atomic(samples_root / f"{sample.sample_id}.png", crop)
        overlay = crop.copy()
        component = sample.candidate_component
        left = component.x - box.x
        top = component.y - box.y
        right = left + component.width
        bottom = top + component.height
        cv2.rectangle(overlay, (left, top), (right, bottom), (0, 165, 255), 3)
        cv2.putText(
            overlay,
            "CANDIDATE REGION - HUMAN REVIEW",
            (12, min(36, max(20, crop.shape[0] - 8))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (0, 90, 220),
            2,
            cv2.LINE_AA,
        )
        write_image_atomic(overlays_root / f"{sample.sample_id}.png", overlay)
        if sha256_file(source) != before:
            raise EvidenceSeparationError("Canonical image changed during preparation")
        written += 1
    return written, duplicates


def write_json_atomic(path: Path, payload: object) -> None:
    """Write private metadata atomically without exposing it to logs."""

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
            "Private teacher metadata write failed"
        ) from error


def write_image_atomic(path: Path, image: np.ndarray) -> None:
    """Atomically write one private derived PNG."""

    temporary = path.with_name(f".{path.stem}.tmp.png")
    try:
        if not cv2.imwrite(str(temporary), image):
            raise EvidenceSeparationError("Private teacher image write failed")
        temporary.replace(path)
    except cv2.error as error:
        raise EvidenceSeparationError("Private teacher image write failed") from error


def sha256_file(path: Path) -> str:
    """Hash a file in bounded blocks."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

"""Private overlays distinguishing V2 evidence, answer spaces, and answers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from app.core.exceptions import EvidenceSeparationError
from app.domain.models.paper import PaperPage
from app.evidence.models import EvidenceType
from app.evidence.v2_models import TestEvidenceV2
from app.ocr.models import BoundingBox

_COLORS = {
    EvidenceType.PRINTED: (0, 150, 0),
    EvidenceType.STUDENT_CANDIDATE: (210, 90, 0),
    EvidenceType.TEACHER_CANDIDATE: (0, 0, 220),
    EvidenceType.UNKNOWN: (120, 120, 120),
}


def render_evidence_overlay_v2(
    page: PaperPage,
    evidence: TestEvidenceV2,
    output_path: Path,
    *,
    crop_bbox: BoundingBox | None = None,
) -> Path:
    """Write a derived, geometry-preserving overlay without source mutation."""

    source = page.image_path.resolve()
    output = output_path.resolve()
    if source == output:
        raise EvidenceSeparationError("V2 overlay cannot overwrite its source")
    source_hash = _sha256(source)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise EvidenceSeparationError("V2 overlay source is invalid")
    for space in evidence.answer_spaces:
        _box(image, space.bbox, (180, 150, 0), "ANSWER-SPACE", 2)
    for region in evidence.evidence_regions:
        _box(
            image,
            region.bbox,
            _COLORS[region.evidence_type],
            region.evidence_type.value.upper(),
            2,
        )
    for answer in evidence.student_answer_regions:
        _box(image, answer.bbox, (220, 180, 0), "STUDENT-ANSWER", 3)
    if crop_bbox is not None:
        image = image[
            crop_bbox.y : crop_bbox.y + crop_bbox.height,
            crop_bbox.x : crop_bbox.x + crop_bbox.width,
        ]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.png")
    if not cv2.imwrite(str(temporary), image):
        raise EvidenceSeparationError("V2 overlay could not be written")
    temporary.replace(output)
    if _sha256(source) != source_hash:
        raise EvidenceSeparationError("Canonical page changed during V2 overlay")
    return output


def _box(
    image: np.ndarray,
    box: BoundingBox,
    color: tuple[int, int, int],
    label: str,
    thickness: int,
) -> None:
    cv2.rectangle(
        image,
        (box.x, box.y),
        (box.x + box.width, box.y + box.height),
        color,
        thickness,
    )
    cv2.putText(
        image,
        label,
        (box.x, max(18, box.y - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        color,
        1,
        cv2.LINE_AA,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

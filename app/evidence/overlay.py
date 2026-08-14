"""Private visual-debug overlays for classified evidence and answer regions."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2

from app.core.exceptions import EvidenceSeparationError
from app.domain.models.paper import PaperPage
from app.evidence.models import EvidenceType, TestEvidence
from app.ocr.models import BoundingBox

_EVIDENCE_COLORS = {
    EvidenceType.PRINTED: (0, 150, 0),
    EvidenceType.STUDENT_CANDIDATE: (210, 90, 0),
    EvidenceType.TEACHER_CANDIDATE: (0, 0, 220),
    EvidenceType.UNKNOWN: (120, 120, 120),
}
_SAFE_LABELS = {
    EvidenceType.PRINTED: "PRINTED",
    EvidenceType.STUDENT_CANDIDATE: "STUDENT",
    EvidenceType.TEACHER_CANDIDATE: "TEACHER",
    EvidenceType.UNKNOWN: "UNKNOWN",
}


def render_evidence_overlay(
    page: PaperPage,
    evidence: TestEvidence,
    output_path: Path,
    *,
    crop_bbox: BoundingBox | None = None,
) -> Path:
    """Render safe class and answer boxes on a derived copy or crop."""

    source = page.image_path.resolve()
    output = output_path.resolve()
    if source == output:
        raise EvidenceSeparationError("Evidence overlay cannot overwrite its source")
    if evidence.paper_id != page.paper_id or evidence.page_number != page.page_number:
        raise EvidenceSeparationError("Evidence overlay provenance does not match page")
    if crop_bbox is not None and not _contains(
        BoundingBox(x=0, y=0, width=page.width, height=page.height), crop_bbox
    ):
        raise EvidenceSeparationError("Evidence overlay crop exceeds source page")
    source_hash = _sha256(source)
    try:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise EvidenceSeparationError("Evidence overlay source is invalid")
        for region in evidence.evidence_regions:
            box = region.bbox
            color = _EVIDENCE_COLORS[region.evidence_type]
            cv2.rectangle(
                image,
                (box.x, box.y),
                (box.x + box.width, box.y + box.height),
                color,
                2,
            )
            cv2.putText(
                image,
                _SAFE_LABELS[region.evidence_type],
                (box.x, max(18, box.y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
        for answer in evidence.answer_regions:
            box = answer.bbox
            cv2.rectangle(
                image,
                (box.x, box.y),
                (box.x + box.width, box.y + box.height),
                (220, 180, 0),
                3,
            )
            cv2.putText(
                image,
                "ANSWER-CANDIDATE",
                (box.x, max(22, box.y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 180, 0),
                2,
                cv2.LINE_AA,
            )
        if crop_bbox is not None:
            image = image[
                crop_bbox.y : crop_bbox.y + crop_bbox.height,
                crop_bbox.x : crop_bbox.x + crop_bbox.width,
            ]
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.tmp.png")
        if not cv2.imwrite(str(temporary), image):
            raise EvidenceSeparationError("Evidence overlay could not be written")
        temporary.replace(output)
        if _sha256(source) != source_hash:
            raise EvidenceSeparationError(
                "Canonical page changed during evidence overlay rendering"
            )
    except EvidenceSeparationError:
        raise
    except (OSError, cv2.error, ValueError) as error:
        raise EvidenceSeparationError(
            "Evidence overlay could not be rendered"
        ) from error
    return output


def _contains(container: BoundingBox, child: BoundingBox) -> bool:
    return (
        child.x >= container.x
        and child.y >= container.y
        and child.x + child.width <= container.x + container.width
        and child.y + child.height <= container.y + container.height
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

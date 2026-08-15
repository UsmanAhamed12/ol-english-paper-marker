"""Private overlays comparing human and predicted evidence geometry."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from app.core.exceptions import EvidenceSeparationError
from app.evaluation.evidence_benchmark.annotations import EvidenceAnnotation
from app.evaluation.evidence_benchmark.models import EvidenceBenchmarkSample
from app.evaluation.evidence_expansion.models import EvidenceExpansionSample
from app.evidence.models import EvidenceType, TestEvidence
from app.ocr.models import BoundingBox

_COLORS = {
    EvidenceType.PRINTED: (0, 150, 0),
    EvidenceType.STUDENT_CANDIDATE: (210, 90, 0),
    EvidenceType.TEACHER_CANDIDATE: (0, 0, 220),
    EvidenceType.UNKNOWN: (120, 120, 120),
}


def render_evaluation_overlay(
    sample: EvidenceBenchmarkSample | EvidenceExpansionSample,
    annotation: EvidenceAnnotation,
    evidence: TestEvidence,
    sample_image_path: Path,
    output_path: Path,
) -> Path:
    """Render classified evidence plus predicted and human answer boxes."""

    source = sample_image_path.resolve(strict=True)
    output = output_path.resolve()
    if source == output:
        raise EvidenceSeparationError("Evaluation overlay cannot replace its source")
    source_hash = _sha256(source)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (sample.region.height, sample.region.width):
        raise EvidenceSeparationError("Private evidence sample image is invalid")
    try:
        for evidence_region in evidence.evidence_regions:
            box = _relative(evidence_region.bbox, sample.region)
            color = _COLORS[evidence_region.evidence_type]
            _rectangle(image, box, color, 1)
        for predicted_answer in evidence.answer_regions:
            _rectangle(
                image,
                _relative(predicted_answer.bbox, sample.region),
                (220, 180, 0),
                3,
            )
        for human_answer in annotation.answer_regions:
            _rectangle(image, human_answer.bbox, (30, 220, 30), 4)
        cv2.putText(
            image,
            "HUMAN ANSWER=GREEN  PREDICTED ANSWER=CYAN",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.tmp.png")
        if not cv2.imwrite(str(temporary), image):
            raise EvidenceSeparationError("Evaluation overlay could not be written")
        temporary.replace(output)
    except cv2.error as error:
        raise EvidenceSeparationError(
            "Evaluation overlay could not be rendered"
        ) from error
    if _sha256(source) != source_hash:
        raise EvidenceSeparationError(
            "Private sample changed during evaluation overlay rendering"
        )
    return output


def _relative(box: BoundingBox, crop: BoundingBox) -> BoundingBox:
    result = BoundingBox(
        x=box.x - crop.x,
        y=box.y - crop.y,
        width=box.width,
        height=box.height,
    )
    if (
        result.x < 0
        or result.y < 0
        or result.x + result.width > crop.width
        or result.y + result.height > crop.height
    ):
        raise EvidenceSeparationError("Overlay evidence exceeds benchmark crop")
    return result


def _rectangle(
    image: np.ndarray,
    box: BoundingBox,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.rectangle(
        image,
        (box.x, box.y),
        (box.x + box.width, box.y + box.height),
        color,
        thickness,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

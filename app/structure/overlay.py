"""Private visual-debug overlays for detected markers and Test regions."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2

from app.core.exceptions import StructureDetectionError
from app.domain.models.paper import PaperPage
from app.structure.models import ExamPageStructure


def render_structure_overlay(
    page: PaperPage,
    structure: ExamPageStructure,
    output_path: Path,
) -> Path:
    """Draw safe normalized labels on a copy without modifying canonical input."""

    source = page.image_path.resolve()
    output = output_path.resolve()
    if source == output:
        raise StructureDetectionError("Structure overlay cannot overwrite its source")
    if structure.page_number != page.page_number:
        raise StructureDetectionError("Structure overlay page does not match source")
    if not source.is_file():
        raise StructureDetectionError("Structure overlay source is unavailable")

    source_hash = _sha256(source)
    try:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise StructureDetectionError("Structure overlay source is invalid")
        for region in structure.regions:
            box = region.bbox
            cv2.rectangle(
                image,
                (box.x, box.y),
                (box.x + box.width - 1, box.y + box.height - 1),
                (255, 120, 0),
                3,
            )
        for marker in structure.markers:
            box = marker.bbox
            cv2.rectangle(
                image,
                (box.x, box.y),
                (box.x + box.width, box.y + box.height),
                (0, 180, 0),
                4,
            )
            cv2.putText(
                image,
                marker.label,
                (box.x, max(30, box.y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 120, 0),
                3,
                cv2.LINE_AA,
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.tmp.png")
        if not cv2.imwrite(str(temporary), image):
            raise StructureDetectionError("Structure overlay could not be written")
        temporary.replace(output)
        if _sha256(source) != source_hash:
            raise StructureDetectionError("Canonical page changed during overlay")
    except StructureDetectionError:
        raise
    except (OSError, cv2.error, ValueError) as error:
        raise StructureDetectionError(
            "Structure overlay could not be rendered"
        ) from error
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

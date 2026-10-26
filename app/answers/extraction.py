"""Immutable canonical-image cropping boundary for answer OCR."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2

from app.answers.models import AnswerSourceRegion
from app.core.exceptions import AnswerExtractionError
from app.domain.models.paper import PaperPage
from app.evidence.models import StudentAnswerRegion


class StudentRegionCropper:
    """Materialize deterministic derived PNGs without changing canonical pages."""

    def __init__(self, runtime_root: Path) -> None:
        self._runtime_root = runtime_root.expanduser().resolve()

    def materialize(
        self,
        page: PaperPage,
        region: StudentAnswerRegion,
        *,
        region_index: int,
    ) -> tuple[AnswerSourceRegion, PaperPage]:
        """Crop one safe page-local region and retain source integrity evidence."""

        if region.paper_id != page.paper_id or region.page_number != page.page_number:
            raise AnswerExtractionError("Answer region does not match its source page")
        if not _fits(page, region):
            raise AnswerExtractionError("Answer region lies outside its source page")
        source_hash = _sha256(page.image_path)
        image = cv2.imread(str(page.image_path), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3:
            raise AnswerExtractionError("Canonical answer source image is invalid")
        box = region.bbox
        crop = image[box.y : box.y + box.height, box.x : box.x + box.width]
        if crop.size == 0:
            raise AnswerExtractionError("Answer region produced an empty crop")
        target = (
            self._runtime_root
            / page.paper_id.hex
            / "answers"
            / f"test_{region.test_number:02d}"
            / f"page_{page.page_number:04d}_region_{region_index:04d}.png"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            encoded, payload = cv2.imencode(".png", crop)
        except cv2.error as error:
            raise AnswerExtractionError(
                "Answer-region image encoding failed"
            ) from error
        if not encoded:
            raise AnswerExtractionError("Answer-region image encoding failed")
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_bytes(payload.tobytes())
        temporary.replace(target)
        if _sha256(page.image_path) != source_hash:
            raise AnswerExtractionError("Canonical page changed during answer cropping")
        source = AnswerSourceRegion(
            region_index=region_index,
            paper_id=region.paper_id,
            test_number=region.test_number,
            page_number=region.page_number,
            bbox=region.bbox,
            evidence_confidence=region.confidence,
            source_evidence_indices=region.source_evidence_indices,
            canonical_image_path=page.image_path,
            canonical_image_sha256=source_hash,
            derived_image_path=target,
            detection_strategy=region.detection_strategy,
        )
        crop_page = PaperPage(
            paper_id=page.paper_id,
            page_number=page.page_number,
            image_path=target,
            width=box.width,
            height=box.height,
        )
        return source, crop_page


def _fits(page: PaperPage, region: StudentAnswerRegion) -> bool:
    box = region.bbox
    return box.x + box.width <= page.width and box.y + box.height <= page.height


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

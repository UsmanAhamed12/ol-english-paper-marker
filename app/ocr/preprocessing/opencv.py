"""Conservative OpenCV preprocessing that preserves source geometry."""

from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from app.core.exceptions import ImagePreprocessingError
from app.ocr.preprocessing.models import PreprocessingResult, PreprocessingVariant

_HASH_BLOCK_SIZE = 1024 * 1024


class OpenCVPreprocessor:
    """Apply one fixed preprocessing variant to a separate PNG artifact."""

    def __init__(self, variant: PreprocessingVariant) -> None:
        if variant is PreprocessingVariant.NONE:
            raise ValueError("baseline does not require an image preprocessor")
        self._variant = variant

    @property
    def name(self) -> str:
        """Return the stable variant identifier."""

        return self._variant.value

    def process(self, source_path: Path, processed_path: Path) -> PreprocessingResult:
        """Create a deterministic derived PNG while proving source immutability."""

        source = source_path.resolve()
        output = processed_path.resolve()
        if source == output:
            raise ImagePreprocessingError("Preprocessing cannot overwrite its source")
        if not source.is_file():
            raise ImagePreprocessingError("Preprocessing source image is unavailable")
        if output.suffix.casefold() != ".png":
            raise ImagePreprocessingError("Preprocessing output must be a PNG")

        started = perf_counter()
        try:
            source_hash = _sha256(source)
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None or image.ndim != 3:
                raise ImagePreprocessingError("Preprocessing source image is invalid")
            source_height, source_width = image.shape[:2]
            processed = self._apply(image)
            processed_height, processed_width = processed.shape[:2]
            if (processed_width, processed_height) != (source_width, source_height):
                raise ImagePreprocessingError(
                    "Preprocessing unexpectedly changed image dimensions"
                )

            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.stem}.tmp.png")
            if not cv2.imwrite(str(temporary), processed):
                raise ImagePreprocessingError("Derived OCR image could not be written")
            temporary.replace(output)
            if _sha256(source) != source_hash:
                raise ImagePreprocessingError(
                    "Canonical OCR image changed unexpectedly"
                )
        except ImagePreprocessingError:
            raise
        except (OSError, cv2.error, ValueError) as error:
            raise ImagePreprocessingError(
                "Derived OCR image could not be produced"
            ) from error

        return PreprocessingResult(
            source_image_path=source,
            processed_image_path=output,
            source_width=source_width,
            source_height=source_height,
            processed_width=processed_width,
            processed_height=processed_height,
            operations=self._variant.operations,
            processing_duration_ms=(perf_counter() - started) * 1000,
        )

    def _apply(self, image: np.ndarray) -> np.ndarray:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self._variant in {
            PreprocessingVariant.GRAYSCALE_DENOISE,
            PreprocessingVariant.GRAYSCALE_DENOISE_THRESHOLD,
        }:
            grayscale = cv2.GaussianBlur(grayscale, (3, 3), 0)
        if self._variant in {
            PreprocessingVariant.GRAYSCALE_THRESHOLD,
            PreprocessingVariant.GRAYSCALE_DENOISE_THRESHOLD,
        }:
            _, grayscale = cv2.threshold(
                grayscale,
                0,
                255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU,
            )
        return grayscale


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_HASH_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()

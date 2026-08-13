"""Synthetic tests for geometry-preserving OpenCV preprocessing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast
from uuid import uuid4

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from app.core.exceptions import ImagePreprocessingError
from app.domain.models.paper import PaperPage
from app.ocr.base import OCRProvider
from app.ocr.models import OCRExtraction
from app.ocr.preprocessing.base import ImagePreprocessor
from app.ocr.preprocessing.models import (
    PreprocessingOperation,
    PreprocessingResult,
    PreprocessingVariant,
)
from app.ocr.preprocessing.opencv import OpenCVPreprocessor
from app.ocr.preprocessing.provider import PreprocessedOCRProvider


class RecordingProvider:
    name = "synthetic-provider"
    model_version = "synthetic-v1"

    def __init__(self) -> None:
        self.pages: list[PaperPage] = []

    def extract_page(self, page: PaperPage) -> OCRExtraction:
        self.pages.append(page)
        return OCRExtraction(raw_text="synthetic", processing_duration_ms=1)


def _image(path: Path) -> Path:
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    cv2.line(image, (5, 20), (110, 20), (10, 10, 10), 2)
    cv2.putText(
        image,
        "Test",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (90, 30, 180),
        2,
    )
    assert cv2.imwrite(str(path), image)
    return path.resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result(tmp_path: Path) -> PreprocessingResult:
    source = _image(tmp_path / "source.png")
    output = (tmp_path / "processed.png").resolve()
    return PreprocessingResult(
        source_image_path=source,
        processed_image_path=output,
        source_width=120,
        source_height=80,
        processed_width=120,
        processed_height=80,
        operations=(PreprocessingOperation.GRAYSCALE,),
        processing_duration_ms=1,
    )


def test_preprocessing_result_validates_paths_dimensions_and_operations(
    tmp_path: Path,
) -> None:
    result = _result(tmp_path)
    assert result.source_width == result.processed_width
    assert result.operations == (PreprocessingOperation.GRAYSCALE,)

    with pytest.raises(ValidationError, match="dimensions"):
        result.model_copy(update={"processed_width": 119}, deep=True).__class__(
            **{**result.model_dump(), "processed_width": 119}
        )
    with pytest.raises(ValidationError, match="differ"):
        PreprocessingResult(
            **{
                **result.model_dump(),
                "processed_image_path": result.source_image_path,
            }
        )


@pytest.mark.parametrize(
    "variant",
    [
        PreprocessingVariant.GRAYSCALE,
        PreprocessingVariant.GRAYSCALE_DENOISE,
        PreprocessingVariant.GRAYSCALE_THRESHOLD,
        PreprocessingVariant.GRAYSCALE_DENOISE_THRESHOLD,
    ],
)
def test_variants_are_deterministic_preserve_source_and_geometry(
    tmp_path: Path,
    variant: PreprocessingVariant,
) -> None:
    source = _image(tmp_path / "source.png")
    source_hash = _sha256(source)
    first = (tmp_path / "first.png").resolve()
    second = (tmp_path / "second.png").resolve()
    preprocessor = OpenCVPreprocessor(variant)

    first_result = preprocessor.process(source, first)
    second_result = preprocessor.process(source, second)

    assert _sha256(source) == source_hash
    assert _sha256(first) == _sha256(second)
    assert first != source
    assert first_result.source_width == first_result.processed_width == 120
    assert first_result.source_height == first_result.processed_height == 80
    assert second_result.operations == variant.operations


def test_grayscale_denoise_and_threshold_outputs(tmp_path: Path) -> None:
    source = _image(tmp_path / "source.png")
    grayscale_path = (tmp_path / "grayscale.png").resolve()
    denoise_path = (tmp_path / "denoise.png").resolve()
    threshold_path = (tmp_path / "threshold.png").resolve()

    OpenCVPreprocessor(PreprocessingVariant.GRAYSCALE).process(source, grayscale_path)
    OpenCVPreprocessor(PreprocessingVariant.GRAYSCALE_DENOISE).process(
        source, denoise_path
    )
    OpenCVPreprocessor(PreprocessingVariant.GRAYSCALE_THRESHOLD).process(
        source, threshold_path
    )

    grayscale = cv2.imread(str(grayscale_path), cv2.IMREAD_UNCHANGED)
    denoise = cv2.imread(str(denoise_path), cv2.IMREAD_UNCHANGED)
    threshold = cv2.imread(str(threshold_path), cv2.IMREAD_UNCHANGED)
    assert grayscale is not None and grayscale.ndim == 2
    assert denoise is not None and not np.array_equal(denoise, grayscale)
    assert threshold is not None
    assert set(np.unique(threshold)).issubset({0, 255})


def test_invalid_source_and_in_place_output_are_rejected(tmp_path: Path) -> None:
    missing = (tmp_path / "missing.png").resolve()
    preprocessor = OpenCVPreprocessor(PreprocessingVariant.GRAYSCALE)
    with pytest.raises(ImagePreprocessingError, match="unavailable"):
        preprocessor.process(missing, (tmp_path / "out.png").resolve())

    source = _image(tmp_path / "source.png")
    with pytest.raises(ImagePreprocessingError, match="overwrite"):
        preprocessor.process(source, source)


def test_invalid_image_and_write_failure_are_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = (tmp_path / "invalid.png").resolve()
    invalid.write_text("not an image", encoding="utf-8")
    preprocessor = OpenCVPreprocessor(PreprocessingVariant.GRAYSCALE)
    with pytest.raises(ImagePreprocessingError, match="invalid"):
        preprocessor.process(invalid, (tmp_path / "out.png").resolve())

    source = _image(tmp_path / "source.png")
    monkeypatch.setattr("cv2.imwrite", lambda *_args, **_kwargs: False)
    with pytest.raises(ImagePreprocessingError, match="written"):
        preprocessor.process(source, (tmp_path / "failed.png").resolve())


def test_preprocessed_provider_uses_derived_page_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    source = _image(tmp_path / "source.png")
    page = PaperPage(
        paper_id=uuid4(),
        page_number=2,
        image_path=source,
        width=120,
        height=80,
    )
    provider = RecordingProvider()
    wrapper = PreprocessedOCRProvider(
        provider=cast(OCRProvider, provider),
        preprocessor=cast(
            ImagePreprocessor,
            OpenCVPreprocessor(PreprocessingVariant.GRAYSCALE),
        ),
        output_root=(tmp_path / "private-derived").resolve(),
    )

    extraction = wrapper.extract_page(page)

    assert isinstance(wrapper, OCRProvider)
    assert extraction.preprocessing is not None
    assert provider.pages[0].image_path != source
    assert provider.pages[0].image_path.name == "page_0002.png"
    assert provider.pages[0].width == page.width
    assert _sha256(source) == _sha256(extraction.preprocessing.source_image_path)

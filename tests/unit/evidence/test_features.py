"""Synthetic raster and OCR geometry feature tests."""

import cv2
import numpy as np
import pytest

from app.core.exceptions import EvidenceSeparationError
from app.evidence.features import (
    detect_chromatic_components,
    extract_geometry_features,
    extract_ink_features,
)
from app.ocr.models import BoundingBox, OCRWord


def test_red_and_blue_ink_features_are_measured_independently() -> None:
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    cv2.line(image, (10, 40), (80, 40), (0, 0, 180), 5)
    cv2.line(image, (110, 60), (190, 60), (180, 80, 0), 5)

    red = extract_ink_features(image, BoundingBox(x=0, y=0, width=100, height=100))
    blue = extract_ink_features(image, BoundingBox(x=100, y=0, width=100, height=100))

    assert red.red_foreground_ratio > 0.5
    assert blue.blue_foreground_ratio > 0.5


def test_chromatic_components_preserve_page_coordinates() -> None:
    image = np.full((200, 300, 3), 255, dtype=np.uint8)
    cv2.line(image, (120, 80), (150, 110), (0, 0, 200), 5)
    cv2.line(image, (150, 80), (120, 110), (0, 0, 200), 5)

    boxes = detect_chromatic_components(
        image, BoundingBox(x=100, y=50, width=100, height=100)
    )

    assert boxes
    assert boxes[0].x >= 100
    assert boxes[0].y >= 50


def test_irregular_line_geometry_is_measured() -> None:
    words = (
        OCRWord(
            text="one",
            confidence=0.9,
            bbox=BoundingBox(x=10, y=20, width=40, height=20),
            block_number=1,
            paragraph_number=1,
            line_number=1,
            word_number=1,
        ),
        OCRWord(
            text="two",
            confidence=0.9,
            bbox=BoundingBox(x=80, y=5, width=35, height=45),
            block_number=1,
            paragraph_number=1,
            line_number=1,
            word_number=2,
        ),
        OCRWord(
            text="three",
            confidence=0.9,
            bbox=BoundingBox(x=200, y=25, width=50, height=18),
            block_number=1,
            paragraph_number=1,
            line_number=1,
            word_number=3,
        ),
    )

    measured = extract_geometry_features(words, 1, page_width=300, component_count=8)

    assert measured.height_irregularity > 0.5
    assert measured.regularity < 0.7


def test_invalid_feature_crop_is_rejected() -> None:
    image = np.full((50, 50, 3), 255, dtype=np.uint8)

    with pytest.raises(EvidenceSeparationError, match="exceeds"):
        extract_ink_features(image, BoundingBox(x=40, y=40, width=20, height=20))

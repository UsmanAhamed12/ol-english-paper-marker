"""Deterministic raster and OCR-geometry feature extraction."""

from __future__ import annotations

from statistics import fmean, median, pstdev

import cv2
import numpy as np

from app.core.exceptions import EvidenceSeparationError
from app.evidence.models import GeometryFeatures, InkFeatures
from app.ocr.models import BoundingBox, OCRWord


def extract_ink_features(image: np.ndarray, bbox: BoundingBox) -> InkFeatures:
    """Measure local ink without assigning authorship from any one feature."""

    crop = _crop(image, bbox)
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        foreground = (gray < 235) | (saturation > 40)
        foreground_count = int(np.count_nonzero(foreground))
        total = int(gray.size)
        if foreground_count:
            foreground_saturation = saturation[foreground].astype(np.float64)
            mean_saturation = float(np.mean(foreground_saturation)) / 255
            saturation_std = float(np.std(foreground_saturation)) / 127.5
            red = foreground & (saturation >= 55) & ((hue <= 12) | (hue >= 168))
            blue = foreground & (saturation >= 45) & (hue >= 88) & (hue <= 142)
            red_ratio = int(np.count_nonzero(red)) / foreground_count
            blue_ratio = int(np.count_nonzero(blue)) / foreground_count
            dark_ratio = int(np.count_nonzero(foreground & (value < 150))) / total
        else:
            mean_saturation = saturation_std = 0.0
            red_ratio = blue_ratio = dark_ratio = 0.0

        binary = np.where(foreground, 255, 0).astype(np.uint8)
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        meaningful_components = sum(
            int(stats[index, cv2.CC_STAT_AREA]) >= 3
            for index in range(1, component_count)
        )
        edges = cv2.Canny(gray, 60, 160)
        return InkFeatures(
            mean_saturation=_unit(mean_saturation),
            saturation_std=_unit(saturation_std),
            foreground_ratio=_unit(foreground_count / total),
            red_foreground_ratio=_unit(red_ratio),
            blue_foreground_ratio=_unit(blue_ratio),
            dark_foreground_ratio=_unit(dark_ratio),
            local_contrast=_unit(float(np.std(gray)) / 127.5),
            edge_density=_unit(int(np.count_nonzero(edges)) / total),
            connected_component_count=meaningful_components,
        )
    except cv2.error as error:
        raise EvidenceSeparationError(
            "Local image features could not be measured"
        ) from error


def extract_geometry_features(
    words: tuple[OCRWord, ...],
    word_index: int,
    *,
    page_width: int,
    component_count: int,
) -> GeometryFeatures:
    """Measure line regularity around one OCR word using hierarchy first."""

    if word_index < 0 or word_index >= len(words):
        raise ValueError("OCR word index is outside the evidence sequence")
    word = words[word_index]
    line = _line_words(words, word)
    ordered = tuple(sorted(line, key=lambda item: (item.bbox.x, item.bbox.y)))
    heights = [item.bbox.height for item in ordered]
    bottoms = [item.bbox.y + item.bbox.height for item in ordered]
    median_height = max(1.0, float(median(heights)))
    height_irregularity = _normalized_deviation(heights, median_height)
    baseline_irregularity = _normalized_deviation(bottoms, median_height)
    gaps = [
        max(0, following.bbox.x - (previous.bbox.x + previous.bbox.width))
        for previous, following in zip(ordered, ordered[1:], strict=False)
    ]
    spacing_irregularity = (
        0.0 if len(gaps) < 2 or fmean(gaps) == 0 else _unit(pstdev(gaps) / fmean(gaps))
    )
    left = min(item.bbox.x for item in ordered)
    top = min(item.bbox.y for item in ordered)
    right = max(item.bbox.x + item.bbox.width for item in ordered)
    bottom = max(item.bbox.y + item.bbox.height for item in ordered)
    line_area = max(1, (right - left) * (bottom - top))
    word_area = sum(item.bbox.width * item.bbox.height for item in ordered)
    regularity = _unit(
        1 - fmean((height_irregularity, baseline_irregularity, spacing_irregularity))
    )
    fragmentation = _unit(component_count / max(2, len(word.text) * 1.5))
    isolation = 1.0 if len(ordered) == 1 else _unit(1 - (len(ordered) - 1) / 5)
    margin_position = float(
        word.bbox.x <= page_width * 0.08
        or word.bbox.x + word.bbox.width >= page_width * 0.92
    )
    return GeometryFeatures(
        regularity=regularity,
        baseline_irregularity=baseline_irregularity,
        height_irregularity=height_irregularity,
        spacing_irregularity=spacing_irregularity,
        line_density=_unit(word_area / line_area),
        fragmentation=fragmentation,
        isolation=isolation,
        margin_position=margin_position,
        word_count_in_line=len(ordered),
    )


def component_geometry_features(
    bbox: BoundingBox,
    *,
    page_width: int,
    ink: InkFeatures,
) -> GeometryFeatures:
    """Represent a visual-only component without pretending OCR hierarchy exists."""

    margin = float(
        bbox.x <= page_width * 0.08 or bbox.x + bbox.width >= page_width * 0.92
    )
    fragmentation = _unit(ink.connected_component_count / 6)
    return GeometryFeatures(
        regularity=0.0,
        baseline_irregularity=0.5,
        height_irregularity=0.5,
        spacing_irregularity=0.5,
        line_density=ink.foreground_ratio,
        fragmentation=fragmentation,
        isolation=1.0,
        margin_position=margin,
        word_count_in_line=0,
    )


def detect_chromatic_components(
    image: np.ndarray,
    region: BoundingBox,
) -> tuple[BoundingBox, ...]:
    """Locate colored ink components for later multi-signal classification."""

    crop = _crop(image, region)
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = np.where((hsv[:, :, 1] >= 55) & (hsv[:, :, 2] <= 250), 255, 0).astype(
            np.uint8
        )
        joined = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)),
            iterations=1,
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(joined)
    except cv2.error as error:
        raise EvidenceSeparationError(
            "Chromatic evidence components could not be measured"
        ) from error

    maximum_area = region.width * region.height * 0.2
    boxes = []
    for index in range(1, count):
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 12 or width < 3 or height < 3 or area > maximum_area:
            continue
        boxes.append(
            BoundingBox(
                x=region.x + x,
                y=region.y + y,
                width=width,
                height=height,
            )
        )
    return tuple(sorted(boxes, key=lambda box: (box.y, box.x, box.width, box.height)))


def intersection_ratio(left: BoundingBox, right: BoundingBox) -> float:
    """Return intersection area divided by the smaller rectangle area."""

    width = max(
        0,
        min(left.x + left.width, right.x + right.width) - max(left.x, right.x),
    )
    height = max(
        0,
        min(left.y + left.height, right.y + right.height) - max(left.y, right.y),
    )
    intersection = width * height
    return intersection / min(left.width * left.height, right.width * right.height)


def _crop(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise EvidenceSeparationError("Evidence source must be a color image")
    height, width = image.shape[:2]
    if bbox.x + bbox.width > width or bbox.y + bbox.height > height:
        raise EvidenceSeparationError("Evidence bounding box exceeds source image")
    crop = image[bbox.y : bbox.y + bbox.height, bbox.x : bbox.x + bbox.width]
    if crop.size == 0:
        raise EvidenceSeparationError("Evidence image crop is empty")
    return crop


def _line_words(words: tuple[OCRWord, ...], target: OCRWord) -> tuple[OCRWord, ...]:
    hierarchy = (
        target.block_number,
        target.paragraph_number,
        target.line_number,
    )
    if all(value is not None for value in hierarchy):
        return tuple(
            word
            for word in words
            if (word.block_number, word.paragraph_number, word.line_number) == hierarchy
        )
    target_center = target.bbox.y + target.bbox.height / 2
    return tuple(
        word
        for word in words
        if abs((word.bbox.y + word.bbox.height / 2) - target_center)
        <= max(target.bbox.height, word.bbox.height) * 0.6
    )


def _normalized_deviation(values: list[int], scale: float) -> float:
    if len(values) < 2:
        return 0.0
    center = float(median(values))
    return _unit(fmean(abs(value - center) for value in values) / scale * 2)


def _unit(value: float) -> float:
    return min(1.0, max(0.0, value))

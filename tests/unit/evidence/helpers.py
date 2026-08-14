"""Synthetic feature and page builders for evidence tests."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import cv2
import numpy as np

from app.domain.models.paper import PaperPage
from app.evidence.models import EvidenceFeatures, GeometryFeatures, InkFeatures


def features(
    *,
    confidence: float | None = 0.9,
    saturation: float = 0.02,
    red: float = 0.0,
    blue: float = 0.0,
    dark: float = 0.2,
    regularity: float = 0.9,
    irregularity: float = 0.1,
    density: float = 0.4,
    fragmentation: float = 0.2,
    isolation: float = 0.1,
    margin: float = 0.0,
    components: int = 8,
) -> EvidenceFeatures:
    return EvidenceFeatures(
        ink=InkFeatures(
            mean_saturation=saturation,
            saturation_std=0.1,
            foreground_ratio=0.25,
            red_foreground_ratio=red,
            blue_foreground_ratio=blue,
            dark_foreground_ratio=dark,
            local_contrast=0.5,
            edge_density=0.3,
            connected_component_count=components,
        ),
        geometry=GeometryFeatures(
            regularity=regularity,
            baseline_irregularity=irregularity,
            height_irregularity=irregularity,
            spacing_irregularity=irregularity,
            line_density=density,
            fragmentation=fragmentation,
            isolation=isolation,
            margin_position=margin,
            word_count_in_line=1 if isolation >= 0.65 else 5,
        ),
        ocr_confidence=confidence,
    )


def page(
    tmp_path: Path,
    *,
    width: int = 1000,
    height: int = 800,
    paper_id: UUID | None = None,
) -> PaperPage:
    image_path = (tmp_path / "canonical.png").resolve()
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    return PaperPage(
        paper_id=paper_id or uuid4(),
        page_number=1,
        image_path=image_path,
        width=width,
        height=height,
    )

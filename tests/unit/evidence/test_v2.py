"""Synthetic tests for deterministic evidence separation V2."""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import UUID

import cv2
import pytest
from pydantic import ValidationError

from app.core.exceptions import EvidenceSeparationError
from app.evaluation.evidence_redesign.runner import passes_teacher_safety_gate
from app.evidence.answer_regions_v2 import (
    AnswerSpaceDetectorV2,
    StudentAnswerRegionDetectorV2,
)
from app.evidence.classifier_v2 import EvidenceClassifierV2
from app.evidence.features_v2 import EvidenceFeatureExtractorV2
from app.evidence.grouping_v2 import group_evidence_components
from app.evidence.models import (
    AnswerRegionSignal,
    EvidenceRegion,
    EvidenceType,
    StudentAnswerRegion,
)
from app.evidence.models import (
    TestEvidence as EvidenceTestResult,
)
from app.evidence.separator_v2 import EvidenceSeparatorV2
from app.evidence.v2_models import (
    EvidenceComponent,
    EvidenceFeaturesV2,
    EvidenceGroup,
    EvidenceReasonV2,
    EvidenceSeparatorV2Config,
)
from app.ocr.models import BoundingBox
from scripts.evaluate_evidence_v2 import _write_new_json
from tests.unit.evidence.helpers import features, page


def _v2_features(**overrides: object) -> EvidenceFeaturesV2:
    values: dict[str, object] = {
        "mean_saturation": 0.05,
        "chromatic_pixel_ratio": 0.02,
        "color_consistency": 0.9,
        "foreground_ratio": 0.2,
        "ink_background_contrast": 0.6,
        "component_density": 0.5,
        "compactness": 0.6,
        "fragmentation": 0.2,
        "alignment": 0.9,
        "isolated_small_mark_likelihood": 0.1,
        "diagonal_dominance": 0.0,
        "crossing_stroke_likelihood": 0.0,
        "underline_likelihood": 0.0,
        "circle_likelihood": 0.0,
        "correction_stroke_likelihood": 0.0,
        "ocr_confidence": 0.9,
        "ocr_token_count": 5,
        "ocr_regularity": 0.9,
        "hierarchy_consistency": 0.9,
        "text_density": 0.5,
        "normalized_x": 0.3,
        "normalized_y": 0.3,
        "margin_proximity": 0.0,
        "answer_space_overlap": 0.0,
        "group_width_ratio": 0.4,
        "group_height_ratio": 0.1,
    }
    values.update(overrides)
    return EvidenceFeaturesV2.model_validate(values)


def _region(
    image_path: Path,
    paper_id: UUID,
    bbox: BoundingBox,
    *,
    kind: EvidenceType = EvidenceType.UNKNOWN,
    index: int = 0,
) -> EvidenceRegion:
    return EvidenceRegion(
        paper_id=paper_id,
        page_number=1,
        test_number=1,
        bbox=bbox,
        evidence_type=kind,
        confidence=0.6,
        signals=(),
        features=features(confidence=0.5, irregularity=0.6, fragmentation=0.6),
        source_word_indices=(index,),
        source_image_path=image_path,
        classification_strategy="synthetic",
    )


def _teacher_features(*, chromatic: float) -> EvidenceFeaturesV2:
    return _v2_features(
        mean_saturation=chromatic,
        chromatic_pixel_ratio=chromatic,
        component_density=0.05,
        isolated_small_mark_likelihood=0.95,
        diagonal_dominance=0.9,
        crossing_stroke_likelihood=0.9,
        correction_stroke_likelihood=0.8,
        ocr_confidence=None,
        ocr_token_count=0,
        ocr_regularity=0.1,
        hierarchy_consistency=0.2,
        text_density=0.05,
        margin_proximity=0.9,
        group_width_ratio=0.03,
        group_height_ratio=0.03,
    )


def test_v2_configuration_is_serializable_stable_and_immutable() -> None:
    config = EvidenceSeparatorV2Config(variant="v2-d")
    restored = EvidenceSeparatorV2Config.model_validate_json(config.model_dump_json())
    assert restored.fingerprint() == config.fingerprint()
    with pytest.raises(ValidationError):
        config.student_threshold = 0.1


def test_grouping_joins_nearby_line_components_but_not_large_whitespace(
    tmp_path: Path,
) -> None:
    paper_page = page(tmp_path)
    regions = (
        _region(
            paper_page.image_path,
            paper_page.paper_id,
            BoundingBox(x=100, y=100, width=40, height=25),
            index=0,
        ),
        _region(
            paper_page.image_path,
            paper_page.paper_id,
            BoundingBox(x=150, y=102, width=45, height=24),
            index=1,
        ),
        _region(
            paper_page.image_path,
            paper_page.paper_id,
            BoundingBox(x=700, y=500, width=45, height=24),
            index=2,
        ),
    )
    groups = group_evidence_components(
        regions, EvidenceSeparatorV2Config(variant="v2-d")
    )
    assert tuple(len(item.components) for item in groups) == (2, 1)
    assert groups[0].bbox == BoundingBox(x=100, y=100, width=95, height=26)


def test_feature_extractor_preserves_typed_geometry(tmp_path: Path) -> None:
    paper_page = page(tmp_path)
    image = cv2.imread(str(paper_page.image_path), cv2.IMREAD_COLOR)
    assert image is not None
    cv2.putText(image, "PRINT", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    base = _region(
        paper_page.image_path,
        paper_page.paper_id,
        BoundingBox(x=95, y=110, width=150, height=50),
    )
    group = EvidenceGroup(
        bbox=base.bbox,
        component_indices=(0,),
        components=(
            EvidenceComponent(
                component_index=0,
                bbox=base.bbox,
                source_word_indices=(0,),
                base_region=base,
            ),
        ),
    )
    measured = EvidenceFeatureExtractorV2().extract(
        image,
        group,
        test_region=BoundingBox(x=50, y=50, width=500, height=300),
    )
    assert measured.ocr_token_count == 1
    assert 0 <= measured.compactness <= 1
    assert 0 <= measured.normalized_x <= 1


@pytest.mark.parametrize("chromatic", [0.0, 0.7])
def test_black_or_colored_multi_signal_tick_is_teacher(chromatic: float) -> None:
    decision = EvidenceClassifierV2(EvidenceSeparatorV2Config(variant="v2-d")).classify(
        _teacher_features(chromatic=chromatic)
    )
    assert decision.evidence_type is EvidenceType.TEACHER_CANDIDATE
    assert EvidenceReasonV2.TICK_OR_CROSS_SHAPE in decision.reasons


@pytest.mark.parametrize("chromatic", [0.0, 0.7])
def test_black_or_blue_grouped_writing_is_student(chromatic: float) -> None:
    decision = EvidenceClassifierV2(
        EvidenceSeparatorV2Config(variant="v2-d", student_threshold=0.38)
    ).classify(
        _v2_features(
            chromatic_pixel_ratio=chromatic,
            component_density=0.8,
            fragmentation=0.8,
            isolated_small_mark_likelihood=0.05,
            ocr_confidence=0.25,
            ocr_regularity=0.2,
            hierarchy_consistency=0.3,
            text_density=0.15,
            answer_space_overlap=0.8,
        )
    )
    assert decision.evidence_type is EvidenceType.STUDENT_CANDIDATE


def test_regular_printed_evidence_is_printed() -> None:
    decision = EvidenceClassifierV2(EvidenceSeparatorV2Config(variant="v2-d")).classify(
        _v2_features()
    )
    assert decision.evidence_type is EvidenceType.PRINTED


def test_high_saturation_or_low_confidence_alone_remains_unknown() -> None:
    classifier = EvidenceClassifierV2(EvidenceSeparatorV2Config(variant="v2-d"))
    saturated = classifier.classify(
        _v2_features(
            chromatic_pixel_ratio=0.9,
            mean_saturation=0.9,
            ocr_confidence=None,
            ocr_token_count=0,
            ocr_regularity=0.55,
            hierarchy_consistency=0.5,
            text_density=0.05,
            component_density=0.1,
        )
    )
    low_confidence = classifier.classify(
        _v2_features(
            ocr_confidence=0.1,
            ocr_token_count=1,
            ocr_regularity=0.55,
            hierarchy_consistency=0.5,
            text_density=0.05,
            component_density=0.1,
        )
    )
    assert saturated.evidence_type is EvidenceType.UNKNOWN
    assert low_confidence.evidence_type is EvidenceType.UNKNOWN


def test_blank_answer_space_does_not_become_student_region(tmp_path: Path) -> None:
    paper_page = page(tmp_path)
    guide = StudentAnswerRegion(
        paper_id=paper_page.paper_id,
        page_number=1,
        test_number=1,
        bbox=BoundingBox(x=100, y=200, width=600, height=200),
        confidence=0.7,
        signals=(
            AnswerRegionSignal.WRITING_GUIDES,
            AnswerRegionSignal.BLANK_WRITING_SPACE,
        ),
        source_image_path=paper_page.image_path,
        detection_strategy="answer-region-v1",
    )
    base = EvidenceTestResult(
        paper_id=paper_page.paper_id,
        page_number=1,
        test_number=1,
        region_bbox=BoundingBox(x=50, y=50, width=800, height=600),
        answer_regions=(guide,),
    )
    spaces = AnswerSpaceDetectorV2().detect(base)
    answers = StudentAnswerRegionDetectorV2(
        EvidenceSeparatorV2Config(variant="v2-d")
    ).detect(base, (), spaces)
    assert len(spaces) == 1
    assert answers == ()


def test_student_regions_merge_on_same_line_and_clip_to_test(tmp_path: Path) -> None:
    paper_page = page(tmp_path)
    region_bbox = BoundingBox(x=100, y=100, width=500, height=300)
    evidence = (
        _region(
            paper_page.image_path,
            paper_page.paper_id,
            BoundingBox(x=105, y=150, width=60, height=30),
            kind=EvidenceType.STUDENT_CANDIDATE,
            index=0,
        ),
        _region(
            paper_page.image_path,
            paper_page.paper_id,
            BoundingBox(x=175, y=151, width=70, height=30),
            kind=EvidenceType.STUDENT_CANDIDATE,
            index=1,
        ),
    )
    base = EvidenceTestResult(
        paper_id=paper_page.paper_id,
        page_number=1,
        test_number=1,
        region_bbox=region_bbox,
        evidence_regions=evidence,
    )
    detected = StudentAnswerRegionDetectorV2(
        EvidenceSeparatorV2Config(variant="v2-d")
    ).detect(base, evidence, ())
    assert len(detected) == 1
    assert detected[0].bbox.x >= region_bbox.x
    assert (
        detected[0].bbox.x + detected[0].bbox.width <= region_bbox.x + region_bbox.width
    )


def test_separator_v2_preserves_source_and_provenance(tmp_path: Path) -> None:
    paper_page = page(tmp_path)
    before = hashlib.sha256(paper_page.image_path.read_bytes()).hexdigest()
    base_region = _region(
        paper_page.image_path,
        paper_page.paper_id,
        BoundingBox(x=100, y=100, width=90, height=30),
    )
    base = EvidenceTestResult(
        paper_id=paper_page.paper_id,
        page_number=1,
        test_number=2,
        region_bbox=BoundingBox(x=50, y=50, width=400, height=250),
        evidence_regions=(base_region.model_copy(update={"test_number": 2}),),
    )
    result = EvidenceSeparatorV2(
        EvidenceSeparatorV2Config(variant="v2-d")
    ).separate_existing(paper_page, base)
    assert result.paper_id == paper_page.paper_id
    assert (
        result.configuration_hash
        == EvidenceSeparatorV2Config(variant="v2-d").fingerprint()
    )
    assert hashlib.sha256(paper_page.image_path.read_bytes()).hexdigest() == before


def test_experiment_artifacts_refuse_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "private" / "experiment.json"
    _write_new_json(output, {"selected": False})
    with pytest.raises(EvidenceSeparationError, match="already exists"):
        _write_new_json(output, {"selected": True})


def test_teacher_safety_gate_rejects_regression() -> None:
    assert passes_teacher_safety_gate(3)
    assert not passes_teacher_safety_gate(4)
    with pytest.raises(ValueError, match="cannot be negative"):
        passes_teacher_safety_gate(-1)

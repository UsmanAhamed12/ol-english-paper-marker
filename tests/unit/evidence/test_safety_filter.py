"""Focused synthetic tests for the minimal post-V1 extraction safety policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.evidence.models import (
    AnswerRegionSignal,
    EvidenceRegion,
    EvidenceType,
    StudentAnswerRegion,
)
from app.evidence.models import TestEvidence as EvidenceTestResult
from app.evidence.safety_filter import StudentCandidateSafetyFilter
from app.evidence.safety_models import (
    CandidateDisposition,
    SafetyReason,
)
from app.evidence.safety_models import TestExtractionStatus as ExtractionStatus
from app.ocr.models import BoundingBox
from tests.unit.evidence.helpers import features, page


def _region(
    tmp_path: Path,
    box: BoundingBox,
    *,
    feature_overrides: dict[str, Any] | None = None,
    word: bool = True,
) -> EvidenceRegion:
    paper_page = page(tmp_path)
    return EvidenceRegion(
        paper_id=paper_page.paper_id,
        page_number=1,
        test_number=1,
        bbox=box,
        evidence_type=EvidenceType.STUDENT_CANDIDATE,
        confidence=0.65,
        signals=(),
        features=features(**(feature_overrides or {})),
        source_word_indices=(0,) if word else (),
        source_image_path=paper_page.image_path,
        classification_strategy="synthetic-v1",
    )


def _test(
    region: EvidenceRegion,
    *,
    others: tuple[EvidenceRegion, ...] = (),
    guides: bool = False,
) -> EvidenceTestResult:
    answer_regions: tuple[StudentAnswerRegion, ...] = ()
    if guides:
        answer_regions = (
            StudentAnswerRegion(
                paper_id=region.paper_id,
                page_number=1,
                test_number=1,
                bbox=BoundingBox(x=80, y=80, width=750, height=500),
                confidence=0.7,
                signals=(AnswerRegionSignal.WRITING_GUIDES,),
                source_evidence_indices=(0,),
                source_image_path=region.source_image_path,
                detection_strategy="synthetic-v1",
            ),
        )
    return EvidenceTestResult(
        paper_id=region.paper_id,
        page_number=1,
        test_number=1,
        region_bbox=BoundingBox(x=50, y=50, width=850, height=650),
        evidence_regions=(region, *others),
        answer_regions=answer_regions,
    )


def test_printed_guide_is_rejected(tmp_path: Path) -> None:
    region = _region(
        tmp_path,
        BoundingBox(x=100, y=200, width=600, height=8),
        feature_overrides={"regularity": 0.8, "density": 0.1},
        word=False,
    )
    result = StudentCandidateSafetyFilter().apply(_test(region))
    assert result.decisions[0].disposition is CandidateDisposition.REJECTED_NON_STUDENT
    assert SafetyReason.PRINTED_GUIDE_OR_RULE in result.decisions[0].reasons


def test_grid_or_border_is_rejected(tmp_path: Path) -> None:
    region = _region(
        tmp_path,
        BoundingBox(x=50, y=50, width=850, height=2),
        word=False,
    )
    result = StudentCandidateSafetyFilter().apply(_test(region))
    assert SafetyReason.GRID_OR_PAGE_BORDER in result.decisions[0].reasons


def test_tiny_noise_outside_answer_space_is_rejected(tmp_path: Path) -> None:
    region = _region(
        tmp_path,
        BoundingBox(x=400, y=300, width=3, height=3),
        feature_overrides={
            "components": 1,
            "isolation": 1.0,
            "confidence": 0.3,
            "regularity": 0.2,
        },
        word=False,
    )
    result = StudentCandidateSafetyFilter().apply(_test(region))
    assert SafetyReason.TINY_NOISE in result.decisions[0].reasons


def test_isolated_chromatic_teacher_risk_is_quarantined(tmp_path: Path) -> None:
    region = _region(
        tmp_path,
        BoundingBox(x=700, y=200, width=30, height=25),
        feature_overrides={"saturation": 0.5, "red": 0.3, "isolation": 1.0},
    )
    result = StudentCandidateSafetyFilter().apply(_test(region))
    assert result.status is ExtractionStatus.REVIEW_REQUIRED
    assert result.decisions[0].disposition is CandidateDisposition.REVIEW_REQUIRED
    assert result.answer_regions == ()


def test_uncertain_evidence_is_quarantined_instead_of_discarded(
    tmp_path: Path,
) -> None:
    region = _region(
        tmp_path,
        BoundingBox(x=300, y=250, width=35, height=24),
        feature_overrides={
            "confidence": 0.4,
            "regularity": 0.4,
            "density": 0.05,
        },
    ).model_copy(update={"evidence_type": EvidenceType.UNKNOWN})

    result = StudentCandidateSafetyFilter().apply(_test(region))

    assert result.status is ExtractionStatus.REVIEW_REQUIRED
    assert result.decisions[0].disposition is CandidateDisposition.REVIEW_REQUIRED
    assert SafetyReason.UNCERTAIN_CLASSIFICATION in result.decisions[0].reasons
    assert result.answer_regions == ()


def test_teacher_classification_is_quarantined_for_review(tmp_path: Path) -> None:
    region = _region(
        tmp_path,
        BoundingBox(x=300, y=250, width=35, height=24),
        feature_overrides={
            "confidence": 0.4,
            "regularity": 0.4,
            "density": 0.05,
        },
    ).model_copy(update={"evidence_type": EvidenceType.TEACHER_CANDIDATE})

    result = StudentCandidateSafetyFilter().apply(_test(region))

    assert result.status is ExtractionStatus.REVIEW_REQUIRED
    assert result.decisions[0].disposition is CandidateDisposition.REVIEW_REQUIRED
    assert SafetyReason.TEACHER_EVIDENCE_CLASSIFIED in result.decisions[0].reasons
    assert result.answer_regions == ()


def test_one_letter_answer_in_expected_space_is_preserved(tmp_path: Path) -> None:
    region = _region(
        tmp_path,
        BoundingBox(x=130, y=160, width=14, height=22),
        feature_overrides={
            "confidence": 0.3,
            "regularity": 0.2,
            "irregularity": 0.7,
            "fragmentation": 0.7,
            "components": 2,
        },
    )
    result = StudentCandidateSafetyFilter().apply(_test(region, guides=True))
    assert result.status is ExtractionStatus.SAFE
    assert len(result.answer_regions) == 1


def test_short_answer_words_merge_on_one_line(tmp_path: Path) -> None:
    first = _region(
        tmp_path,
        BoundingBox(x=140, y=170, width=55, height=30),
        feature_overrides={"confidence": 0.3, "irregularity": 0.7},
    )
    second = first.model_copy(
        update={
            "bbox": BoundingBox(x=210, y=172, width=60, height=28),
            "source_word_indices": (1,),
        }
    )
    result = StudentCandidateSafetyFilter().apply(_test(first, others=(second,)))
    assert len(result.answer_regions) == 1
    assert result.answer_regions[0].source_evidence_indices == (0, 1)


def test_paragraph_lines_are_preserved_as_bounded_regions(tmp_path: Path) -> None:
    first = _region(
        tmp_path,
        BoundingBox(x=140, y=170, width=260, height=35),
        feature_overrides={"confidence": 0.3, "irregularity": 0.7},
    )
    second = first.model_copy(
        update={
            "bbox": BoundingBox(x=145, y=225, width=250, height=35),
            "source_word_indices": (1,),
        }
    )
    result = StudentCandidateSafetyFilter().apply(_test(first, others=(second,)))
    assert result.answer_regions
    assert all(
        item.bbox.width * item.bbox.height <= 850 * 650 * 0.20
        for item in result.answer_regions
    )


def test_blank_guides_without_safe_evidence_are_empty(tmp_path: Path) -> None:
    printed = _region(
        tmp_path,
        BoundingBox(x=100, y=180, width=500, height=8),
        word=False,
    ).model_copy(update={"evidence_type": EvidenceType.PRINTED})
    result = StudentCandidateSafetyFilter().apply(_test(printed, guides=True))
    assert result.status is ExtractionStatus.EMPTY
    assert result.answer_regions == ()


def test_review_and_safe_evidence_make_test_review_required(tmp_path: Path) -> None:
    safe = _region(
        tmp_path,
        BoundingBox(x=150, y=180, width=100, height=35),
        feature_overrides={"confidence": 0.3, "irregularity": 0.7},
    )
    risky = safe.model_copy(
        update={
            "bbox": BoundingBox(x=700, y=220, width=30, height=25),
            "features": features(saturation=0.5, red=0.3, isolation=1.0),
            "source_word_indices": (1,),
        }
    )
    result = StudentCandidateSafetyFilter().apply(_test(safe, others=(risky,)))
    assert result.status is ExtractionStatus.REVIEW_REQUIRED
    assert len(result.answer_regions) == 1


def test_complete_v1_provenance_is_preserved(tmp_path: Path) -> None:
    region = _region(
        tmp_path,
        BoundingBox(x=150, y=180, width=100, height=35),
        feature_overrides={"confidence": 0.3, "irregularity": 0.7},
    )
    result = StudentCandidateSafetyFilter().apply(_test(region))
    assert result.decisions[0].evidence == region
    assert result.paper_id == region.paper_id
    assert result.answer_regions[0].source_image_path == region.source_image_path

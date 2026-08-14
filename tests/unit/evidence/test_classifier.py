"""Conservative multi-signal classification tests."""

from app.evidence.classifier import EvidenceClassifier
from app.evidence.models import EvidenceSignal, EvidenceType
from tests.unit.evidence.helpers import features


def test_black_regular_printed_text_is_supported() -> None:
    result = EvidenceClassifier().classify(features())

    assert result.evidence_type is EvidenceType.PRINTED
    assert EvidenceSignal.HIGH_OCR_CONFIDENCE in result.signals
    assert EvidenceSignal.REGULAR_GEOMETRY in result.signals


def test_blue_handwriting_like_evidence_uses_geometry_and_color() -> None:
    result = EvidenceClassifier().classify(
        features(
            blue=0.55,
            saturation=0.65,
            regularity=0.15,
            irregularity=0.82,
            fragmentation=0.7,
        )
    )

    assert result.evidence_type is EvidenceType.STUDENT_CANDIDATE
    assert EvidenceSignal.BLUE_INK_DOMINANT in result.signals
    assert EvidenceSignal.IRREGULAR_BASELINE in result.signals


def test_black_handwriting_like_evidence_does_not_require_color() -> None:
    result = EvidenceClassifier().classify(
        features(
            saturation=0.0,
            dark=0.2,
            regularity=0.1,
            irregularity=0.85,
            fragmentation=0.85,
            components=10,
        )
    )

    assert result.evidence_type is EvidenceType.STUDENT_CANDIDATE
    assert EvidenceSignal.FRAGMENTED_STROKES in result.signals


def test_red_isolated_annotation_is_teacher_candidate() -> None:
    result = EvidenceClassifier().classify(
        features(
            confidence=None,
            red=0.85,
            saturation=0.75,
            regularity=0.0,
            irregularity=0.5,
            fragmentation=0.3,
            isolation=1.0,
        )
    )

    assert result.evidence_type is EvidenceType.TEACHER_CANDIDATE
    assert EvidenceSignal.RED_INK_DOMINANT in result.signals
    assert EvidenceSignal.ISOLATED_MARK in result.signals


def test_low_confidence_alone_does_not_imply_student() -> None:
    result = EvidenceClassifier().classify(features(confidence=0.2))

    assert result.evidence_type is EvidenceType.UNKNOWN
    assert EvidenceSignal.LOW_OCR_CONFIDENCE in result.signals


def test_high_confidence_does_not_override_handwriting_geometry() -> None:
    result = EvidenceClassifier().classify(
        features(
            confidence=0.98,
            blue=0.45,
            saturation=0.55,
            regularity=0.1,
            irregularity=0.9,
            fragmentation=0.75,
        )
    )

    assert result.evidence_type is EvidenceType.STUDENT_CANDIDATE


def test_high_saturation_alone_does_not_imply_student() -> None:
    result = EvidenceClassifier().classify(
        features(
            confidence=None,
            saturation=0.9,
            red=0.0,
            blue=0.0,
            regularity=0.9,
            irregularity=0.1,
            fragmentation=0.1,
        )
    )

    assert result.evidence_type is EvidenceType.UNKNOWN


def test_mixed_conflicting_evidence_remains_unknown() -> None:
    result = EvidenceClassifier().classify(
        features(
            confidence=0.9,
            saturation=0.5,
            red=0.2,
            blue=0.2,
            regularity=0.55,
            irregularity=0.48,
            fragmentation=0.4,
        )
    )

    assert result.evidence_type is EvidenceType.UNKNOWN


def test_ambiguous_evidence_remains_unknown() -> None:
    result = EvidenceClassifier().classify(
        features(
            confidence=None,
            saturation=0.1,
            regularity=0.5,
            irregularity=0.5,
            density=0.1,
            fragmentation=0.3,
            components=2,
        )
    )

    assert result.evidence_type is EvidenceType.UNKNOWN

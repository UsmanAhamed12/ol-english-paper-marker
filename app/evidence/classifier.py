"""Conservative deterministic multi-signal evidence classification."""

from __future__ import annotations

from statistics import fmean

from app.evidence.models import (
    EvidenceClassification,
    EvidenceFeatures,
    EvidenceSignal,
    EvidenceType,
)

CLASSIFICATION_STRATEGY_VERSION = "evidence-separation-v1"


class EvidenceClassifier:
    """Classify only when independent signals provide sufficient support."""

    def classify(self, features: EvidenceFeatures) -> EvidenceClassification:
        """Return a conservative attribution with UNKNOWN as a valid result."""

        ink = features.ink
        geometry = features.geometry
        signals = _signals(features)
        irregularity = fmean(
            (
                geometry.baseline_irregularity,
                geometry.height_irregularity,
                geometry.spacing_irregularity,
            )
        )
        teacher_score = _unit(
            0.50 * ink.red_foreground_ratio
            + 0.20 * ink.mean_saturation
            + 0.20 * geometry.isolation
            + 0.10 * geometry.margin_position
        )
        printed_score = _unit(
            0.35 * (features.ocr_confidence or 0.0)
            + 0.30 * geometry.regularity
            + 0.20 * geometry.line_density
            + 0.15 * (1 - ink.mean_saturation)
        )
        student_score = _unit(
            0.40 * irregularity
            + 0.20 * geometry.fragmentation
            + 0.20 * ink.blue_foreground_ratio
            + 0.10 * ink.edge_density
            + 0.10 * ink.dark_foreground_ratio
        )

        teacher_supported = (
            ink.red_foreground_ratio >= 0.30
            and ink.mean_saturation >= 0.25
            and (geometry.isolation >= 0.65 or geometry.margin_position >= 1.0)
            and teacher_score >= 0.58
        )
        printed_supported = (
            features.ocr_confidence is not None
            and features.ocr_confidence >= 0.72
            and geometry.regularity >= 0.68
            and geometry.line_density >= 0.10
            and ink.red_foreground_ratio < 0.18
            and ink.blue_foreground_ratio < 0.18
            and printed_score >= 0.62
        )
        student_secondary = (
            ink.blue_foreground_ratio >= 0.12
            or geometry.fragmentation >= 0.55
            or (
                ink.dark_foreground_ratio >= 0.08 and ink.connected_component_count >= 4
            )
        )
        student_supported = (
            irregularity >= 0.52
            and student_secondary
            and ink.red_foreground_ratio < 0.30
            and student_score >= 0.42
        )

        supported = sum((teacher_supported, printed_supported, student_supported))
        if supported != 1:
            evidence_type = EvidenceType.UNKNOWN
            score = _unit(1 - max(teacher_score, printed_score, student_score))
        elif teacher_supported:
            evidence_type = EvidenceType.TEACHER_CANDIDATE
            score = teacher_score
        elif printed_supported:
            evidence_type = EvidenceType.PRINTED
            score = printed_score
        else:
            evidence_type = EvidenceType.STUDENT_CANDIDATE
            score = student_score
        return EvidenceClassification(
            evidence_type=evidence_type,
            score=score,
            signals=signals,
            strategy_version=CLASSIFICATION_STRATEGY_VERSION,
        )


def _signals(features: EvidenceFeatures) -> tuple[EvidenceSignal, ...]:
    ink = features.ink
    geometry = features.geometry
    signals: list[EvidenceSignal] = []
    if features.ocr_confidence is not None:
        if features.ocr_confidence >= 0.72:
            signals.append(EvidenceSignal.HIGH_OCR_CONFIDENCE)
        elif features.ocr_confidence <= 0.45:
            signals.append(EvidenceSignal.LOW_OCR_CONFIDENCE)
    if geometry.regularity >= 0.68:
        signals.append(EvidenceSignal.REGULAR_GEOMETRY)
    if geometry.baseline_irregularity >= 0.52:
        signals.append(EvidenceSignal.IRREGULAR_BASELINE)
    if geometry.height_irregularity >= 0.52:
        signals.append(EvidenceSignal.IRREGULAR_HEIGHT)
    if geometry.spacing_irregularity >= 0.52:
        signals.append(EvidenceSignal.IRREGULAR_SPACING)
    if geometry.line_density >= 0.20:
        signals.append(EvidenceSignal.DENSE_TEXT)
    if geometry.fragmentation >= 0.55:
        signals.append(EvidenceSignal.FRAGMENTED_STROKES)
    if ink.mean_saturation >= 0.25:
        signals.append(EvidenceSignal.CHROMATIC_INK)
    if ink.red_foreground_ratio >= 0.30:
        signals.append(EvidenceSignal.RED_INK_DOMINANT)
    if ink.blue_foreground_ratio >= 0.12:
        signals.append(EvidenceSignal.BLUE_INK_DOMINANT)
    if geometry.isolation >= 0.65:
        signals.append(EvidenceSignal.ISOLATED_MARK)
    if geometry.margin_position >= 1.0:
        signals.append(EvidenceSignal.MARGIN_POSITION)
    if ink.local_contrast >= 0.30:
        signals.append(EvidenceSignal.HIGH_LOCAL_CONTRAST)
    return tuple(signals)


def _unit(value: float) -> float:
    return min(1.0, max(0.0, value))

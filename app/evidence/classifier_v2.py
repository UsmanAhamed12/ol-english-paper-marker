"""Explainable deterministic classification for grouped V2 evidence."""

from __future__ import annotations

from app.evidence.models import EvidenceType
from app.evidence.v2_models import (
    EvidenceDecisionV2,
    EvidenceFeaturesV2,
    EvidenceReasonV2,
    EvidenceScoresV2,
    EvidenceSeparatorV2Config,
)


class EvidenceClassifierV2:
    """Score independent feature families and preserve genuine ambiguity."""

    def __init__(self, config: EvidenceSeparatorV2Config) -> None:
        self._config = config

    def classify(self, features: EvidenceFeaturesV2) -> EvidenceDecisionV2:
        """Return a non-probabilistic decision with machine-readable reasons."""

        irregularity = 1 - features.ocr_regularity
        group_extent = _unit(features.component_density / 0.55)
        printed = _unit(
            0.32 * (features.ocr_confidence or 0)
            + 0.23 * features.ocr_regularity
            + 0.16 * features.hierarchy_consistency
            + 0.16 * features.text_density
            + 0.13 * (1 - features.chromatic_pixel_ratio)
        )
        student = _unit(
            0.22 * irregularity
            + 0.18 * features.fragmentation
            + 0.16 * features.answer_space_overlap
            + 0.16 * group_extent
            + 0.12 * features.chromatic_pixel_ratio
            + 0.10 * features.ink_background_contrast
            + 0.06 * (1 - features.isolated_small_mark_likelihood)
        )
        shape = max(
            features.crossing_stroke_likelihood,
            features.diagonal_dominance,
            features.circle_likelihood,
        )
        teacher = _unit(
            0.23 * features.isolated_small_mark_likelihood
            + 0.18 * features.margin_proximity
            + 0.18 * shape
            + 0.14 * features.correction_stroke_likelihood
            + 0.10 * features.chromatic_pixel_ratio
            + 0.09 * (1 - group_extent)
            + 0.08 * features.underline_likelihood
        )
        reasons = _reasons(features, group_extent)
        teacher_reasons = sum(
            reason
            in {
                EvidenceReasonV2.ISOLATED_SMALL_MARK,
                EvidenceReasonV2.MARGIN_ANNOTATION,
                EvidenceReasonV2.TICK_OR_CROSS_SHAPE,
                EvidenceReasonV2.SCORE_LIKE_GEOMETRY,
                EvidenceReasonV2.CORRECTION_STROKE,
            }
            for reason in reasons
        )
        printed_reasons = sum(
            reason
            in {
                EvidenceReasonV2.REGULAR_OCR_LINE,
                EvidenceReasonV2.HIGH_OCR_SUPPORT,
                EvidenceReasonV2.DENSE_ALIGNED_TEXT,
            }
            for reason in reasons
        )
        student_reasons = sum(
            reason
            in {
                EvidenceReasonV2.IRREGULAR_GEOMETRY,
                EvidenceReasonV2.FRAGMENTED_COMPONENTS,
                EvidenceReasonV2.WRITING_SPACE_CONTEXT,
                EvidenceReasonV2.CONTIGUOUS_WRITING_GROUP,
                EvidenceReasonV2.CHROMATIC_DIFFERENCE,
            }
            for reason in reasons
        )
        supported: dict[EvidenceType, float] = {}
        if printed >= self._config.printed_threshold and printed_reasons >= 2:
            supported[EvidenceType.PRINTED] = printed
        if student >= self._config.student_threshold and student_reasons >= 2:
            supported[EvidenceType.STUDENT_CANDIDATE] = student
        teacher_risk = (
            self._config.enable_teacher_risk
            and teacher >= self._config.teacher_suppression_threshold
            and teacher_reasons >= 2
        )
        if (
            self._config.enable_teacher_risk
            and teacher >= self._config.teacher_threshold
            and teacher_reasons >= 2
        ):
            supported[EvidenceType.TEACHER_CANDIDATE] = teacher
        if teacher_risk:
            supported.pop(EvidenceType.STUDENT_CANDIDATE, None)

        ordered = sorted(supported.items(), key=lambda item: (-item[1], item[0].value))
        ambiguity_reasons: list[EvidenceReasonV2] = []
        decision = EvidenceType.UNKNOWN
        if not ordered or ordered[0][1] < self._config.minimum_decision_score:
            ambiguity_reasons.append(EvidenceReasonV2.WEAK_EVIDENCE)
        else:
            margin = ordered[0][1] - (ordered[1][1] if len(ordered) > 1 else 0)
            contradictory = (
                EvidenceType.STUDENT_CANDIDATE in supported
                and EvidenceType.TEACHER_CANDIDATE in supported
            )
            if contradictory:
                ambiguity_reasons.append(EvidenceReasonV2.CONTRADICTORY_AUTHORSHIP)
            if (
                self._config.enable_ambiguity_margin
                and margin < self._config.ambiguity_margin
            ):
                ambiguity_reasons.append(EvidenceReasonV2.NARROW_SCORE_MARGIN)
            if not ambiguity_reasons:
                decision = ordered[0][0]
        maximum = max(printed, student, teacher)
        ambiguity = _unit(1 - maximum + (0.25 if ambiguity_reasons else 0))
        return EvidenceDecisionV2(
            evidence_type=decision,
            scores=EvidenceScoresV2(
                printed=printed,
                student=student,
                teacher=teacher,
                ambiguity=ambiguity,
            ),
            reasons=tuple(dict.fromkeys((*reasons, *ambiguity_reasons))),
        )


def _reasons(
    features: EvidenceFeaturesV2, group_extent: float
) -> tuple[EvidenceReasonV2, ...]:
    reasons: list[EvidenceReasonV2] = []
    if features.ocr_regularity >= 0.68 and features.hierarchy_consistency >= 0.60:
        reasons.append(EvidenceReasonV2.REGULAR_OCR_LINE)
    if (features.ocr_confidence or 0) >= 0.72 and features.ocr_token_count > 0:
        reasons.append(EvidenceReasonV2.HIGH_OCR_SUPPORT)
    if features.text_density >= 0.18 and features.alignment >= 0.65:
        reasons.append(EvidenceReasonV2.DENSE_ALIGNED_TEXT)
    if features.ocr_regularity <= 0.50:
        reasons.append(EvidenceReasonV2.IRREGULAR_GEOMETRY)
    if features.fragmentation >= 0.45:
        reasons.append(EvidenceReasonV2.FRAGMENTED_COMPONENTS)
    if features.answer_space_overlap >= 0.20:
        reasons.append(EvidenceReasonV2.WRITING_SPACE_CONTEXT)
    if group_extent >= 0.30 and features.isolated_small_mark_likelihood < 0.72:
        reasons.append(EvidenceReasonV2.CONTIGUOUS_WRITING_GROUP)
    if features.chromatic_pixel_ratio >= 0.18 and features.color_consistency >= 0.55:
        reasons.append(EvidenceReasonV2.CHROMATIC_DIFFERENCE)
    if features.isolated_small_mark_likelihood >= 0.65:
        reasons.append(EvidenceReasonV2.ISOLATED_SMALL_MARK)
    if features.margin_proximity >= 0.72:
        reasons.append(EvidenceReasonV2.MARGIN_ANNOTATION)
    if (
        features.crossing_stroke_likelihood >= 0.60
        or features.diagonal_dominance >= 0.55
    ):
        reasons.append(EvidenceReasonV2.TICK_OR_CROSS_SHAPE)
    if (
        features.margin_proximity >= 0.60
        and features.isolated_small_mark_likelihood >= 0.55
        and features.ocr_token_count <= 2
    ):
        reasons.append(EvidenceReasonV2.SCORE_LIKE_GEOMETRY)
    if features.correction_stroke_likelihood >= 0.50:
        reasons.append(EvidenceReasonV2.CORRECTION_STROKE)
    return tuple(reasons)


def _unit(value: float) -> float:
    return min(1.0, max(0.0, value))

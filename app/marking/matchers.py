"""Small deterministic matchers for explicitly supported criterion types."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.marking.models import (
    CriterionComponentResult,
    CriterionDecision,
    MarkingReasonCode,
)
from app.marking.normalization import (
    contains_keyword_or_phrase,
    normalize_for_marking,
)
from app.marking_scheme.models import MarkingCriterion


@dataclass(frozen=True)
class MatcherOutcome:
    """Framework-independent output consumed by the marking engine."""

    awarded_marks: Decimal
    decision: CriterionDecision
    reason_codes: tuple[MarkingReasonCode, ...]
    matched_value: str | None = None
    matched_keywords: tuple[str, ...] = ()
    missing_keywords: tuple[str, ...] = ()
    component_results: tuple[CriterionComponentResult, ...] = ()


def match_exact(student_text: str, criterion: MarkingCriterion) -> MatcherOutcome:
    """Award all-or-nothing marks for one explicitly expected answer."""

    expected = criterion.accepted_answers[0]
    maximum = _maximum(criterion)
    if student_text == expected:
        return MatcherOutcome(
            maximum,
            CriterionDecision.EXACT_MATCH,
            (MarkingReasonCode.EXACT_ANSWER_MATCH,),
            matched_value=expected,
        )
    if normalize_for_marking(student_text) == normalize_for_marking(expected):
        return MatcherOutcome(
            maximum,
            CriterionDecision.NORMALIZED_MATCH,
            (MarkingReasonCode.NORMALIZED_ANSWER_MATCH,),
            matched_value=expected,
        )
    return _no_match()


def match_accepted_variant(
    student_text: str, criterion: MarkingCriterion
) -> MatcherOutcome:
    """Match only one of the marking scheme's explicit accepted variants."""

    normalized = normalize_for_marking(student_text)
    for variant in criterion.accepted_answers:
        if normalized == normalize_for_marking(variant):
            return MatcherOutcome(
                _maximum(criterion),
                CriterionDecision.ACCEPTED_VARIANT_MATCH,
                (MarkingReasonCode.ACCEPTED_VARIANT_MATCH,),
                matched_value=variant,
            )
    return _no_match()


def match_keywords(student_text: str, criterion: MarkingCriterion) -> MatcherOutcome:
    """Match required tokens/phrases without substring or semantic guessing."""

    if criterion.partial_credit_allowed and criterion.components:
        component_results = tuple(
            CriterionComponentResult(
                component_id=component.component_id,
                awarded_marks=(
                    component.maximum_marks
                    if all(
                        contains_keyword_or_phrase(student_text, keyword)
                        for keyword in component.keywords
                    )
                    else Decimal("0")
                ),
                maximum_marks=component.maximum_marks,
                matched=all(
                    contains_keyword_or_phrase(student_text, keyword)
                    for keyword in component.keywords
                ),
                matched_value=None,
            )
            for component in criterion.components
        )
        awarded = sum((item.awarded_marks for item in component_results), Decimal("0"))
        matched = tuple(
            keyword
            for keyword in criterion.keywords
            if contains_keyword_or_phrase(student_text, keyword)
        )
        missing = tuple(
            keyword for keyword in criterion.keywords if keyword not in matched
        )
        return MatcherOutcome(
            awarded,
            (
                CriterionDecision.KEYWORD_MATCH
                if matched
                else CriterionDecision.NO_MATCH
            ),
            tuple(
                reason
                for condition, reason in (
                    (bool(matched), MarkingReasonCode.REQUIRED_KEYWORD_MATCH),
                    (bool(missing), MarkingReasonCode.MISSING_REQUIRED_KEYWORD),
                )
                if condition
            ),
            matched_keywords=matched,
            missing_keywords=missing,
            component_results=component_results,
        )
    matched = tuple(
        keyword
        for keyword in criterion.keywords
        if contains_keyword_or_phrase(student_text, keyword)
    )
    missing = tuple(keyword for keyword in criterion.keywords if keyword not in matched)
    if not missing:
        return MatcherOutcome(
            _maximum(criterion),
            CriterionDecision.KEYWORD_MATCH,
            (MarkingReasonCode.REQUIRED_KEYWORD_MATCH,),
            matched_keywords=matched,
        )
    return MatcherOutcome(
        Decimal("0"),
        CriterionDecision.NO_MATCH,
        (MarkingReasonCode.MISSING_REQUIRED_KEYWORD,),
        matched_keywords=matched,
        missing_keywords=missing,
    )


def match_multi_part(student_text: str, criterion: MarkingCriterion) -> MatcherOutcome:
    """Award only the marks explicitly allocated to matched components."""

    results: list[CriterionComponentResult] = []
    for component in criterion.components:
        matched_value = next(
            (
                value
                for value in component.accepted_answers
                if contains_keyword_or_phrase(student_text, value)
            ),
            None,
        )
        keyword_match = bool(component.keywords) and all(
            contains_keyword_or_phrase(student_text, keyword)
            for keyword in component.keywords
        )
        matched = matched_value is not None or keyword_match
        results.append(
            CriterionComponentResult(
                component_id=component.component_id,
                awarded_marks=component.maximum_marks if matched else Decimal("0"),
                maximum_marks=component.maximum_marks,
                matched=matched,
                matched_value=matched_value,
            )
        )
    awarded = sum((item.awarded_marks for item in results), Decimal("0"))
    reasons = tuple(
        dict.fromkeys(
            MarkingReasonCode.MULTI_PART_COMPONENT_MATCH
            if item.matched
            else MarkingReasonCode.MULTI_PART_COMPONENT_MISMATCH
            for item in results
        )
    )
    return MatcherOutcome(
        awarded,
        CriterionDecision.MULTI_PART_MATCH,
        reasons,
        component_results=tuple(results),
    )


def _maximum(criterion: MarkingCriterion) -> Decimal:
    if criterion.maximum_marks is None:
        raise ValueError("Matcher requires explicit maximum marks")
    return criterion.maximum_marks


def _no_match() -> MatcherOutcome:
    return MatcherOutcome(
        Decimal("0"),
        CriterionDecision.NO_MATCH,
        (MarkingReasonCode.NO_DETERMINISTIC_MATCH,),
    )

"""Alignment-aware deterministic marking and Decimal-safe aggregation."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from time import perf_counter

from app.answers.models import AnswerExtractionStatus, StructuredStudentAnswer
from app.core.exceptions import MarkingEngineError
from app.marking.matchers import (
    MatcherOutcome,
    match_accepted_variant,
    match_exact,
    match_keywords,
    match_multi_part,
)
from app.marking.models import (
    CriterionDecision,
    CriterionMarkResult,
    MarkingProvenance,
    MarkingReasonCode,
    MarkingStatus,
    MarkingWarning,
    PaperMarkingResult,
    StudentEvidenceReference,
    TestMarkResult,
)
from app.marking.normalization import MARKING_NORMALIZATION_VERSION
from app.marking_scheme.models import (
    AlignedPaper,
    AlignedTest,
    AlignmentStatus,
    CriterionType,
    MarkingCriterion,
)

ENGINE_VERSION = "deterministic-marking-v1"
Matcher = Callable[[str, MarkingCriterion], MatcherOutcome]
_MATCHERS: dict[CriterionType, tuple[str, Matcher]] = {
    CriterionType.EXACT_ANSWER: ("exact-answer-v1", match_exact),
    CriterionType.ACCEPTED_VARIANTS: ("accepted-variants-v1", match_accepted_variant),
    CriterionType.KEYWORD: ("keyword-v1", match_keywords),
    CriterionType.MULTI_PART: ("multi-part-v1", match_multi_part),
}
_SUBJECTIVE = {
    CriterionType.CONTENT_POINT,
    CriterionType.LANGUAGE_QUALITY,
    CriterionType.FORMAT_REQUIREMENT,
    CriterionType.HOLISTIC,
}


class DeterministicMarkingEngine:
    """Award only marks supported by explicit, reproducible scheme rules."""

    def mark_paper(self, aligned: AlignedPaper) -> PaperMarkingResult:
        started = perf_counter()
        tests = tuple(self.mark_test(item) for item in aligned.tests)
        confirmed = sum((item.awarded_marks for item in tests), Decimal("0"))
        maximum = sum(
            (
                item.maximum_marks
                if item.maximum_marks is not None
                else sum(
                    (
                        criterion.maximum_marks
                        for criterion in item.criterion_results
                        if criterion.maximum_marks is not None
                    ),
                    Decimal("0"),
                )
                for item in tests
            ),
            Decimal("0"),
        )
        if confirmed > maximum:
            raise MarkingEngineError("Paper confirmed marks exceed known maximum")
        review_marks = sum((item.review_required_marks for item in tests), Decimal("0"))
        unsupported_marks = sum(
            (item.unsupported_marks for item in tests), Decimal("0")
        )
        partial = any(
            item.status not in {MarkingStatus.MARKED, MarkingStatus.NOT_ATTEMPTED}
            for item in tests
        )
        return PaperMarkingResult(
            scheme_id=aligned.scheme_id,
            student_paper_id=aligned.paper_id,
            test_results=tests,
            confirmed_marks=confirmed,
            maximum_known_marks=maximum,
            review_required_marks=review_marks,
            unsupported_marks=unsupported_marks,
            tests_marked=tuple(
                item.test_number
                for item in tests
                if item.status in {MarkingStatus.MARKED, MarkingStatus.NOT_ATTEMPTED}
            ),
            tests_review_required=tuple(
                item.test_number
                for item in tests
                if item.review_required_marks > 0
                or item.status
                in {
                    MarkingStatus.REVIEW_REQUIRED,
                    MarkingStatus.OCR_FAILED,
                    MarkingStatus.SCHEME_INCOMPLETE,
                    MarkingStatus.STRUCTURE_INCOMPLETE,
                }
            ),
            tests_unsupported=tuple(
                item.test_number
                for item in tests
                if item.unsupported_marks > 0
                or item.status is MarkingStatus.UNSUPPORTED_CRITERION
            ),
            warnings=(MarkingWarning.PAPER_PARTIALLY_MARKED,) if partial else (),
            processing_duration_ms=(perf_counter() - started) * 1000,
            provenance=_provenance("paper-aggregation-v1"),
        )

    def mark_test(self, aligned: AlignedTest) -> TestMarkResult:
        started = perf_counter()
        scheme = aligned.marking_scheme_test
        if scheme is None:
            return _unscorable_test(
                aligned,
                MarkingStatus.SCHEME_INCOMPLETE,
                MarkingWarning.SCHEME_TEST_MISSING,
                "missing-scheme-test-v1",
                started,
            )
        maximum = scheme.maximum_marks
        if aligned.student_answer is None:
            results = tuple(
                _unresolved_criterion(
                    criterion,
                    None,
                    MarkingStatus.STRUCTURE_INCOMPLETE,
                    MarkingReasonCode.MISSING_STUDENT_TEST,
                    MarkingWarning.STUDENT_TEST_MISSING,
                )
                for criterion in scheme.criteria
            )
        elif aligned.status is AlignmentStatus.STUDENT_REVIEW_REQUIRED:
            results = tuple(
                self._review_criterion(aligned.student_answer, criterion)
                for criterion in scheme.criteria
            )
        elif aligned.status is AlignmentStatus.STUDENT_OCR_FAILED:
            results = tuple(
                _unresolved_criterion(
                    criterion,
                    aligned.student_answer,
                    MarkingStatus.OCR_FAILED,
                    MarkingReasonCode.STUDENT_OCR_FAILED,
                    MarkingWarning.OCR_EVIDENCE_UNAVAILABLE,
                )
                for criterion in scheme.criteria
            )
        elif aligned.status is AlignmentStatus.STRUCTURE_INCOMPLETE:
            results = tuple(
                _unresolved_criterion(
                    criterion,
                    aligned.student_answer,
                    MarkingStatus.STRUCTURE_INCOMPLETE,
                    MarkingReasonCode.STUDENT_STRUCTURE_INCOMPLETE,
                    MarkingWarning.STRUCTURE_EVIDENCE_INCOMPLETE,
                )
                for criterion in scheme.criteria
            )
        else:
            results = tuple(
                self._mark_criterion(aligned.student_answer, criterion)
                for criterion in scheme.criteria
            )
        return _aggregate_test(aligned.test_number, results, maximum, started)

    def _mark_criterion(
        self,
        answer: StructuredStudentAnswer,
        criterion: MarkingCriterion,
    ) -> CriterionMarkResult:
        evidence = _student_reference(answer)
        if answer.status is AnswerExtractionStatus.REVIEW_REQUIRED:
            return self._review_criterion(answer, criterion)
        if answer.status is AnswerExtractionStatus.OCR_FAILED:
            return _unresolved_criterion(
                criterion,
                answer,
                MarkingStatus.OCR_FAILED,
                MarkingReasonCode.STUDENT_OCR_FAILED,
                MarkingWarning.OCR_EVIDENCE_UNAVAILABLE,
            )
        if answer.status is AnswerExtractionStatus.STRUCTURE_INCOMPLETE:
            return _unresolved_criterion(
                criterion,
                answer,
                MarkingStatus.STRUCTURE_INCOMPLETE,
                MarkingReasonCode.STUDENT_STRUCTURE_INCOMPLETE,
                MarkingWarning.STRUCTURE_EVIDENCE_INCOMPLETE,
            )
        if criterion.criterion_type in _SUBJECTIVE or (
            criterion.criterion_type is CriterionType.UNKNOWN
        ):
            reason = (
                MarkingReasonCode.UNKNOWN_CRITERION
                if criterion.criterion_type is CriterionType.UNKNOWN
                else MarkingReasonCode.UNSUPPORTED_SUBJECTIVE_CRITERION
            )
            return _unresolved_criterion(
                criterion,
                answer,
                MarkingStatus.UNSUPPORTED_CRITERION,
                reason,
                MarkingWarning.SUBJECTIVE_CRITERION_UNSUPPORTED,
            )
        configuration_error = _criterion_configuration_error(criterion)
        if configuration_error is not None:
            return _unresolved_criterion(
                criterion,
                answer,
                MarkingStatus.SCHEME_INCOMPLETE,
                configuration_error,
                MarkingWarning.MARK_ALLOCATION_UNKNOWN,
            )
        if answer.status is AnswerExtractionStatus.EMPTY:
            return CriterionMarkResult(
                criterion_id=criterion.criterion_id,
                test_number=criterion.test_number,
                criterion_type=criterion.criterion_type,
                status=MarkingStatus.NOT_ATTEMPTED,
                student_evidence=evidence,
                awarded_marks=Decimal("0"),
                maximum_marks=criterion.maximum_marks,
                decision=CriterionDecision.EMPTY_ANSWER,
                comparison_would_match=False,
                reason_codes=(MarkingReasonCode.EMPTY_STUDENT_ANSWER,),
                provenance=_criterion_provenance(criterion, "empty-answer-v1"),
            )
        matcher_name, matcher = _MATCHERS[criterion.criterion_type]
        outcome = matcher(answer.text, criterion)
        return CriterionMarkResult(
            criterion_id=criterion.criterion_id,
            test_number=criterion.test_number,
            criterion_type=criterion.criterion_type,
            status=MarkingStatus.MARKED,
            student_evidence=evidence,
            awarded_marks=outcome.awarded_marks,
            maximum_marks=criterion.maximum_marks,
            decision=outcome.decision,
            comparison_would_match=outcome.awarded_marks > 0,
            matched_value=outcome.matched_value,
            matched_keywords=outcome.matched_keywords,
            missing_keywords=outcome.missing_keywords,
            component_results=outcome.component_results,
            reason_codes=outcome.reason_codes,
            provenance=_criterion_provenance(criterion, matcher_name),
        )

    def _review_criterion(
        self,
        answer: StructuredStudentAnswer,
        criterion: MarkingCriterion,
    ) -> CriterionMarkResult:
        configuration_error = _criterion_configuration_error(criterion)
        matcher_entry = _MATCHERS.get(criterion.criterion_type)
        if configuration_error is not None or matcher_entry is None:
            return _unresolved_criterion(
                criterion,
                answer,
                MarkingStatus.REVIEW_REQUIRED,
                MarkingReasonCode.STUDENT_REVIEW_REQUIRED,
                MarkingWarning.MANUAL_REVIEW_REQUIRED,
            )
        matcher_name, matcher = matcher_entry
        outcome = matcher(answer.text, criterion)
        return CriterionMarkResult(
            criterion_id=criterion.criterion_id,
            test_number=criterion.test_number,
            criterion_type=criterion.criterion_type,
            status=MarkingStatus.REVIEW_REQUIRED,
            student_evidence=_student_reference(answer),
            awarded_marks=None,
            maximum_marks=criterion.maximum_marks,
            decision=CriterionDecision.MANUAL_REVIEW,
            comparison_would_match=outcome.awarded_marks > 0,
            matched_value=outcome.matched_value,
            matched_keywords=outcome.matched_keywords,
            missing_keywords=outcome.missing_keywords,
            reason_codes=(MarkingReasonCode.STUDENT_REVIEW_REQUIRED,),
            warnings=(MarkingWarning.MANUAL_REVIEW_REQUIRED,),
            provenance=_criterion_provenance(
                criterion, f"{matcher_name}-review-evidence"
            ),
        )


def _criterion_configuration_error(
    criterion: MarkingCriterion,
) -> MarkingReasonCode | None:
    if criterion.maximum_marks is None:
        return MarkingReasonCode.MISSING_MAXIMUM_MARKS
    if (
        criterion.criterion_type
        in {
            CriterionType.EXACT_ANSWER,
            CriterionType.ACCEPTED_VARIANTS,
        }
        and not criterion.accepted_answers
    ):
        return MarkingReasonCode.MISSING_COMPONENT_ALLOCATION
    if criterion.criterion_type is CriterionType.KEYWORD and not criterion.keywords:
        return MarkingReasonCode.MISSING_COMPONENT_ALLOCATION
    if (
        criterion.criterion_type is CriterionType.KEYWORD
        and criterion.partial_credit_allowed
        and (
            not criterion.components
            or any(not component.keywords for component in criterion.components)
        )
    ):
        return MarkingReasonCode.MISSING_COMPONENT_ALLOCATION
    if (
        criterion.criterion_type is CriterionType.MULTI_PART
        and not criterion.components
    ):
        return MarkingReasonCode.MISSING_COMPONENT_ALLOCATION
    return None


def _student_reference(answer: StructuredStudentAnswer) -> StudentEvidenceReference:
    return StudentEvidenceReference(
        paper_id=answer.paper_id,
        test_number=answer.test_number,
        extraction_status=answer.status,
        original_text=answer.text,
        pages=answer.pages,
    )


def _unresolved_criterion(
    criterion: MarkingCriterion,
    answer: StructuredStudentAnswer | None,
    status: MarkingStatus,
    reason: MarkingReasonCode,
    warning: MarkingWarning,
) -> CriterionMarkResult:
    return CriterionMarkResult(
        criterion_id=criterion.criterion_id,
        test_number=criterion.test_number,
        criterion_type=criterion.criterion_type,
        status=status,
        student_evidence=_student_reference(answer) if answer else None,
        awarded_marks=None,
        maximum_marks=criterion.maximum_marks,
        decision=CriterionDecision.MANUAL_REVIEW,
        reason_codes=(reason,),
        warnings=(warning,),
        provenance=_criterion_provenance(criterion, "unresolved-policy-v1"),
    )


def _aggregate_test(
    test_number: int,
    results: tuple[CriterionMarkResult, ...],
    maximum: Decimal | None,
    started: float,
) -> TestMarkResult:
    confirmed = sum(
        (item.awarded_marks for item in results if item.awarded_marks is not None),
        Decimal("0"),
    )
    if maximum is not None and confirmed > maximum:
        raise MarkingEngineError("Criterion awards exceed Test maximum")
    review = sum(
        (
            item.maximum_marks
            for item in results
            if item.maximum_marks is not None
            and item.status
            in {
                MarkingStatus.REVIEW_REQUIRED,
                MarkingStatus.OCR_FAILED,
                MarkingStatus.SCHEME_INCOMPLETE,
                MarkingStatus.STRUCTURE_INCOMPLETE,
            }
        ),
        Decimal("0"),
    )
    unsupported = sum(
        (
            item.maximum_marks
            for item in results
            if item.maximum_marks is not None
            and item.status is MarkingStatus.UNSUPPORTED_CRITERION
        ),
        Decimal("0"),
    )
    allocated = sum(
        (item.maximum_marks for item in results if item.maximum_marks is not None),
        Decimal("0"),
    )
    warnings: list[MarkingWarning] = []
    if maximum is not None and allocated > maximum:
        raise MarkingEngineError("Criterion allocations exceed Test maximum")
    if maximum is not None and allocated < maximum:
        review += maximum - allocated
        warnings.append(MarkingWarning.MARK_ALLOCATION_UNKNOWN)
    status = _test_status(results, has_unallocated=bool(warnings))
    return TestMarkResult(
        test_number=test_number,
        status=status,
        criterion_results=results,
        awarded_marks=confirmed,
        maximum_marks=maximum,
        automatically_marked_marks=confirmed,
        review_required_marks=review,
        unsupported_marks=unsupported,
        warnings=tuple(warnings),
        processing_duration_ms=(perf_counter() - started) * 1000,
        provenance=_provenance("test-aggregation-v1"),
    )


def _test_status(
    results: tuple[CriterionMarkResult, ...], *, has_unallocated: bool
) -> MarkingStatus:
    statuses = {item.status for item in results}
    resolved = {MarkingStatus.MARKED, MarkingStatus.NOT_ATTEMPTED}
    if not results or has_unallocated:
        return (
            MarkingStatus.PARTIALLY_MARKED
            if statuses & resolved
            else MarkingStatus.SCHEME_INCOMPLETE
        )
    if statuses <= resolved:
        return (
            MarkingStatus.NOT_ATTEMPTED
            if statuses == {MarkingStatus.NOT_ATTEMPTED}
            else MarkingStatus.MARKED
        )
    if statuses & resolved:
        return MarkingStatus.PARTIALLY_MARKED
    if len(statuses) == 1:
        return next(iter(statuses))
    return MarkingStatus.PARTIALLY_MARKED


def _unscorable_test(
    aligned: AlignedTest,
    status: MarkingStatus,
    warning: MarkingWarning,
    matcher: str,
    started: float,
) -> TestMarkResult:
    return TestMarkResult(
        test_number=aligned.test_number,
        status=status,
        criterion_results=(),
        awarded_marks=Decimal("0"),
        maximum_marks=None,
        automatically_marked_marks=Decimal("0"),
        review_required_marks=Decimal("0"),
        unsupported_marks=Decimal("0"),
        warnings=(warning,),
        processing_duration_ms=(perf_counter() - started) * 1000,
        provenance=_provenance(matcher),
    )


def _criterion_provenance(
    criterion: MarkingCriterion, matcher: str
) -> MarkingProvenance:
    return MarkingProvenance(
        engine_version=ENGINE_VERSION,
        normalization_version=MARKING_NORMALIZATION_VERSION,
        matcher=matcher,
        criterion_id=criterion.criterion_id,
        criterion_source_page=criterion.provenance.source.page_number,
    )


def _provenance(matcher: str) -> MarkingProvenance:
    return MarkingProvenance(
        engine_version=ENGINE_VERSION,
        normalization_version=MARKING_NORMALIZATION_VERSION,
        matcher=matcher,
    )

"""Synthetic validation for the deterministic Phase 5B marking engine."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.answers.models import (
    AnswerExtractionProvenance,
    AnswerExtractionStatus,
    AnswerFragment,
    AnswerFragmentStatus,
    AnswerSourceContext,
    StructuredStudentAnswer,
)
from app.core.exceptions import MarkingEngineError
from app.marking.engine import DeterministicMarkingEngine
from app.marking.models import (
    CriterionDecision,
    MarkingReasonCode,
    MarkingStatus,
    MarkingWarning,
    PaperMarkingResult,
)
from app.marking.models import TestMarkResult as MarkTestResult
from app.marking.normalization import (
    contains_keyword_or_phrase,
    normalize_for_marking,
)
from app.marking.serialization import marking_result_json, write_private_marking_result
from app.marking_scheme.models import (
    AlignedPaper,
    AlignedTest,
    AlignmentStatus,
    CriterionComponent,
    CriterionParserRule,
    CriterionProvenance,
    CriterionType,
    MarkingCriterion,
    MarkingSchemeTest,
    SchemeSourceRegion,
)
from app.ocr.models import BoundingBox
from app.structure.models import MarkerDetectionStrategy

PAPER_ID = UUID("10000000-0000-0000-0000-000000000001")
SCHEME_ID = UUID("20000000-0000-0000-0000-000000000002")


def _criterion(
    criterion_type: CriterionType,
    *,
    maximum: Decimal | None = Decimal("1"),
    accepted: tuple[str, ...] = (),
    keywords: tuple[str, ...] = (),
    components: tuple[CriterionComponent, ...] = (),
    partial: bool | None = None,
    index: int = 1,
) -> MarkingCriterion:
    return MarkingCriterion(
        criterion_id=f"test-01-criterion-{index:03d}",
        test_number=1,
        criterion_type=criterion_type,
        description="Synthetic criterion",
        accepted_answers=accepted,
        keywords=keywords,
        components=components,
        maximum_marks=maximum,
        partial_credit_allowed=partial,
        provenance=CriterionProvenance(
            parser_rule=CriterionParserRule.PRESERVED_UNKNOWN,
            source=SchemeSourceRegion(page_number=1, raw_source_text="synthetic"),
        ),
    )


def _components() -> tuple[CriterionComponent, ...]:
    return (
        CriterionComponent(
            component_id="component-01",
            accepted_answers=("A",),
            maximum_marks=Decimal("1"),
        ),
        CriterionComponent(
            component_id="component-02",
            accepted_answers=("B",),
            maximum_marks=Decimal("1"),
        ),
        CriterionComponent(
            component_id="component-03",
            accepted_answers=("C",),
            maximum_marks=Decimal("1"),
        ),
    )


def _answer(
    text: str,
    status: AnswerExtractionStatus = AnswerExtractionStatus.EXTRACTED,
    *,
    paper_id: UUID = PAPER_ID,
) -> StructuredStudentAnswer:
    fragments: tuple[AnswerFragment, ...] = ()
    if status is AnswerExtractionStatus.EXTRACTED:
        fragments = (
            AnswerFragment(
                fragment_index=0,
                source_region_index=0,
                paper_id=paper_id,
                test_number=1,
                page_number=1,
                bbox=BoundingBox(x=10, y=10, width=100, height=20),
                status=AnswerFragmentStatus.EXTRACTED,
                raw_text=text,
                normalized_text=text,
                provider="fake",
                processing_duration_ms=1,
            ),
        )
    return StructuredStudentAnswer(
        paper_id=paper_id,
        test_number=1,
        status=status,
        text=text,
        pages=(1,),
        ocr_fragments=fragments,
        provenance=AnswerExtractionProvenance(
            extraction_version="synthetic-v1",
            test_label="Test 01",
            marker_page_number=1,
            marker_bbox=BoundingBox(x=0, y=0, width=20, height=10),
            marker_strategy=MarkerDetectionStrategy.EXACT_TOKENS,
            marker_confidence=1,
            marker_source_word_indices=(0,),
            safety_policy_versions=("safety-v1",),
            ocr_provider="fake",
            source_context=AnswerSourceContext.RUNTIME_UNMARKED,
        ),
        processing_duration_ms=1,
    )


def _aligned(
    criteria: tuple[MarkingCriterion, ...],
    *,
    answer: StructuredStudentAnswer | None = None,
    status: AlignmentStatus = AlignmentStatus.ALIGNED,
    test_maximum: Decimal | None = None,
) -> AlignedTest:
    maximum = test_maximum
    if maximum is None and all(item.maximum_marks is not None for item in criteria):
        maximum = sum(
            (item.maximum_marks for item in criteria if item.maximum_marks is not None),
            Decimal("0"),
        )
    scheme = MarkingSchemeTest(
        test_number=1,
        maximum_marks=maximum,
        criteria=criteria,
        source_pages=(1,),
        source_regions=(
            SchemeSourceRegion(page_number=1, raw_source_text="synthetic"),
        ),
    )
    return AlignedTest(
        test_number=1,
        status=status,
        student_answer=answer,
        marking_scheme_test=scheme,
    )


def _paper(test: AlignedTest) -> AlignedPaper:
    summaries: dict[str, tuple[int, ...]] = {
        "aligned_test_numbers": (),
        "student_missing_test_numbers": (),
        "scheme_missing_test_numbers": (),
        "review_required_test_numbers": (),
        "ocr_failed_test_numbers": (),
        "structure_incomplete_test_numbers": (),
    }
    field = {
        AlignmentStatus.ALIGNED: "aligned_test_numbers",
        AlignmentStatus.STUDENT_TEST_MISSING: "student_missing_test_numbers",
        AlignmentStatus.SCHEME_TEST_MISSING: "scheme_missing_test_numbers",
        AlignmentStatus.STUDENT_REVIEW_REQUIRED: "review_required_test_numbers",
        AlignmentStatus.STUDENT_OCR_FAILED: "ocr_failed_test_numbers",
        AlignmentStatus.STRUCTURE_INCOMPLETE: "structure_incomplete_test_numbers",
    }[test.status]
    summaries[field] = (test.test_number,)
    return AlignedPaper(
        paper_id=PAPER_ID,
        scheme_id=SCHEME_ID,
        tests=(test,),
        **summaries,
    )


def _mark(
    criterion: MarkingCriterion,
    text: str,
    *,
    answer_status: AnswerExtractionStatus = AnswerExtractionStatus.EXTRACTED,
) -> MarkTestResult:
    answer = _answer(text, answer_status)
    return DeterministicMarkingEngine().mark_test(_aligned((criterion,), answer=answer))


def test_exact_match_awards_full_marks() -> None:
    result = _mark(
        _criterion(CriterionType.EXACT_ANSWER, accepted=("Colombo",)), "Colombo"
    )
    criterion = result.criterion_results[0]
    assert criterion.awarded_marks == 1
    assert criterion.decision is CriterionDecision.EXACT_MATCH


def test_exact_mismatch_awards_zero() -> None:
    result = _mark(_criterion(CriterionType.EXACT_ANSWER, accepted=("cat",)), "dog")
    assert result.awarded_marks == 0
    assert result.criterion_results[0].decision is CriterionDecision.NO_MATCH


@pytest.mark.parametrize("student", [" COLOMBO ", "Colombo.\n", "COLOMBO"])
def test_conservative_normalization_matches(student: str) -> None:
    result = _mark(
        _criterion(CriterionType.EXACT_ANSWER, accepted=("Colombo",)), student
    )
    assert result.criterion_results[0].decision is CriterionDecision.NORMALIZED_MATCH


def test_normalization_does_not_change_original_evidence() -> None:
    student = "  orderd.  "
    result = _mark(
        _criterion(CriterionType.EXACT_ANSWER, accepted=("orderd",)), student
    )
    assert result.criterion_results[0].student_evidence is not None
    assert result.criterion_results[0].student_evidence.original_text == student
    assert normalize_for_marking(student) == "orderd"


def test_accepted_variant_match_preserves_variant() -> None:
    result = _mark(
        _criterion(
            CriterionType.ACCEPTED_VARIANTS,
            accepted=("colour", "color"),
        ),
        "Color",
    )
    criterion = result.criterion_results[0]
    assert criterion.awarded_marks == 1
    assert criterion.matched_value == "color"


def test_accepted_variant_mismatch_is_zero() -> None:
    result = _mark(
        _criterion(CriterionType.ACCEPTED_VARIANTS, accepted=("colour", "color")),
        "shade",
    )
    assert result.awarded_marks == 0


def test_token_aware_keyword_match() -> None:
    result = _mark(_criterion(CriterionType.KEYWORD, keywords=("in",)), "come in now")
    assert result.awarded_marks == 1


def test_keyword_does_not_match_substring() -> None:
    assert not contains_keyword_or_phrase("interesting", "in")
    result = _mark(_criterion(CriterionType.KEYWORD, keywords=("in",)), "interesting")
    assert result.awarded_marks == 0


def test_phrase_keyword_match() -> None:
    result = _mark(
        _criterion(CriterionType.KEYWORD, keywords=("Indian Ocean",)),
        "It is in the Indian Ocean.",
    )
    assert result.awarded_marks == 1


def test_multiple_keywords_require_all_without_allocation() -> None:
    result = _mark(
        _criterion(
            CriterionType.KEYWORD, maximum=Decimal("2"), keywords=("river", "sea")
        ),
        "river only",
    )
    criterion = result.criterion_results[0]
    assert criterion.awarded_marks == 0
    assert criterion.matched_keywords == ("river",)
    assert criterion.missing_keywords == ("sea",)


def test_keyword_partial_credit_requires_explicit_component_allocations() -> None:
    components = (
        CriterionComponent(
            component_id="component-01",
            keywords=("river",),
            maximum_marks=Decimal("1"),
        ),
        CriterionComponent(
            component_id="component-02",
            keywords=("sea",),
            maximum_marks=Decimal("1"),
        ),
    )
    criterion = _criterion(
        CriterionType.KEYWORD,
        maximum=Decimal("2"),
        keywords=("river", "sea"),
        components=components,
        partial=True,
    )
    result = _mark(criterion, "river only")
    assert result.awarded_marks == 1


def test_multi_part_full_marks() -> None:
    criterion = _criterion(
        CriterionType.MULTI_PART,
        maximum=Decimal("3"),
        components=_components(),
        partial=True,
    )
    result = _mark(criterion, "A B C")
    assert result.awarded_marks == 3


def test_multi_part_partial_marks() -> None:
    criterion = _criterion(
        CriterionType.MULTI_PART,
        maximum=Decimal("3"),
        components=_components(),
        partial=True,
    )
    result = _mark(criterion, "A B wrong")
    assert result.awarded_marks == 2


def test_multi_part_without_allocation_does_not_invent_partial_credit() -> None:
    result = _mark(_criterion(CriterionType.MULTI_PART, maximum=Decimal("3")), "A B C")
    criterion = result.criterion_results[0]
    assert criterion.awarded_marks is None
    assert criterion.status is MarkingStatus.SCHEME_INCOMPLETE


def test_empty_objective_answer_is_legitimate_zero() -> None:
    result = _mark(
        _criterion(CriterionType.EXACT_ANSWER, accepted=("yes",)),
        "",
        answer_status=AnswerExtractionStatus.EMPTY,
    )
    criterion = result.criterion_results[0]
    assert criterion.awarded_marks == 0
    assert criterion.decision is CriterionDecision.EMPTY_ANSWER
    assert result.status is MarkingStatus.NOT_ATTEMPTED


def test_review_required_has_no_final_mark() -> None:
    aligned = _aligned(
        (_criterion(CriterionType.EXACT_ANSWER, accepted=("yes",)),),
        answer=_answer("yes", AnswerExtractionStatus.REVIEW_REQUIRED),
        status=AlignmentStatus.STUDENT_REVIEW_REQUIRED,
    )
    result = DeterministicMarkingEngine().mark_test(aligned)
    assert result.criterion_results[0].awarded_marks is None
    assert result.criterion_results[0].comparison_would_match is True
    assert result.review_required_marks == 1


def test_ocr_failure_is_not_a_zero_mark() -> None:
    aligned = _aligned(
        (_criterion(CriterionType.EXACT_ANSWER, accepted=("yes",)),),
        answer=_answer("", AnswerExtractionStatus.OCR_FAILED),
        status=AlignmentStatus.STUDENT_OCR_FAILED,
    )
    result = DeterministicMarkingEngine().mark_test(aligned)
    assert result.status is MarkingStatus.OCR_FAILED
    assert result.criterion_results[0].awarded_marks is None


@pytest.mark.parametrize(
    ("answer_status", "expected_status"),
    [
        (AnswerExtractionStatus.OCR_FAILED, MarkingStatus.OCR_FAILED),
        (
            AnswerExtractionStatus.STRUCTURE_INCOMPLETE,
            MarkingStatus.STRUCTURE_INCOMPLETE,
        ),
    ],
)
def test_answer_failure_status_blocks_marking_when_alignment_is_stale(
    answer_status: AnswerExtractionStatus,
    expected_status: MarkingStatus,
) -> None:
    aligned = _aligned(
        (_criterion(CriterionType.EXACT_ANSWER, accepted=("yes",)),),
        answer=_answer("yes", answer_status),
    )

    result = DeterministicMarkingEngine().mark_test(aligned)

    assert result.status is expected_status
    assert result.criterion_results[0].awarded_marks is None
    assert result.review_required_marks == 1


@pytest.mark.parametrize(
    "criterion_type",
    [
        CriterionType.CONTENT_POINT,
        CriterionType.LANGUAGE_QUALITY,
        CriterionType.FORMAT_REQUIREMENT,
        CriterionType.HOLISTIC,
    ],
)
def test_subjective_criteria_are_unsupported(criterion_type: CriterionType) -> None:
    result = _mark(_criterion(criterion_type), "synthetic response")
    assert result.criterion_results[0].status is MarkingStatus.UNSUPPORTED_CRITERION
    assert result.criterion_results[0].awarded_marks is None


def test_unknown_criterion_is_unsupported() -> None:
    result = _mark(_criterion(CriterionType.UNKNOWN), "synthetic response")
    criterion = result.criterion_results[0]
    assert criterion.status is MarkingStatus.UNSUPPORTED_CRITERION
    assert MarkingReasonCode.UNKNOWN_CRITERION in criterion.reason_codes


def test_missing_mark_allocation_requires_review() -> None:
    result = _mark(
        _criterion(CriterionType.EXACT_ANSWER, maximum=None, accepted=("yes",)),
        "yes",
    )
    assert result.criterion_results[0].status is MarkingStatus.SCHEME_INCOMPLETE


def test_criterion_component_award_is_capped() -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        from app.marking.models import CriterionComponentResult

        CriterionComponentResult(
            component_id="component-01",
            awarded_marks=Decimal("2"),
            maximum_marks=Decimal("1"),
            matched=True,
        )


def test_test_mark_cap_is_enforced() -> None:
    criteria = (
        _criterion(CriterionType.EXACT_ANSWER, accepted=("yes",), index=1),
        _criterion(CriterionType.EXACT_ANSWER, accepted=("yes",), index=2),
    )
    aligned = _aligned(criteria, answer=_answer("yes"), test_maximum=Decimal("1"))
    with pytest.raises(MarkingEngineError, match="exceed Test maximum"):
        DeterministicMarkingEngine().mark_test(aligned)


def test_paper_mark_cap_model_validation() -> None:
    result = DeterministicMarkingEngine().mark_paper(
        _paper(
            _aligned(
                (_criterion(CriterionType.EXACT_ANSWER, accepted=("yes",)),),
                answer=_answer("yes"),
            )
        )
    )
    payload = result.model_dump()
    payload["maximum_known_marks"] = Decimal("0")
    with pytest.raises(ValidationError, match="cannot exceed"):
        PaperMarkingResult.model_validate(payload)


def test_repeated_evaluation_is_semantically_deterministic() -> None:
    aligned = _paper(
        _aligned(
            (_criterion(CriterionType.EXACT_ANSWER, accepted=("yes",)),),
            answer=_answer("yes"),
        )
    )
    engine = DeterministicMarkingEngine()
    first = engine.mark_paper(aligned).model_dump(exclude={"processing_duration_ms"})
    second = engine.mark_paper(aligned).model_dump(exclude={"processing_duration_ms"})
    for payload in (first, second):
        for test in payload["test_results"]:
            test.pop("processing_duration_ms")
    assert first == second


def test_serialization_is_deterministic_and_private(tmp_path: Path) -> None:
    result = DeterministicMarkingEngine().mark_paper(
        _paper(
            _aligned(
                (_criterion(CriterionType.EXACT_ANSWER, accepted=("yes",)),),
                answer=_answer("yes"),
            )
        )
    )
    private = tmp_path / "private"
    assert marking_result_json(result) == marking_result_json(result)
    assert write_private_marking_result(
        result, private / "result.json", private_root=private
    ).is_file()
    with pytest.raises(MarkingEngineError, match="remain private"):
        write_private_marking_result(
            result, tmp_path / "public.json", private_root=private
        )


def test_reason_codes_and_provenance_are_preserved() -> None:
    result = _mark(_criterion(CriterionType.EXACT_ANSWER, accepted=("yes",)), "no")
    criterion = result.criterion_results[0]
    assert criterion.reason_codes == (MarkingReasonCode.NO_DETERMINISTIC_MATCH,)
    assert criterion.provenance.criterion_id == criterion.criterion_id
    assert criterion.student_evidence is not None
    assert criterion.student_evidence.paper_id == PAPER_ID


def test_missing_student_test_is_structure_incomplete() -> None:
    result = DeterministicMarkingEngine().mark_test(
        _aligned(
            (_criterion(CriterionType.EXACT_ANSWER, accepted=("yes",)),),
            answer=None,
            status=AlignmentStatus.STUDENT_TEST_MISSING,
        )
    )
    assert result.status is MarkingStatus.STRUCTURE_INCOMPLETE
    assert result.criterion_results[0].awarded_marks is None


def test_missing_scheme_test_is_explicit() -> None:
    aligned = AlignedTest(
        test_number=1,
        status=AlignmentStatus.SCHEME_TEST_MISSING,
        student_answer=_answer("yes"),
    )
    result = DeterministicMarkingEngine().mark_test(aligned)
    assert result.status is MarkingStatus.SCHEME_INCOMPLETE
    assert MarkingWarning.SCHEME_TEST_MISSING in result.warnings


def test_partially_marked_test_separates_unresolved_marks() -> None:
    criteria = (
        _criterion(CriterionType.EXACT_ANSWER, accepted=("yes",), index=1),
        _criterion(CriterionType.HOLISTIC, maximum=Decimal("2"), index=2),
    )
    result = DeterministicMarkingEngine().mark_test(
        _aligned(criteria, answer=_answer("yes"))
    )
    assert result.status is MarkingStatus.PARTIALLY_MARKED
    assert result.awarded_marks == 1
    assert result.unsupported_marks == 2


def test_partially_marked_paper_preserves_confirmed_and_unresolved() -> None:
    criteria = (
        _criterion(CriterionType.EXACT_ANSWER, accepted=("yes",), index=1),
        _criterion(CriterionType.HOLISTIC, maximum=Decimal("2"), index=2),
    )
    result = DeterministicMarkingEngine().mark_paper(
        _paper(_aligned(criteria, answer=_answer("yes")))
    )
    assert result.confirmed_marks == 1
    assert result.maximum_known_marks == 3
    assert result.unsupported_marks == 2
    assert result.tests_unsupported == (1,)
    assert MarkingWarning.PAPER_PARTIALLY_MARKED in result.warnings


def test_paper_maximum_includes_explicit_criteria_when_test_total_is_unknown() -> None:
    criteria = (
        _criterion(CriterionType.EXACT_ANSWER, accepted=("yes",), index=1),
        _criterion(CriterionType.UNKNOWN, maximum=None, index=2),
    )

    result = DeterministicMarkingEngine().mark_paper(
        _paper(
            _aligned(
                criteria,
                answer=_answer("", AnswerExtractionStatus.REVIEW_REQUIRED),
            )
        )
    )

    assert result.maximum_known_marks == 1
    assert result.review_required_marks == 1
    assert result.confirmed_marks == 0


def test_unallocated_test_marks_remain_review_required() -> None:
    criterion = _criterion(CriterionType.EXACT_ANSWER, accepted=("yes",))
    result = DeterministicMarkingEngine().mark_test(
        _aligned((criterion,), answer=_answer("yes"), test_maximum=Decimal("2"))
    )
    assert result.status is MarkingStatus.PARTIALLY_MARKED
    assert result.awarded_marks == 1
    assert result.review_required_marks == 1


def test_explicit_components_must_equal_criterion_maximum() -> None:
    with pytest.raises(ValidationError, match="equal criterion maximum"):
        _criterion(
            CriterionType.MULTI_PART,
            maximum=Decimal("4"),
            components=_components(),
        )


def test_component_ids_are_ordered() -> None:
    component = CriterionComponent(
        component_id="component-02",
        accepted_answers=("A",),
        maximum_marks=Decimal("1"),
    )
    with pytest.raises(ValidationError, match="deterministic and ordered"):
        _criterion(
            CriterionType.MULTI_PART,
            maximum=Decimal("1"),
            components=(component,),
        )


def test_source_fixture_uses_no_private_paths() -> None:
    assert Path("synthetic").name == "synthetic"
    assert uuid4() != PAPER_ID

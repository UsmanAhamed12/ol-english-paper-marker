"""Run a privacy-safe synthetic smoke of the deterministic marking engine."""

from __future__ import annotations

import json
from collections.abc import Sequence
from decimal import Decimal
from uuid import UUID

from app.answers.models import (
    AnswerExtractionProvenance,
    AnswerExtractionStatus,
    AnswerFragment,
    AnswerFragmentStatus,
    AnswerSourceContext,
    StructuredStudentAnswer,
)
from app.marking.engine import DeterministicMarkingEngine
from app.marking.models import MarkingStatus
from app.marking_scheme.models import (
    AlignedPaper,
    AlignedTest,
    AlignmentStatus,
    CriterionParserRule,
    CriterionProvenance,
    CriterionType,
    MarkingCriterion,
    MarkingSchemeTest,
    SchemeSourceRegion,
)
from app.ocr.models import BoundingBox
from app.structure.models import MarkerDetectionStrategy

PAPER_ID = UUID("50000000-0000-0000-0000-000000000005")
SCHEME_ID = UUID("60000000-0000-0000-0000-000000000006")


def build_synthetic_alignment() -> AlignedPaper:
    """Build non-private fixtures covering automatic and fallback paths."""

    specifications = (
        (CriterionType.EXACT_ANSWER, "Colombo", ("Colombo",), (), Decimal("1")),
        (
            CriterionType.ACCEPTED_VARIANTS,
            "color",
            ("colour", "color"),
            (),
            Decimal("1"),
        ),
        (
            CriterionType.KEYWORD,
            "river only",
            (),
            ("river", "sea"),
            Decimal("2"),
        ),
        (CriterionType.EXACT_ANSWER, "", ("yes",), (), Decimal("1")),
        (CriterionType.HOLISTIC, "synthetic paragraph", (), (), Decimal("5")),
        (CriterionType.EXACT_ANSWER, "uncertain", ("yes",), (), Decimal("1")),
    )
    tests: list[AlignedTest] = []
    for test_number, (kind, text, answers, keywords, maximum) in enumerate(
        specifications, start=1
    ):
        answer_status = (
            AnswerExtractionStatus.EMPTY
            if test_number == 4
            else AnswerExtractionStatus.REVIEW_REQUIRED
            if test_number == 6
            else AnswerExtractionStatus.EXTRACTED
        )
        alignment_status = (
            AlignmentStatus.STUDENT_REVIEW_REQUIRED
            if answer_status is AnswerExtractionStatus.REVIEW_REQUIRED
            else AlignmentStatus.ALIGNED
        )
        criterion = _criterion(
            test_number,
            kind,
            answers=answers,
            keywords=keywords,
            maximum=maximum,
        )
        tests.append(
            AlignedTest(
                test_number=test_number,
                status=alignment_status,
                student_answer=_answer(test_number, text, answer_status),
                marking_scheme_test=MarkingSchemeTest(
                    test_number=test_number,
                    maximum_marks=maximum,
                    criteria=(criterion,),
                    source_pages=(1,),
                    source_regions=(
                        SchemeSourceRegion(
                            page_number=1,
                            raw_source_text="synthetic marking fixture",
                        ),
                    ),
                ),
            )
        )
    return AlignedPaper(
        paper_id=PAPER_ID,
        scheme_id=SCHEME_ID,
        tests=tuple(tests),
        aligned_test_numbers=(1, 2, 3, 4, 5),
        review_required_test_numbers=(6,),
    )


def synthetic_summary() -> dict[str, int | str]:
    """Return safe operational counts without answer content."""

    result = DeterministicMarkingEngine().mark_paper(build_synthetic_alignment())
    criteria = tuple(
        criterion
        for test in result.test_results
        for criterion in test.criterion_results
    )
    return {
        "tests_evaluated": len(result.test_results),
        "criteria_evaluated": len(criteria),
        "automatically_marked": sum(
            item.status in {MarkingStatus.MARKED, MarkingStatus.NOT_ATTEMPTED}
            for item in criteria
        ),
        "review_required": sum(
            item.status is MarkingStatus.REVIEW_REQUIRED for item in criteria
        ),
        "unsupported": sum(
            item.status is MarkingStatus.UNSUPPORTED_CRITERION for item in criteria
        ),
        "confirmed_marks": str(result.confirmed_marks),
        "unresolved_marks": str(
            result.review_required_marks + result.unsupported_marks
        ),
        "known_maximum": str(result.maximum_known_marks),
    }


def _criterion(
    test_number: int,
    kind: CriterionType,
    *,
    answers: tuple[str, ...],
    keywords: tuple[str, ...],
    maximum: Decimal,
) -> MarkingCriterion:
    return MarkingCriterion(
        criterion_id=f"test-{test_number:02d}-criterion-001",
        test_number=test_number,
        criterion_type=kind,
        description="Synthetic marking criterion",
        accepted_answers=answers,
        keywords=keywords,
        maximum_marks=maximum,
        provenance=CriterionProvenance(
            parser_rule=CriterionParserRule.PRESERVED_UNKNOWN,
            source=SchemeSourceRegion(
                page_number=1,
                raw_source_text="synthetic marking fixture",
            ),
        ),
    )


def _answer(
    test_number: int,
    text: str,
    status: AnswerExtractionStatus,
) -> StructuredStudentAnswer:
    fragments: tuple[AnswerFragment, ...] = ()
    if status is AnswerExtractionStatus.EXTRACTED:
        fragments = (
            AnswerFragment(
                fragment_index=0,
                source_region_index=0,
                paper_id=PAPER_ID,
                test_number=test_number,
                page_number=1,
                bbox=BoundingBox(x=10, y=10, width=100, height=20),
                status=AnswerFragmentStatus.EXTRACTED,
                raw_text=text,
                normalized_text=text,
                provider="synthetic",
                processing_duration_ms=0,
            ),
        )
    return StructuredStudentAnswer(
        paper_id=PAPER_ID,
        test_number=test_number,
        status=status,
        text=text,
        pages=(1,),
        ocr_fragments=fragments,
        provenance=AnswerExtractionProvenance(
            extraction_version="synthetic-v1",
            test_label=f"Test {test_number:02d}",
            marker_page_number=1,
            marker_bbox=BoundingBox(x=0, y=0, width=20, height=10),
            marker_strategy=MarkerDetectionStrategy.EXACT_TOKENS,
            marker_confidence=1,
            marker_source_word_indices=(0,),
            safety_policy_versions=("synthetic-safety-v1",),
            ocr_provider="synthetic",
            source_context=AnswerSourceContext.RUNTIME_UNMARKED,
        ),
        processing_duration_ms=0,
    )


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise SystemExit("This synthetic smoke accepts no arguments")
    print(json.dumps(synthetic_summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

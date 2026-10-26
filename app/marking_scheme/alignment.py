"""Exact Test-number alignment between student answers and marking criteria."""

from __future__ import annotations

from app.answers.models import (
    AnswerExtractionStatus,
    StructuredStudentAnswer,
    StructuredStudentPaper,
)
from app.marking_scheme.models import (
    AlignedPaper,
    AlignedTest,
    AlignmentStatus,
    AlignmentWarning,
    MarkingSchemeTest,
    StructuredMarkingScheme,
)


class TestAlignmentService:
    """Align explicit Tests without semantic guessing or invented records."""

    def align(
        self,
        student: StructuredStudentPaper,
        scheme: StructuredMarkingScheme,
    ) -> AlignedPaper:
        student_by_number = {item.test_number: item for item in student.answers}
        scheme_by_number = {item.test_number: item for item in scheme.tests}
        numbers = tuple(sorted(student_by_number.keys() | scheme_by_number.keys()))
        tests = tuple(
            _align_test(
                number,
                student_by_number.get(number),
                scheme_by_number.get(number),
            )
            for number in numbers
        )
        return AlignedPaper(
            paper_id=student.paper_id,
            scheme_id=scheme.scheme_id,
            tests=tests,
            aligned_test_numbers=_numbers(tests, AlignmentStatus.ALIGNED),
            student_missing_test_numbers=_numbers(
                tests, AlignmentStatus.STUDENT_TEST_MISSING
            ),
            scheme_missing_test_numbers=_numbers(
                tests, AlignmentStatus.SCHEME_TEST_MISSING
            ),
            review_required_test_numbers=_numbers(
                tests, AlignmentStatus.STUDENT_REVIEW_REQUIRED
            ),
            ocr_failed_test_numbers=_numbers(tests, AlignmentStatus.STUDENT_OCR_FAILED),
            structure_incomplete_test_numbers=_numbers(
                tests, AlignmentStatus.STRUCTURE_INCOMPLETE
            ),
        )


def _align_test(
    number: int,
    student: StructuredStudentAnswer | None,
    scheme: MarkingSchemeTest | None,
) -> AlignedTest:
    answer = student
    warnings: list[AlignmentWarning] = []
    if answer is None:
        status = AlignmentStatus.STUDENT_TEST_MISSING
        warnings.append(AlignmentWarning.STUDENT_TEST_ABSENT)
    elif scheme is None:
        status = AlignmentStatus.SCHEME_TEST_MISSING
        warnings.append(AlignmentWarning.SCHEME_TEST_ABSENT)
    elif answer.status is AnswerExtractionStatus.REVIEW_REQUIRED:
        status = AlignmentStatus.STUDENT_REVIEW_REQUIRED
        warnings.append(AlignmentWarning.STUDENT_REVIEW_REQUIRED)
    elif answer.status is AnswerExtractionStatus.OCR_FAILED:
        status = AlignmentStatus.STUDENT_OCR_FAILED
        warnings.append(AlignmentWarning.STUDENT_OCR_FAILED)
    elif answer.status is AnswerExtractionStatus.STRUCTURE_INCOMPLETE:
        status = AlignmentStatus.STRUCTURE_INCOMPLETE
        warnings.append(AlignmentWarning.STUDENT_STRUCTURE_INCOMPLETE)
    else:
        status = AlignmentStatus.ALIGNED
    if scheme is not None and scheme.maximum_marks is None:
        warnings.append(AlignmentWarning.SCHEME_MAXIMUM_UNKNOWN)
    return AlignedTest(
        test_number=number,
        status=status,
        student_answer=answer,
        marking_scheme_test=scheme,
        warnings=tuple(warnings),
    )


def _numbers(
    tests: tuple[AlignedTest, ...], status: AlignmentStatus
) -> tuple[int, ...]:
    return tuple(item.test_number for item in tests if item.status is status)

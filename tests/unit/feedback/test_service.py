"""Tests for rubric-bound local formative feedback."""

from __future__ import annotations

import json
from uuid import UUID

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
from app.core.exceptions import SubjectiveMarkingProviderError
from app.feedback.models import QuestionFeedbackStatus
from app.feedback.service import (
    QUESTION_FEEDBACK_PROMPT_VERSION,
    PaperFeedbackService,
)
from app.marking.subjective.provider import SubjectiveProviderResponse
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

PAPER_ID = UUID("a1000000-0000-0000-0000-000000000001")
SCHEME_ID = UUID("a2000000-0000-0000-0000-000000000002")


class FakeFeedbackProvider:
    name = "fake-ollama"
    model_version = "fake-llama2"

    def __init__(self, *, raw_response: str | None = None, fail: bool = False) -> None:
        self.raw_response = raw_response
        self.fail = fail
        self.calls = 0
        self.prompt = ""

    def assess(self, prompt: str) -> SubjectiveProviderResponse:
        self.calls += 1
        self.prompt = prompt
        if self.fail:
            raise SubjectiveMarkingProviderError("synthetic unavailable")
        question_data = prompt.split("QUESTION_DATA_JSON:\n", 1)[1].split(
            "\n\nPrompt version:", 1
        )[0]
        questions = json.loads(question_data)
        raw_response = self.raw_response or json.dumps(
            {
                "questions": [
                    {
                        "test_number": question["test_number"],
                        "feedback": f"Feedback for Test {question['test_number']}.",
                        "improvement_tips": ["Use a detail supported by the scheme."],
                    }
                    for question in questions
                ]
            }
        )
        return SubjectiveProviderResponse(
            raw_text=raw_response,
            provider=self.name,
            model_version=self.model_version,
            processing_duration_ms=12,
        )


def _answer(
    test_number: int,
    *,
    status: AnswerExtractionStatus = AnswerExtractionStatus.EXTRACTED,
    text: str = "Synthetic answer evidence.",
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
                bbox=BoundingBox(x=0, y=0, width=100, height=20),
                status=AnswerFragmentStatus.EXTRACTED,
                raw_text=text,
                normalized_text=text,
                confidence=0.95,
                provider="synthetic-ocr",
                model_version="synthetic-ocr-v1",
                processing_duration_ms=1,
            ),
        )
    return StructuredStudentAnswer(
        paper_id=PAPER_ID,
        test_number=test_number,
        status=status,
        text=text,
        pages=(1,),
        ocr_fragments=fragments,
        ocr_confidence=0.95 if status is AnswerExtractionStatus.EXTRACTED else None,
        provenance=AnswerExtractionProvenance(
            extraction_version="synthetic-extraction-v1",
            test_label=f"Test {test_number:02d}",
            marker_page_number=1,
            marker_bbox=BoundingBox(x=0, y=0, width=100, height=20),
            marker_strategy=MarkerDetectionStrategy.EXACT_TOKENS,
            marker_confidence=0.95,
            marker_source_word_indices=(0,),
            safety_policy_versions=("synthetic-safety-v1",),
            ocr_provider="synthetic-ocr",
            source_context=AnswerSourceContext.RUNTIME_UNMARKED,
        ),
        processing_duration_ms=1,
    )


def _criterion(test_number: int, kind: CriterionType) -> MarkingCriterion:
    return MarkingCriterion(
        criterion_id=f"test-{test_number:02d}-criterion-001",
        test_number=test_number,
        criterion_type=kind,
        description="Use a clear topic sentence and one relevant supporting detail.",
        maximum_marks=2,
        provenance=CriterionProvenance(
            parser_rule=CriterionParserRule.CONTENT_LABEL,
            source=SchemeSourceRegion(
                page_number=1, raw_source_text="Synthetic rubric"
            ),
        ),
    )


def _scheme_test(test_number: int, *criteria: MarkingCriterion) -> MarkingSchemeTest:
    return MarkingSchemeTest(
        test_number=test_number,
        title=f"Synthetic Test {test_number}",
        maximum_marks=2,
        criteria=criteria,
        source_pages=(1,),
        source_regions=(
            SchemeSourceRegion(page_number=1, raw_source_text="Synthetic"),
        ),
    )


def _aligned(*tests: AlignedTest) -> AlignedPaper:
    by_status = {
        status: tuple(item.test_number for item in tests if item.status is status)
        for status in AlignmentStatus
    }
    return AlignedPaper(
        paper_id=PAPER_ID,
        scheme_id=SCHEME_ID,
        tests=tuple(tests),
        aligned_test_numbers=by_status[AlignmentStatus.ALIGNED],
        student_missing_test_numbers=by_status[AlignmentStatus.STUDENT_TEST_MISSING],
        scheme_missing_test_numbers=by_status[AlignmentStatus.SCHEME_TEST_MISSING],
        review_required_test_numbers=by_status[AlignmentStatus.STUDENT_REVIEW_REQUIRED],
        ocr_failed_test_numbers=by_status[AlignmentStatus.STUDENT_OCR_FAILED],
        structure_incomplete_test_numbers=by_status[
            AlignmentStatus.STRUCTURE_INCOMPLETE
        ],
    )


def test_generates_feedback_and_marks_uncertain_answers_provisional() -> None:
    provider = FakeFeedbackProvider()
    service = PaperFeedbackService(provider)
    aligned = _aligned(
        AlignedTest(
            test_number=1,
            status=AlignmentStatus.ALIGNED,
            student_answer=_answer(1),
            marking_scheme_test=_scheme_test(
                1, _criterion(1, CriterionType.CONTENT_POINT)
            ),
        ),
        AlignedTest(
            test_number=2,
            status=AlignmentStatus.STUDENT_REVIEW_REQUIRED,
            student_answer=_answer(
                2,
                status=AnswerExtractionStatus.REVIEW_REQUIRED,
            ),
            marking_scheme_test=_scheme_test(
                2, _criterion(2, CriterionType.EXACT_ANSWER)
            ),
        ),
    )

    results = service.generate(aligned)

    assert provider.calls == 1
    assert [item.test_number for item in results] == [1, 2]
    assert results[0].status is QuestionFeedbackStatus.GENERATED
    assert results[0].feedback == "Feedback for Test 1."
    assert results[0].improvement_tips == ("Use a detail supported by the scheme.",)
    assert results[0].prompt_version == QUESTION_FEEDBACK_PROMPT_VERSION
    assert results[1].status is QuestionFeedbackStatus.PROVISIONAL
    assert results[1].answer_status is AnswerExtractionStatus.REVIEW_REQUIRED
    assert "Do not award, estimate, or mention marks" in provider.prompt
    assert "Never ask" in provider.prompt
    assert "extra content beyond a stated requirement" in provider.prompt
    assert "Synthetic answer evidence." in provider.prompt


def test_skips_missing_or_unrecognized_evidence_without_model_call() -> None:
    provider = FakeFeedbackProvider()
    aligned = _aligned(
        AlignedTest(
            test_number=1,
            status=AlignmentStatus.STUDENT_TEST_MISSING,
            marking_scheme_test=_scheme_test(
                1, _criterion(1, CriterionType.CONTENT_POINT)
            ),
        ),
        AlignedTest(
            test_number=2,
            status=AlignmentStatus.ALIGNED,
            student_answer=_answer(2),
            marking_scheme_test=_scheme_test(2, _criterion(2, CriterionType.UNKNOWN)),
        ),
    )

    results = PaperFeedbackService(provider).generate(aligned)

    assert provider.calls == 0
    assert [item.status for item in results] == [
        QuestionFeedbackStatus.SKIPPED,
        QuestionFeedbackStatus.SKIPPED,
    ]
    assert [item.reason_code for item in results] == [
        "student_answer_missing",
        "no_recognized_scheme_criteria",
    ]


@pytest.mark.parametrize(
    ("raw_response", "expected_reason"),
    [
        ("not-json", "invalid_local_feedback_response"),
        (
            json.dumps(
                {
                    "questions": [
                        {
                            "test_number": 99,
                            "feedback": "Wrong ownership.",
                            "improvement_tips": [],
                        }
                    ]
                }
            ),
            "feedback_question_coverage_mismatch",
        ),
    ],
)
def test_rejects_malformed_or_wrong_test_feedback(
    raw_response: str,
    expected_reason: str,
) -> None:
    provider = FakeFeedbackProvider(raw_response=raw_response)
    aligned = _aligned(
        AlignedTest(
            test_number=1,
            status=AlignmentStatus.ALIGNED,
            student_answer=_answer(1),
            marking_scheme_test=_scheme_test(
                1, _criterion(1, CriterionType.CONTENT_POINT)
            ),
        )
    )

    result = PaperFeedbackService(provider).generate(aligned)[0]

    assert result.status is QuestionFeedbackStatus.FAILED
    assert result.reason_code == expected_reason
    assert result.feedback is None


def test_provider_failure_is_reported_without_marking_fallback() -> None:
    provider = FakeFeedbackProvider(fail=True)
    aligned = _aligned(
        AlignedTest(
            test_number=1,
            status=AlignmentStatus.ALIGNED,
            student_answer=_answer(1),
            marking_scheme_test=_scheme_test(
                1, _criterion(1, CriterionType.CONTENT_POINT)
            ),
        )
    )

    result = PaperFeedbackService(provider).generate(aligned)[0]

    assert result.status is QuestionFeedbackStatus.FAILED
    assert result.reason_code == "local_ollama_unavailable"
    assert result.feedback is None


def test_feedback_contract_rejects_mark_fields() -> None:
    with pytest.raises(ValidationError, match="awarded_marks"):
        from app.feedback.models import QuestionFeedback

        QuestionFeedback.model_validate(
            {
                "test_number": 1,
                "status": "generated",
                "feedback": "Synthetic advice.",
                "improvement_tips": [],
                "provider": "ollama",
                "model_version": "llama2",
                "prompt_version": "v1",
                "awarded_marks": "10",
            }
        )

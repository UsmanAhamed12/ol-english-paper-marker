"""Deterministic end-to-end tests for the Phase 6 application boundary."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

from app.answers.models import (
    AnswerExtractionProvenance,
    AnswerExtractionStatus,
    AnswerFragment,
    AnswerFragmentStatus,
    AnswerSourceContext,
    PaperExtractionProvenance,
    StructuredStudentAnswer,
    StructuredStudentPaper,
)
from app.core.exceptions import (
    AnswerExtractionError,
    MarkingSchemeIngestionError,
    OrchestrationError,
)
from app.marking.subjective.combined import CombinedMarkingService
from app.marking.subjective.models import SubjectiveMarkingStatus
from app.marking.subjective.provider import SubjectiveProviderResponse
from app.marking.subjective.service import SubjectiveMarkingService
from app.marking_scheme.alignment import TestAlignmentService
from app.marking_scheme.models import (
    CriterionComponent,
    CriterionParserRule,
    CriterionProvenance,
    CriterionType,
    MarkingCriterion,
    MarkingSchemeOCRStatus,
    MarkingSchemePageOCR,
    MarkingSchemeProvenance,
    MarkingSchemeTest,
    SchemeSourceRegion,
    StructuredMarkingScheme,
)
from app.ocr.models import BoundingBox
from app.orchestration.interfaces import SchemePipelineOutput, StudentPipelineOutput
from app.orchestration.models import (
    FinalPaperStatus,
    OrchestrationErrorCode,
    PaperMarkingRequest,
    PipelineStage,
    StageStatus,
    StageSummary,
)
from app.orchestration.serialization import (
    final_result_json,
    write_private_final_result,
)
from app.orchestration.service import PaperMarkingService
from app.structure.models import MarkerDetectionStrategy

PAPER_ID = UUID("90000000-0000-0000-0000-000000000009")
SCHEME_ID = UUID("a0000000-0000-0000-0000-00000000000a")
SOURCE_ID = UUID("b0000000-0000-0000-0000-00000000000b")


class FakeValidator:
    def __init__(self, invalid_name: str | None = None) -> None:
        self.invalid_name = invalid_name
        self.paths: list[Path] = []

    def validate(self, path: Path) -> None:
        self.paths.append(path)
        if path.name == self.invalid_name:
            raise ValueError("synthetic invalid input")


class FakeStudentPipeline:
    def __init__(
        self,
        paper: StructuredStudentPaper,
        error: Exception | None = None,
    ) -> None:
        self.paper = paper
        self.error = error
        self.calls = 0

    def process(self, path: Path, *, run_id: UUID) -> StudentPipelineOutput:
        del path, run_id
        self.calls += 1
        if self.error is not None:
            raise self.error
        return StudentPipelineOutput(
            paper=self.paper,
            stages=(
                _stage(PipelineStage.STUDENT_PDF_PROCESSING),
                _stage(PipelineStage.STRUCTURE_DETECTION),
                _stage(PipelineStage.STUDENT_ANSWER_EXTRACTION),
            ),
        )


class FakeSchemePipeline:
    def __init__(
        self,
        scheme: StructuredMarkingScheme,
        error: Exception | None = None,
    ) -> None:
        self.scheme = scheme
        self.error = error
        self.calls = 0

    def process(self, path: Path, *, run_id: UUID) -> SchemePipelineOutput:
        del path, run_id
        self.calls += 1
        if self.error is not None:
            raise self.error
        return SchemePipelineOutput(
            scheme=self.scheme,
            stages=(_stage(PipelineStage.MARKING_SCHEME_INGESTION),),
        )


class FakeSubjectiveProvider:
    name = "fake-subjective"
    model_version = "fake-v1"

    def __init__(self, *, malformed: bool = False, fail: bool = False) -> None:
        self.malformed = malformed
        self.fail = fail
        self.calls = 0

    def assess(self, prompt: str) -> SubjectiveProviderResponse:
        from app.core.exceptions import SubjectiveMarkingProviderError

        self.calls += 1
        if self.fail:
            raise SubjectiveMarkingProviderError("safe synthetic failure")
        if self.malformed:
            raw = "not-json"
        else:
            line = prompt.split("AUTHORITATIVE_CRITERION_JSON:\n", 1)[1]
            criterion = json.loads(line.split("\n", 1)[0])
            kind = criterion["criterion_type"]
            award = "2" if kind == "content_point" else "3"
            raw = json.dumps(
                {
                    "test_number": criterion["test_number"],
                    "criterion_id": criterion["criterion_id"],
                    "criterion_type": kind,
                    "maximum_marks": criterion["maximum_marks"],
                    "awarded_marks": award,
                    "status": "marked",
                    "matched_requirements": ["synthetic requirement"],
                    "missing_requirements": [],
                    "reason_codes": ["synthetic_rubric_match"],
                    "feedback": "Synthetic criterion feedback.",
                    "confidence": 0.95,
                }
            )
        return SubjectiveProviderResponse(
            raw_text=raw,
            provider=self.name,
            model_version=self.model_version,
            processing_duration_ms=1,
        )


def _stage(stage: PipelineStage) -> StageSummary:
    return StageSummary(stage=stage, status=StageStatus.COMPLETED, duration_ms=1)


def _answer(
    test_number: int,
    text: str,
    status: AnswerExtractionStatus = AnswerExtractionStatus.EXTRACTED,
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
                confidence=0.95,
                provider="fake-ocr",
                model_version="fake-ocr-v1",
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
            marker_bbox=BoundingBox(x=0, y=0, width=20, height=10),
            marker_strategy=MarkerDetectionStrategy.EXACT_TOKENS,
            marker_confidence=1,
            marker_source_word_indices=(0,),
            safety_policy_versions=("synthetic-safety-v1",),
            ocr_provider="fake-ocr",
            ocr_model_version="fake-ocr-v1",
            source_context=AnswerSourceContext.RUNTIME_UNMARKED,
        ),
        processing_duration_ms=1,
    )


def _student_paper(
    answers: tuple[StructuredStudentAnswer, ...],
) -> StructuredStudentPaper:
    return StructuredStudentPaper(
        paper_id=PAPER_ID,
        answers=answers,
        detected_test_numbers=tuple(item.test_number for item in answers),
        review_required_test_numbers=tuple(
            item.test_number
            for item in answers
            if item.status is AnswerExtractionStatus.REVIEW_REQUIRED
        ),
        empty_test_numbers=tuple(
            item.test_number
            for item in answers
            if item.status is AnswerExtractionStatus.EMPTY
        ),
        ocr_failed_test_numbers=tuple(
            item.test_number
            for item in answers
            if item.status is AnswerExtractionStatus.OCR_FAILED
        ),
        structure_incomplete_test_numbers=tuple(
            item.test_number
            for item in answers
            if item.status is AnswerExtractionStatus.STRUCTURE_INCOMPLETE
        ),
        total_processing_duration_ms=5,
        provenance=PaperExtractionProvenance(
            extraction_version="synthetic-extraction-v1",
            structure_test_count=len(answers),
            source_page_count=1,
            ocr_provider="fake-ocr",
            ocr_model_version="fake-ocr-v1",
            source_context=AnswerSourceContext.RUNTIME_UNMARKED,
        ),
    )


def _criterion(
    test_number: int,
    kind: CriterionType,
    maximum: str,
    *,
    accepted: tuple[str, ...] = (),
    components: tuple[CriterionComponent, ...] = (),
) -> MarkingCriterion:
    return MarkingCriterion(
        criterion_id=f"test-{test_number:02d}-criterion-001",
        test_number=test_number,
        criterion_type=kind,
        description="Synthetic authoritative criterion.",
        accepted_answers=accepted,
        components=components,
        maximum_marks=Decimal(maximum),
        partial_credit_allowed=True if components else None,
        provenance=CriterionProvenance(
            parser_rule=CriterionParserRule.PRESERVED_UNKNOWN,
            source=SchemeSourceRegion(page_number=1, raw_source_text="synthetic"),
        ),
    )


def _scheme(criteria: tuple[MarkingCriterion, ...]) -> StructuredMarkingScheme:
    tests = tuple(
        MarkingSchemeTest(
            test_number=item.test_number,
            maximum_marks=item.maximum_marks,
            criteria=(item,),
            source_pages=(1,),
            source_regions=(
                SchemeSourceRegion(page_number=1, raw_source_text="synthetic"),
            ),
        )
        for item in criteria
    )
    total = sum((item.maximum_marks or Decimal("0") for item in tests), Decimal("0"))
    return StructuredMarkingScheme(
        scheme_id=SCHEME_ID,
        source_document_id=SOURCE_ID,
        page_count=1,
        tests=tests,
        total_available_marks=total,
        page_ocr=(
            MarkingSchemePageOCR(
                page_number=1,
                status=MarkingSchemeOCRStatus.EXTRACTED,
                raw_text="synthetic",
                normalized_text="synthetic",
                provider="fake-ocr",
                model_version="fake-ocr-v1",
                duration_ms=1,
            ),
        ),
        provenance=MarkingSchemeProvenance(
            ingestion_version="synthetic-ingestion-v1",
            parser_version="synthetic-parser-v1",
            source_sha256="a" * 64,
            ocr_provider="fake-ocr",
            ocr_model_version="fake-ocr-v1",
            successful_ocr_pages=(1,),
        ),
        processing_duration_ms=2,
    )


def _scenario() -> tuple[StructuredStudentPaper, StructuredMarkingScheme]:
    components = (
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
    )
    criteria = (
        _criterion(1, CriterionType.EXACT_ANSWER, "1", accepted=("yes",)),
        _criterion(2, CriterionType.ACCEPTED_VARIANTS, "1", accepted=("color",)),
        _criterion(3, CriterionType.MULTI_PART, "2", components=components),
        _criterion(4, CriterionType.CONTENT_POINT, "3"),
        _criterion(5, CriterionType.HOLISTIC, "4"),
        _criterion(6, CriterionType.EXACT_ANSWER, "1", accepted=("answer",)),
        _criterion(7, CriterionType.CONTENT_POINT, "2"),
        _criterion(8, CriterionType.UNKNOWN, "1"),
    )
    answers = (
        _answer(1, "yes"),
        _answer(2, "Color"),
        _answer(3, "A B"),
        _answer(4, "Synthetic content answer."),
        _answer(5, "Synthetic paragraph answer."),
        _answer(6, "", AnswerExtractionStatus.EMPTY),
        _answer(7, "uncertain", AnswerExtractionStatus.REVIEW_REQUIRED),
        _answer(8, "unclassified evidence"),
    )
    return _student_paper(answers), _scheme(criteria)


def _service(
    *,
    student: StructuredStudentPaper | None = None,
    scheme: StructuredMarkingScheme | None = None,
    provider: FakeSubjectiveProvider | None = None,
    validator: FakeValidator | None = None,
    student_error: Exception | None = None,
    scheme_error: Exception | None = None,
    private_root: Path = Path("data/runtime"),
) -> tuple[PaperMarkingService, FakeSubjectiveProvider]:
    default_student, default_scheme = _scenario()
    provider = provider or FakeSubjectiveProvider()
    service = PaperMarkingService(
        input_validator=validator or FakeValidator(),
        student_pipeline=FakeStudentPipeline(
            student or default_student, error=student_error
        ),
        scheme_pipeline=FakeSchemePipeline(
            scheme or default_scheme, error=scheme_error
        ),
        alignment_service=TestAlignmentService(),
        combined_marking_service=CombinedMarkingService(
            SubjectiveMarkingService(provider)
        ),
        private_output_root=private_root,
    )
    return service, provider


def _request(output: Path | None = None) -> PaperMarkingRequest:
    return PaperMarkingRequest(
        student_pdf=Path("synthetic-student.pdf"),
        marking_scheme_pdf=Path("synthetic-scheme.pdf"),
        output_path=output,
    )


def test_complete_synthetic_orchestration_preserves_every_boundary() -> None:
    service, provider = _service()
    result = service.mark_paper(_request())
    assert result.status is FinalPaperStatus.COMPLETED_WITH_REVIEW
    assert result.detected_tests == tuple(range(1, 9))
    assert result.aligned_tests == tuple(range(1, 9))
    assert result.objective_confirmed_marks == 4
    assert result.subjective_confirmed_marks == 5
    assert result.total_confirmed_marks == 9
    assert result.known_maximum_marks == 15
    assert result.review_required_marks == 2
    assert result.unsupported_marks == 1
    assert result.unresolved_marks == 3
    assert result.final_score is None
    assert len(result.review_items) == 2
    assert len(result.feedback) == 2
    assert provider.calls == 2
    assert len({item.stage for item in result.stages}) == 9
    assert all(item.duration_ms >= 0 for item in result.stages)
    assert result.total_duration_ms >= 0
    assert result.provenance.orchestration_version == "paper-marking-orchestration-v1"
    components = {item.component for item in result.provenance.components}
    assert {
        "student_extraction",
        "student_ocr",
        "marking_scheme_ingestion",
        "marking_scheme_parser",
        "alignment",
        "subjective_provider",
        "subjective_prompt",
    } <= components


def test_criterion_ownership_prevents_double_counting() -> None:
    result = _service()[0].mark_paper(_request())
    objective_ids = [
        item.criterion_id
        for test in result.test_results
        for item in test.objective_results
    ]
    subjective_ids = [
        item.criterion_id
        for test in result.test_results
        for item in test.subjective_results
    ]
    ids = objective_ids + subjective_ids
    assert len(ids) == len(set(ids)) == 8
    assert sum(len(test.objective_results) for test in result.test_results) == 4
    assert sum(len(test.subjective_results) for test in result.test_results) == 4


def test_fully_resolved_paper_exposes_final_score() -> None:
    student, scheme = _scenario()
    selected_student = _student_paper(student.answers[:5])
    selected_scheme = _scheme(tuple(test.criteria[0] for test in scheme.tests[:5]))
    result = _service(student=selected_student, scheme=selected_scheme)[0].mark_paper(
        _request()
    )
    assert result.status is FinalPaperStatus.COMPLETED
    assert result.final_score == result.total_confirmed_marks == 9
    assert result.unresolved_marks == 0


@pytest.mark.parametrize(
    ("invalid", "code"),
    [
        ("synthetic-student.pdf", OrchestrationErrorCode.INVALID_STUDENT_PDF),
        ("synthetic-scheme.pdf", OrchestrationErrorCode.INVALID_MARKING_SCHEME),
    ],
)
def test_invalid_inputs_fail_fast(invalid: str, code: OrchestrationErrorCode) -> None:
    service, provider = _service(validator=FakeValidator(invalid))
    result = service.mark_paper(_request())
    assert result.status is FinalPaperStatus.FAILED
    assert result.errors[0].code is code
    assert result.test_results == ()
    assert provider.calls == 0


@pytest.mark.parametrize(
    ("student_error", "scheme_error", "code"),
    [
        (
            AnswerExtractionError("private source path"),
            None,
            OrchestrationErrorCode.STUDENT_PROCESSING_FAILED,
        ),
        (
            None,
            MarkingSchemeIngestionError("private rubric"),
            OrchestrationErrorCode.SCHEME_INGESTION_FAILED,
        ),
    ],
)
def test_pipeline_failures_return_safe_errors(
    student_error: Exception | None,
    scheme_error: Exception | None,
    code: OrchestrationErrorCode,
) -> None:
    result = _service(student_error=student_error, scheme_error=scheme_error)[
        0
    ].mark_paper(_request())
    assert result.status is FinalPaperStatus.FAILED
    assert result.errors[0].code is code
    assert "private" not in result.errors[0].message


@pytest.mark.parametrize(("malformed", "fail"), [(True, False), (False, True)])
def test_subjective_failures_are_isolated_and_queued(
    malformed: bool, fail: bool
) -> None:
    provider = FakeSubjectiveProvider(malformed=malformed, fail=fail)
    result = _service(provider=provider)[0].mark_paper(_request())
    assert result.status is FinalPaperStatus.COMPLETED_WITH_REVIEW
    assert result.objective_confirmed_marks == 4
    assert result.subjective_confirmed_marks == 0
    assert result.review_required_marks == 9
    assert result.final_score is None
    assert len(result.review_items) == 4


def test_missing_and_ocr_failed_tests_are_partial_not_fatal() -> None:
    student, scheme = _scenario()
    answers = (
        *student.answers[:2],
        _answer(3, "", AnswerExtractionStatus.OCR_FAILED),
        *student.answers[3:7],
    )
    partial_student = _student_paper(answers)
    result = _service(student=partial_student, scheme=scheme)[0].mark_paper(_request())
    assert result.status is FinalPaperStatus.PARTIALLY_COMPLETED
    assert 8 in result.missing_tests
    assert result.final_score is None
    assert any(item.test_number == 3 for item in result.review_items)


def test_missing_scheme_test_is_partial_and_reviewable() -> None:
    student, scheme = _scenario()
    partial_scheme = _scheme(tuple(test.criteria[0] for test in scheme.tests[:-1]))
    result = _service(student=student, scheme=partial_scheme)[0].mark_paper(_request())
    assert result.status is FinalPaperStatus.PARTIALLY_COMPLETED
    assert 8 in result.missing_tests
    assert result.final_score is None
    assert any(item.test_number == 8 for item in result.review_items)


def test_private_serialization_is_deterministic_and_atomic(tmp_path: Path) -> None:
    private = tmp_path / "runtime"
    result = _service(private_root=private)[0].mark_paper(_request())
    assert final_result_json(result) == final_result_json(result)
    target = private / "result.json"
    assert write_private_final_result(result, target, private_root=private) == target
    assert target.is_file()
    with pytest.raises(OrchestrationError, match="remain private"):
        write_private_final_result(
            result, tmp_path / "public.json", private_root=private
        )


def test_requested_private_output_is_written(tmp_path: Path) -> None:
    private = tmp_path / "runtime"
    target = private / "run" / "result.json"
    result = _service(private_root=private)[0].mark_paper(_request(target))
    assert result.status is FinalPaperStatus.COMPLETED_WITH_REVIEW
    assert target.is_file()


def test_public_output_path_fails_before_processing(tmp_path: Path) -> None:
    private = tmp_path / "runtime"
    result = _service(private_root=private)[0].mark_paper(
        _request(tmp_path / "public.json")
    )
    assert result.status is FinalPaperStatus.FAILED
    assert result.errors[0].code is OrchestrationErrorCode.INVALID_OUTPUT_PATH


def test_stage_failures_never_expose_stack_trace_or_paths() -> None:
    result = _service(student_error=AnswerExtractionError("/private/student/name.pdf"))[
        0
    ].mark_paper(_request())
    payload = final_result_json(result)
    assert "/private/student/name.pdf" not in payload
    assert "Traceback" not in payload


def test_alignment_failure_is_reported_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _ = _service()

    def fail_alignment(*_: object) -> object:
        raise ValueError("private alignment detail")

    monkeypatch.setattr(service._alignment, "align", fail_alignment)
    result = service.mark_paper(_request())
    assert result.status is FinalPaperStatus.FAILED
    assert result.errors[0].code is OrchestrationErrorCode.ALIGNMENT_FAILED
    assert "private" not in final_result_json(result)


def test_objective_and_aggregation_failures_are_mapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective_service, _ = _service()

    def fail_objective(*_: object) -> object:
        raise ValueError("private objective detail")

    monkeypatch.setattr(
        objective_service._combined,
        "mark_objective",
        fail_objective,
    )
    objective_result = objective_service.mark_paper(_request())
    assert (
        objective_result.errors[0].code
        is OrchestrationErrorCode.OBJECTIVE_MARKING_FAILED
    )

    aggregation_service, _ = _service()

    def fail_aggregation(*_: object) -> object:
        raise ValueError("private aggregation detail")

    monkeypatch.setattr(
        aggregation_service._combined,
        "combine",
        fail_aggregation,
    )
    aggregation_result = aggregation_service.mark_paper(_request())
    assert (
        aggregation_result.errors[0].code
        is OrchestrationErrorCode.RESULT_AGGREGATION_FAILED
    )


def test_subjective_statuses_remain_visible_in_review_queue() -> None:
    result = _service(provider=FakeSubjectiveProvider(malformed=True))[0].mark_paper(
        _request()
    )
    statuses = {item.status for item in result.review_items}
    assert SubjectiveMarkingStatus.INVALID_MODEL_RESPONSE.value in statuses
    assert SubjectiveMarkingStatus.UNSUPPORTED_CRITERION.value in statuses

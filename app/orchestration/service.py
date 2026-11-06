"""One application service coordinating the complete marking workflow."""

from __future__ import annotations

import logging
from decimal import Decimal
from pathlib import Path
from time import perf_counter
from uuid import UUID

from pydantic import ValidationError

from app.core.exceptions import ApplicationError, InvalidPDFError, OrchestrationError
from app.feedback.models import QuestionFeedback, QuestionFeedbackStatus
from app.feedback.service import PaperFeedbackService
from app.marking.models import MarkingStatus
from app.marking.subjective.combined import CombinedMarkingService
from app.marking.subjective.models import (
    CombinedPaperMarkingResult,
    SubjectiveMarkingStatus,
)
from app.marking_scheme.alignment import TestAlignmentService
from app.marking_scheme.models import AlignedPaper
from app.orchestration.interfaces import (
    InputValidator,
    MarkingSchemePipeline,
    StudentPaperPipeline,
)
from app.orchestration.models import (
    ORCHESTRATION_VERSION,
    ComponentVersion,
    CriterionFeedback,
    FinalPaperMarkingResult,
    FinalPaperStatus,
    OrchestrationErrorCode,
    OrchestrationIssue,
    OrchestrationProvenance,
    PaperMarkingRequest,
    PipelineStage,
    ReviewItem,
    StageStatus,
    StageSummary,
)
from app.orchestration.serialization import write_private_final_result

logger = logging.getLogger(__name__)


class PaperMarkingService:
    """Coordinate existing services without owning their business rules."""

    def __init__(
        self,
        *,
        input_validator: InputValidator,
        student_pipeline: StudentPaperPipeline,
        scheme_pipeline: MarkingSchemePipeline,
        alignment_service: TestAlignmentService,
        combined_marking_service: CombinedMarkingService,
        private_output_root: Path,
        formative_feedback_service: PaperFeedbackService | None = None,
    ) -> None:
        self._input_validator = input_validator
        self._student_pipeline = student_pipeline
        self._scheme_pipeline = scheme_pipeline
        self._alignment = alignment_service
        self._combined = combined_marking_service
        self._private_output_root = private_output_root
        self._formative_feedback = formative_feedback_service

    def mark_paper(self, request: PaperMarkingRequest) -> FinalPaperMarkingResult:
        """Run one complete paper and return a safe result for every outcome."""

        started = perf_counter()
        stages: list[StageSummary] = []

        validation_started = perf_counter()
        try:
            self._input_validator.validate(request.student_pdf)
        except (InvalidPDFError, OSError, ValueError):
            return self._failed(
                request,
                stages,
                PipelineStage.INPUT_VALIDATION,
                OrchestrationErrorCode.INVALID_STUDENT_PDF,
                "Student PDF validation failed",
                validation_started,
                started,
            )
        try:
            self._input_validator.validate(request.marking_scheme_pdf)
        except (InvalidPDFError, OSError, ValueError):
            return self._failed(
                request,
                stages,
                PipelineStage.INPUT_VALIDATION,
                OrchestrationErrorCode.INVALID_MARKING_SCHEME,
                "Marking-scheme PDF validation failed",
                validation_started,
                started,
            )
        if request.output_path is not None and not _is_private_path(
            request.output_path, self._private_output_root
        ):
            return self._failed(
                request,
                stages,
                PipelineStage.INPUT_VALIDATION,
                OrchestrationErrorCode.INVALID_OUTPUT_PATH,
                "Output path must remain under private runtime storage",
                validation_started,
                started,
            )
        stages.append(
            self._complete_stage(
                PipelineStage.INPUT_VALIDATION,
                validation_started,
                run_id=request.run_id,
            )
        )

        student_started = perf_counter()
        try:
            student_output = self._student_pipeline.process(
                request.student_pdf, run_id=request.run_id
            )
        except (ApplicationError, ValidationError, OSError, ValueError):
            return self._failed(
                request,
                stages,
                PipelineStage.STUDENT_PDF_PROCESSING,
                OrchestrationErrorCode.STUDENT_PROCESSING_FAILED,
                "Student paper processing failed",
                student_started,
                started,
            )
        stages.extend(student_output.stages)
        for summary in student_output.stages:
            _log_stage(summary, run_id=request.run_id)

        scheme_started = perf_counter()
        try:
            scheme_output = self._scheme_pipeline.process(
                request.marking_scheme_pdf, run_id=request.run_id
            )
        except (ApplicationError, ValidationError, OSError, ValueError):
            return self._failed(
                request,
                stages,
                PipelineStage.MARKING_SCHEME_INGESTION,
                OrchestrationErrorCode.SCHEME_INGESTION_FAILED,
                "Marking-scheme ingestion failed",
                scheme_started,
                started,
                student_document_id=student_output.paper.paper_id,
            )
        stages.extend(scheme_output.stages)
        for summary in scheme_output.stages:
            _log_stage(summary, run_id=request.run_id)

        alignment_started = perf_counter()
        try:
            aligned = self._alignment.align(student_output.paper, scheme_output.scheme)
        except (ApplicationError, ValidationError, ValueError):
            return self._failed(
                request,
                stages,
                PipelineStage.TEST_ALIGNMENT,
                OrchestrationErrorCode.ALIGNMENT_FAILED,
                "Exact Test alignment failed",
                alignment_started,
                started,
                student_document_id=student_output.paper.paper_id,
                marking_scheme_id=scheme_output.scheme.scheme_id,
            )
        alignment_warnings = _alignment_warnings(aligned)
        stages.append(
            self._complete_stage(
                PipelineStage.TEST_ALIGNMENT,
                alignment_started,
                warnings=alignment_warnings,
                run_id=request.run_id,
            )
        )

        objective_started = perf_counter()
        try:
            objective = self._combined.mark_objective(aligned)
        except (ApplicationError, ValidationError, ValueError):
            return self._failed_after_alignment(
                request,
                stages,
                aligned,
                scheme_output.scheme.scheme_id,
                PipelineStage.OBJECTIVE_MARKING,
                OrchestrationErrorCode.OBJECTIVE_MARKING_FAILED,
                "Objective marking failed",
                objective_started,
                started,
            )
        objective_warnings = tuple(warning.value for warning in objective.warnings)
        stages.append(
            self._complete_stage(
                PipelineStage.OBJECTIVE_MARKING,
                objective_started,
                warnings=objective_warnings,
                run_id=request.run_id,
            )
        )

        subjective_started = perf_counter()
        try:
            subjective = self._combined.mark_subjective(aligned)
        except (ApplicationError, ValidationError, OSError, ValueError):
            return self._failed_after_alignment(
                request,
                stages,
                aligned,
                scheme_output.scheme.scheme_id,
                PipelineStage.SUBJECTIVE_MARKING,
                OrchestrationErrorCode.SUBJECTIVE_MARKING_PARTIAL,
                "Subjective marking could not produce safe results",
                subjective_started,
                started,
            )
        subjective_warnings = (
            ("subjective_review_or_unsupported",)
            if subjective.review_required_marks or subjective.unsupported_marks
            else ()
        )
        stages.append(
            self._complete_stage(
                PipelineStage.SUBJECTIVE_MARKING,
                subjective_started,
                warnings=subjective_warnings,
                run_id=request.run_id,
            )
        )

        question_feedback: tuple[QuestionFeedback, ...] = ()
        if self._formative_feedback is not None:
            feedback_started = perf_counter()
            question_feedback = self._formative_feedback.generate(aligned)
            feedback_failures = sum(
                item.status is QuestionFeedbackStatus.FAILED
                for item in question_feedback
            )
            feedback_warnings = (
                (f"feedback_generation_failed:{feedback_failures}",)
                if feedback_failures
                else ()
            )
            stages.append(
                self._complete_stage(
                    PipelineStage.FORMATIVE_FEEDBACK,
                    feedback_started,
                    warnings=feedback_warnings,
                    run_id=request.run_id,
                )
            )

        aggregation_started = perf_counter()
        try:
            combined = self._combined.combine(aligned, objective, subjective)
            result = _final_result(
                request,
                student_output.paper,
                scheme_output.scheme,
                aligned,
                combined,
                question_feedback,
                (
                    *stages,
                    self._complete_stage(
                        PipelineStage.RESULT_AGGREGATION,
                        aggregation_started,
                        run_id=request.run_id,
                    ),
                ),
                (perf_counter() - started) * 1000,
            )
            if request.output_path is not None:
                write_private_final_result(
                    result,
                    request.output_path,
                    private_root=self._private_output_root,
                )
        except (ApplicationError, ValidationError, OSError, ValueError):
            return self._failed_after_alignment(
                request,
                stages,
                aligned,
                scheme_output.scheme.scheme_id,
                PipelineStage.RESULT_AGGREGATION,
                OrchestrationErrorCode.RESULT_AGGREGATION_FAILED,
                "Final result aggregation failed",
                aggregation_started,
                started,
            )
        return result

    def _complete_stage(
        self,
        stage: PipelineStage,
        started: float,
        *,
        warnings: tuple[str, ...] = (),
        run_id: UUID | None = None,
    ) -> StageSummary:
        summary = StageSummary(
            stage=stage,
            status=(
                StageStatus.COMPLETED_WITH_WARNINGS
                if warnings
                else StageStatus.COMPLETED
            ),
            duration_ms=(perf_counter() - started) * 1000,
            warnings=warnings,
        )
        _log_stage(summary, run_id=run_id)
        return summary

    def _failed(
        self,
        request: PaperMarkingRequest,
        stages: list[StageSummary],
        stage: PipelineStage,
        code: OrchestrationErrorCode,
        message: str,
        stage_started: float,
        pipeline_started: float,
        *,
        student_document_id: UUID | None = None,
        marking_scheme_id: UUID | None = None,
    ) -> FinalPaperMarkingResult:
        failed_stage = StageSummary(
            stage=stage,
            status=StageStatus.FAILED,
            duration_ms=(perf_counter() - stage_started) * 1000,
            error_code=code,
        )
        _log_stage(failed_stage, run_id=request.run_id)
        return FinalPaperMarkingResult(
            run_id=request.run_id,
            status=FinalPaperStatus.FAILED,
            student_document_id=student_document_id,
            marking_scheme_id=marking_scheme_id,
            stages=(*stages, failed_stage),
            errors=(OrchestrationIssue(code=code, stage=stage, message=message),),
            total_duration_ms=(perf_counter() - pipeline_started) * 1000,
            provenance=_failure_provenance(),
        )

    def _failed_after_alignment(
        self,
        request: PaperMarkingRequest,
        stages: list[StageSummary],
        aligned: AlignedPaper,
        scheme_id: UUID,
        stage: PipelineStage,
        code: OrchestrationErrorCode,
        message: str,
        stage_started: float,
        pipeline_started: float,
    ) -> FinalPaperMarkingResult:
        return self._failed(
            request,
            stages,
            stage,
            code,
            message,
            stage_started,
            pipeline_started,
            student_document_id=aligned.paper_id,
            marking_scheme_id=scheme_id,
        )


def _final_result(
    request: PaperMarkingRequest,
    student: object,
    scheme: object,
    aligned: AlignedPaper,
    combined: CombinedPaperMarkingResult,
    question_feedback: tuple[QuestionFeedback, ...],
    stages: tuple[StageSummary, ...],
    duration_ms: float,
) -> FinalPaperMarkingResult:
    from app.answers.models import StructuredStudentPaper
    from app.marking_scheme.models import StructuredMarkingScheme

    if not isinstance(student, StructuredStudentPaper) or not isinstance(
        scheme, StructuredMarkingScheme
    ):
        raise OrchestrationError("Pipeline returned invalid domain artifacts")
    partial = bool(
        aligned.student_missing_test_numbers
        or aligned.scheme_missing_test_numbers
        or aligned.ocr_failed_test_numbers
        or aligned.structure_incomplete_test_numbers
    )
    review_items = _review_items(combined, aligned)
    status = (
        FinalPaperStatus.PARTIALLY_COMPLETED
        if partial
        else FinalPaperStatus.COMPLETED_WITH_REVIEW
        if combined.final_score is None
        else FinalPaperStatus.COMPLETED
    )
    warnings = tuple(
        dict.fromkeys(warning for stage in stages for warning in stage.warnings)
    )
    return FinalPaperMarkingResult(
        run_id=request.run_id,
        status=status,
        student_document_id=student.paper_id,
        marking_scheme_id=scheme.scheme_id,
        detected_tests=student.detected_test_numbers,
        missing_tests=tuple(
            sorted(
                set(student.missing_test_numbers)
                | set(aligned.student_missing_test_numbers)
                | set(aligned.scheme_missing_test_numbers)
            )
        ),
        aligned_tests=tuple(
            item.test_number
            for item in aligned.tests
            if item.student_answer is not None and item.marking_scheme_test is not None
        ),
        objective_confirmed_marks=combined.confirmed_objective_marks,
        subjective_confirmed_marks=combined.confirmed_subjective_marks,
        total_confirmed_marks=combined.confirmed_marks,
        known_maximum_marks=combined.maximum_known_marks,
        review_required_marks=combined.review_required_marks,
        unsupported_marks=combined.unsupported_marks,
        unresolved_marks=(combined.review_required_marks + combined.unsupported_marks),
        final_score=combined.final_score
        if status is FinalPaperStatus.COMPLETED
        else None,
        test_results=combined.test_results,
        feedback=_feedback(combined),
        question_feedback=question_feedback,
        review_items=review_items,
        stages=stages,
        warnings=warnings,
        total_duration_ms=duration_ms,
        provenance=_provenance(student, scheme, combined),
    )


def _review_items(
    combined: CombinedPaperMarkingResult, aligned: AlignedPaper
) -> tuple[ReviewItem, ...]:
    output: list[ReviewItem] = []
    for test in combined.test_results:
        for result in test.objective_results:
            if result.status not in {MarkingStatus.MARKED, MarkingStatus.NOT_ATTEMPTED}:
                output.append(
                    ReviewItem(
                        test_number=test.test_number,
                        criterion_id=result.criterion_id,
                        reason=result.reason_codes[0].value,
                        status=result.status.value,
                        possible_marks=result.maximum_marks or Decimal("0"),
                        source_reference=result.criterion_id,
                    )
                )
        for subjective_result in test.subjective_results:
            if subjective_result.status is not SubjectiveMarkingStatus.MARKED:
                output.append(
                    ReviewItem(
                        test_number=test.test_number,
                        criterion_id=subjective_result.criterion_id,
                        reason=str(subjective_result.reason_codes[0]),
                        status=subjective_result.status.value,
                        possible_marks=(
                            subjective_result.maximum_marks or Decimal("0")
                        ),
                        source_reference=subjective_result.criterion_id,
                    )
                )
    existing = {item.test_number for item in output if item.criterion_id is None}
    for test_number in tuple(
        sorted(
            set(aligned.student_missing_test_numbers)
            | set(aligned.scheme_missing_test_numbers)
            | set(aligned.ocr_failed_test_numbers)
            | set(aligned.structure_incomplete_test_numbers)
        )
    ):
        if test_number not in existing:
            output.append(
                ReviewItem(
                    test_number=test_number,
                    reason="test_alignment_incomplete",
                    status="review_required",
                    possible_marks=Decimal("0"),
                    source_reference=f"test-{test_number:02d}",
                )
            )
    return tuple(
        sorted(output, key=lambda item: (item.test_number, item.criterion_id or ""))
    )


def _feedback(combined: CombinedPaperMarkingResult) -> tuple[CriterionFeedback, ...]:
    return tuple(
        CriterionFeedback(
            test_number=test.test_number,
            criterion_id=result.criterion_id,
            status=result.status.value,
            feedback=result.feedback,
        )
        for test in combined.test_results
        for result in test.subjective_results
        if result.feedback
    )


def _provenance(
    student: object,
    scheme: object,
    combined: CombinedPaperMarkingResult,
) -> OrchestrationProvenance:
    from app.answers.models import StructuredStudentPaper
    from app.marking_scheme.models import StructuredMarkingScheme

    if not isinstance(student, StructuredStudentPaper) or not isinstance(
        scheme, StructuredMarkingScheme
    ):
        raise OrchestrationError("Cannot construct orchestration provenance")
    versions = [
        ComponentVersion(
            component="student_extraction",
            version=student.provenance.extraction_version,
        ),
        ComponentVersion(
            component="student_ocr",
            version=(
                f"{student.provenance.ocr_provider}:"
                f"{student.provenance.ocr_model_version or 'unknown'}"
            ),
        ),
        ComponentVersion(
            component="marking_scheme_ingestion",
            version=scheme.provenance.ingestion_version,
        ),
        ComponentVersion(
            component="marking_scheme_parser",
            version=scheme.provenance.parser_version,
        ),
        ComponentVersion(component="alignment", version="exact-test-alignment-v1"),
    ]
    versions.extend(
        ComponentVersion(component=f"marking_engine_{index}", version=version)
        for index, version in enumerate(combined.engine_versions, start=1)
    )
    subjective_provenance = next(
        (
            result.provenance
            for test in combined.test_results
            for result in test.subjective_results
        ),
        None,
    )
    if subjective_provenance is not None:
        versions.extend(
            (
                ComponentVersion(
                    component="subjective_provider",
                    version=(
                        f"{subjective_provenance.provider}:"
                        f"{subjective_provenance.model_version}"
                    ),
                ),
                ComponentVersion(
                    component="subjective_prompt",
                    version=subjective_provenance.prompt_version,
                ),
            )
        )
    return OrchestrationProvenance(
        orchestration_version=ORCHESTRATION_VERSION,
        components=tuple(versions),
    )


def _failure_provenance() -> OrchestrationProvenance:
    return OrchestrationProvenance(
        orchestration_version=ORCHESTRATION_VERSION,
        components=(),
    )


def _alignment_warnings(aligned: AlignedPaper) -> tuple[str, ...]:
    warnings: list[str] = []
    for label, numbers in (
        ("student_missing", aligned.student_missing_test_numbers),
        ("scheme_missing", aligned.scheme_missing_test_numbers),
        ("review_required", aligned.review_required_test_numbers),
        ("ocr_failed", aligned.ocr_failed_test_numbers),
        ("structure_incomplete", aligned.structure_incomplete_test_numbers),
    ):
        if numbers:
            warnings.append(f"{label}_tests:{len(numbers)}")
    return tuple(warnings)


def _is_private_path(path: Path, private_root: Path) -> bool:
    return (
        path.expanduser().resolve().is_relative_to(private_root.expanduser().resolve())
    )


def _log_stage(summary: StageSummary, *, run_id: object | None = None) -> None:
    logger.info(
        "paper_marking_stage",
        extra={
            "run_id": str(run_id) if run_id is not None else None,
            "stage": summary.stage.value,
            "duration_ms": round(summary.duration_ms, 3),
            "status": summary.status.value,
            "error_code": (
                summary.error_code.value if summary.error_code is not None else None
            ),
        },
    )

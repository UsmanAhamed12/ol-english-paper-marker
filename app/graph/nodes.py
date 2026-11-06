"""Thin LangGraph node adapters around existing application services."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from langgraph.types import interrupt

from app.graph.review import (
    HumanReviewResponse,
    apply_human_review,
    build_human_review_payload,
)
from app.graph.state import PaperMarkingState
from app.marking.models import PaperMarkingResult
from app.marking.subjective.combined import CombinedMarkingService
from app.marking.subjective.models import (
    CombinedPaperMarkingResult,
    SubjectivePaperMarkingResult,
)
from app.marking_scheme.alignment import TestAlignmentService
from app.orchestration.interfaces import (
    InputValidator,
    MarkingSchemePipeline,
    StudentPaperPipeline,
)
from app.orchestration.models import FinalPaperStatus, PipelineStage

NodeFunction = Callable[[PaperMarkingState], PaperMarkingState]


def create_document_validation_node(validator: InputValidator) -> NodeFunction:
    """Validate both file references without loading either document into state."""

    def validate_documents(state: PaperMarkingState) -> PaperMarkingState:
        student_path = state.get("student_pdf_path")
        scheme_path = state.get("marking_scheme_pdf_path")
        if not student_path or not scheme_path:
            raise ValueError("Both PDF paths are required before document validation")
        validator.validate(Path(student_path))
        validator.validate(Path(scheme_path))
        return {
            "documents_validated": True,
            "current_stage": PipelineStage.INPUT_VALIDATION,
        }

    return validate_documents


def create_student_paper_node(pipeline: StudentPaperPipeline) -> NodeFunction:
    """Run the existing student-paper pipeline and retain its typed output."""

    def process_student_paper(state: PaperMarkingState) -> PaperMarkingState:
        path = state.get("student_pdf_path")
        run_id = state.get("run_id")
        if not path or not run_id:
            raise ValueError("Student PDF path and run ID are required")
        output = pipeline.process(Path(path), run_id=UUID(run_id))
        return {
            "student_output": output,
            "current_stage": PipelineStage.STUDENT_PDF_PROCESSING,
        }

    return process_student_paper


def create_marking_scheme_node(pipeline: MarkingSchemePipeline) -> NodeFunction:
    """Run the existing marking-scheme pipeline and retain typed output."""

    def process_marking_scheme(state: PaperMarkingState) -> PaperMarkingState:
        path = state.get("marking_scheme_pdf_path")
        run_id = state.get("run_id")
        if not path or not run_id:
            raise ValueError("Marking-scheme PDF path and run ID are required")
        output = pipeline.process(Path(path), run_id=UUID(run_id))
        return {
            "scheme_output": output,
            "current_stage": PipelineStage.MARKING_SCHEME_INGESTION,
        }

    return process_marking_scheme


def create_alignment_node(alignment: TestAlignmentService) -> NodeFunction:
    """Align the structured student paper and marking scheme by exact Test."""

    def align_tests(state: PaperMarkingState) -> PaperMarkingState:
        student_output = state.get("student_output")
        scheme_output = state.get("scheme_output")
        if student_output is None or scheme_output is None:
            raise ValueError("Student and scheme outputs are required for alignment")
        aligned = alignment.align(student_output.paper, scheme_output.scheme)
        return {
            "aligned_paper": aligned,
            "current_stage": PipelineStage.TEST_ALIGNMENT,
        }

    return align_tests


def create_objective_marking_node(marking: CombinedMarkingService) -> NodeFunction:
    """Run deterministic marking once against the aligned paper."""

    def mark_objective(state: PaperMarkingState) -> PaperMarkingState:
        aligned = state.get("aligned_paper")
        if aligned is None:
            raise ValueError("Aligned paper is required for objective marking")
        result: PaperMarkingResult = marking.mark_objective(aligned)
        return {
            "objective_result": result,
            "current_stage": PipelineStage.OBJECTIVE_MARKING,
        }

    return mark_objective


def create_subjective_marking_node(marking: CombinedMarkingService) -> NodeFunction:
    """Run guarded subjective marking once against the aligned paper."""

    def mark_subjective(state: PaperMarkingState) -> PaperMarkingState:
        aligned = state.get("aligned_paper")
        if aligned is None:
            raise ValueError("Aligned paper is required for subjective marking")
        result: SubjectivePaperMarkingResult = marking.mark_subjective(aligned)
        return {
            "subjective_result": result,
            "current_stage": PipelineStage.SUBJECTIVE_MARKING,
        }

    return mark_subjective


def create_aggregation_node(marking: CombinedMarkingService) -> NodeFunction:
    """Combine prior marking results without invoking either marking engine."""

    def aggregate_scores(state: PaperMarkingState) -> PaperMarkingState:
        aligned = state.get("aligned_paper")
        objective = state.get("objective_result")
        subjective = state.get("subjective_result")
        if aligned is None or objective is None or subjective is None:
            raise ValueError("Alignment and both marking results are required")
        result: CombinedPaperMarkingResult = marking.combine(
            aligned,
            objective,
            subjective,
        )
        return {
            "combined_result": result,
            "current_stage": PipelineStage.RESULT_AGGREGATION,
        }

    return aggregate_scores


def create_completed_node() -> NodeFunction:
    """Set completed status after a fully resolved combined result."""

    def mark_completed(state: PaperMarkingState) -> PaperMarkingState:
        if state.get("combined_result") is None:
            raise ValueError("Combined result is required before completion")
        return {
            "workflow_status": FinalPaperStatus.COMPLETED,
            "review_required": False,
        }

    return mark_completed


def create_review_required_node() -> NodeFunction:
    """Record review status without changing confirmed or unresolved marks."""

    def mark_review_required(state: PaperMarkingState) -> PaperMarkingState:
        if state.get("combined_result") is None:
            raise ValueError("Combined result is required before review routing")
        return {
            "workflow_status": FinalPaperStatus.COMPLETED_WITH_REVIEW,
            "review_required": True,
        }

    return mark_review_required


def create_human_review_node() -> NodeFunction:
    """Pause on unresolved scores and apply bounded teacher decisions on resume."""

    def review_unresolved_marks(state: PaperMarkingState) -> PaperMarkingState:
        combined = state.get("combined_result")
        run_id = state.get("run_id")
        if combined is None or not run_id:
            raise ValueError("Combined result and run ID are required for human review")
        aligned = state.get("aligned_paper")
        payload = build_human_review_payload(combined, aligned, run_id)
        review_items = payload["items"]
        if not isinstance(review_items, list):
            raise ValueError("Human review payload must contain a list of items")
        if not any(item["can_assign_mark"] for item in review_items):
            return {"current_stage": PipelineStage.HUMAN_REVIEW}

        response = HumanReviewResponse.model_validate(
            interrupt(payload, response_schema=HumanReviewResponse)
        )
        return {
            "combined_result": apply_human_review(combined, response),
            "human_review_response": response,
            "current_stage": PipelineStage.HUMAN_REVIEW,
        }

    return review_unresolved_marks

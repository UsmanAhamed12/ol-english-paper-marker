"""Typed, serializable workflow state for paper marking."""

from __future__ import annotations

from typing import NotRequired

from typing_extensions import TypedDict

from app.graph.review import HumanReviewResponse
from app.marking.models import PaperMarkingResult
from app.marking.subjective.models import (
    CombinedPaperMarkingResult,
    SubjectivePaperMarkingResult,
)
from app.marking_scheme.models import AlignedPaper
from app.orchestration.interfaces import SchemePipelineOutput, StudentPipelineOutput
from app.orchestration.models import (
    FinalPaperStatus,
    OrchestrationIssue,
    PipelineStage,
)


class PaperMarkingState(TypedDict, total=False):
    """References and typed workflow artifacts; never raw PDF/image bytes."""

    run_id: str
    student_pdf_path: str
    marking_scheme_pdf_path: str
    documents_validated: NotRequired[bool]
    student_output: NotRequired[StudentPipelineOutput]
    scheme_output: NotRequired[SchemePipelineOutput]
    aligned_paper: NotRequired[AlignedPaper]
    objective_result: NotRequired[PaperMarkingResult]
    subjective_result: NotRequired[SubjectivePaperMarkingResult]
    combined_result: NotRequired[CombinedPaperMarkingResult]
    human_review_response: NotRequired[HumanReviewResponse]
    current_stage: NotRequired[PipelineStage]
    current_question: NotRequired[int]
    ocr_review_required: NotRequired[bool]
    grading_review_required: NotRequired[bool]
    review_required: NotRequired[bool]
    workflow_status: NotRequired[FinalPaperStatus]
    warnings: NotRequired[tuple[str, ...]]
    errors: NotRequired[tuple[OrchestrationIssue, ...]]

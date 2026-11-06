"""Tests for typed marking graph node updates and dependency calls."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import cast
from unittest.mock import Mock, call
from uuid import UUID

from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.builder import build_marking_graph
from app.graph.checkpoints import graph_config_for_run
from app.graph.nodes import (
    create_aggregation_node,
    create_alignment_node,
    create_document_validation_node,
    create_marking_scheme_node,
    create_objective_marking_node,
    create_student_paper_node,
    create_subjective_marking_node,
)
from app.graph.routing import route_after_aggregation
from app.graph.state import PaperMarkingState
from app.marking.models import PaperMarkingResult
from app.marking.subjective.combined import CombinedMarkingService
from app.marking.subjective.models import (
    CombinedPaperMarkingResult,
    SubjectivePaperMarkingResult,
)
from app.marking_scheme.alignment import TestAlignmentService
from app.marking_scheme.models import AlignedPaper
from app.orchestration.interfaces import (
    InputValidator,
    MarkingSchemePipeline,
    SchemePipelineOutput,
    StudentPaperPipeline,
    StudentPipelineOutput,
)
from app.orchestration.models import FinalPaperStatus, PipelineStage

RUN_ID = UUID("30000000-0000-0000-0000-000000000003")


def test_document_validation_node_uses_path_references_and_returns_update() -> None:
    validator = Mock()
    node = create_document_validation_node(cast(InputValidator, validator))
    state: PaperMarkingState = {
        "run_id": str(RUN_ID),
        "student_pdf_path": "student.pdf",
        "marking_scheme_pdf_path": "scheme.pdf",
    }

    update = node(state)

    assert update == {
        "documents_validated": True,
        "current_stage": PipelineStage.INPUT_VALIDATION,
    }
    assert validator.validate.call_count == 2


def test_processing_nodes_call_injected_pipelines_and_return_artifact_updates() -> None:
    student_output = cast(StudentPipelineOutput, Mock(paper=object()))
    scheme_output = cast(SchemePipelineOutput, Mock(scheme=object()))
    student_pipeline = Mock()
    student_pipeline.process.return_value = student_output
    scheme_pipeline = Mock()
    scheme_pipeline.process.return_value = scheme_output
    state: PaperMarkingState = {
        "run_id": str(RUN_ID),
        "student_pdf_path": "student.pdf",
        "marking_scheme_pdf_path": "scheme.pdf",
    }

    student_update = create_student_paper_node(
        cast(StudentPaperPipeline, student_pipeline)
    )(state)
    scheme_update = create_marking_scheme_node(
        cast(MarkingSchemePipeline, scheme_pipeline)
    )(state)

    assert student_update["student_output"] is student_output
    assert student_update["current_stage"] is PipelineStage.STUDENT_PDF_PROCESSING
    assert scheme_update["scheme_output"] is scheme_output
    assert scheme_update["current_stage"] is PipelineStage.MARKING_SCHEME_INGESTION
    assert student_pipeline.process.call_args.kwargs["run_id"] == RUN_ID
    assert scheme_pipeline.process.call_args.kwargs["run_id"] == RUN_ID


def test_alignment_and_marking_nodes_return_typed_partial_updates() -> None:
    aligned = cast(AlignedPaper, object())
    objective = cast(PaperMarkingResult, object())
    subjective = cast(SubjectivePaperMarkingResult, object())
    combined = cast(CombinedPaperMarkingResult, object())
    student_output = cast(StudentPipelineOutput, Mock(paper=object()))
    scheme_output = cast(SchemePipelineOutput, Mock(scheme=object()))
    alignment = Mock()
    alignment.align.return_value = aligned
    marking = Mock()
    marking.mark_objective.return_value = objective
    marking.mark_subjective.return_value = subjective
    marking.combine.return_value = combined
    state: PaperMarkingState = {
        "run_id": str(RUN_ID),
        "student_pdf_path": "student.pdf",
        "marking_scheme_pdf_path": "scheme.pdf",
        "student_output": student_output,
        "scheme_output": scheme_output,
        "aligned_paper": aligned,
        "objective_result": objective,
        "subjective_result": subjective,
    }

    alignment_update = create_alignment_node(cast(TestAlignmentService, alignment))(
        state
    )
    objective_update = create_objective_marking_node(
        cast(CombinedMarkingService, marking)
    )(state)
    subjective_update = create_subjective_marking_node(
        cast(CombinedMarkingService, marking)
    )(state)
    aggregate_update = create_aggregation_node(cast(CombinedMarkingService, marking))(
        state
    )

    assert alignment_update["aligned_paper"] is aligned
    assert objective_update["objective_result"] is objective
    assert subjective_update["subjective_result"] is subjective
    assert aggregate_update["combined_result"] is combined


def test_state_schema_is_accepted_by_langgraph() -> None:
    validator = Mock()
    builder = StateGraph(PaperMarkingState)
    builder.add_node(
        "validate_documents",
        RunnableLambda(
            create_document_validation_node(cast(InputValidator, validator))
        ),
    )
    builder.add_edge(START, "validate_documents")
    builder.add_edge("validate_documents", END)
    graph = builder.compile()

    result = graph.invoke(
        {
            "run_id": str(RUN_ID),
            "student_pdf_path": "student.pdf",
            "marking_scheme_pdf_path": "scheme.pdf",
        }
    )

    assert result["documents_validated"] is True
    assert result["current_stage"] == PipelineStage.INPUT_VALIDATION


def test_compiled_graph_uses_injected_checkpointer() -> None:
    checkpointer = InMemorySaver()
    graph = build_marking_graph(
        input_validator=cast(InputValidator, Mock()),
        student_pipeline=cast(StudentPaperPipeline, Mock()),
        scheme_pipeline=cast(MarkingSchemePipeline, Mock()),
        alignment=cast(TestAlignmentService, Mock()),
        marking=cast(CombinedMarkingService, Mock()),
        checkpointer=checkpointer,
    )

    assert graph.checkpointer is checkpointer


def test_checkpointer_persists_state_for_run_thread() -> None:
    builder = StateGraph(PaperMarkingState)

    def validate(_: PaperMarkingState) -> PaperMarkingState:
        return {"documents_validated": True}

    builder.add_node("validate", RunnableLambda(validate))
    builder.add_edge(START, "validate")
    builder.add_edge("validate", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = graph_config_for_run(RUN_ID)
    input_state: PaperMarkingState = {"run_id": str(RUN_ID)}

    result = graph.invoke(input_state, config=config)

    assert graph.get_state(config).values == result


def test_complete_marking_graph_routes_fully_resolved_score() -> None:
    validator = Mock()
    student_output = Mock(paper=object())
    student_pipeline = Mock()
    student_pipeline.process.return_value = student_output
    scheme_output = Mock(scheme=object())
    scheme_pipeline = Mock()
    scheme_pipeline.process.return_value = scheme_output
    aligned = object()
    alignment = Mock()
    alignment.align.return_value = aligned
    objective = object()
    subjective = object()
    combined = Mock(final_score=Decimal("3"))
    marking = Mock()
    marking.mark_objective.return_value = objective
    marking.mark_subjective.return_value = subjective
    marking.combine.return_value = combined
    graph = build_marking_graph(
        input_validator=cast(InputValidator, validator),
        student_pipeline=cast(StudentPaperPipeline, student_pipeline),
        scheme_pipeline=cast(MarkingSchemePipeline, scheme_pipeline),
        alignment=cast(TestAlignmentService, alignment),
        marking=cast(CombinedMarkingService, marking),
    )

    result = graph.invoke(
        {
            "run_id": str(RUN_ID),
            "student_pdf_path": "student.pdf",
            "marking_scheme_pdf_path": "scheme.pdf",
        }
    )

    assert result["documents_validated"] is True
    assert result["combined_result"] is combined
    assert result["current_stage"] is PipelineStage.RESULT_AGGREGATION
    assert result["workflow_status"] is FinalPaperStatus.COMPLETED
    assert result["review_required"] is False
    validator.validate.assert_has_calls(
        [
            call(Path("student.pdf")),
            call(Path("scheme.pdf")),
        ]
    )
    assert student_pipeline.process.call_args.kwargs["run_id"] == RUN_ID
    assert scheme_pipeline.process.call_args.kwargs["run_id"] == RUN_ID
    alignment.align.assert_called_once_with(student_output.paper, scheme_output.scheme)
    marking.mark_objective.assert_called_once_with(aligned)
    marking.mark_subjective.assert_called_once_with(aligned)
    marking.combine.assert_called_once_with(aligned, objective, subjective)


def test_unresolved_score_routes_to_human_review() -> None:
    assert route_after_aggregation({"combined_result": Mock(final_score=None)}) == (
        "review_required"
    )

"""Deterministic LangGraph topology for the paper-marking workflow."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes import (
    create_aggregation_node,
    create_alignment_node,
    create_completed_node,
    create_document_validation_node,
    create_human_review_node,
    create_marking_scheme_node,
    create_objective_marking_node,
    create_review_required_node,
    create_student_paper_node,
    create_subjective_marking_node,
)
from app.graph.routing import route_after_aggregation
from app.graph.state import PaperMarkingState
from app.marking.subjective.combined import CombinedMarkingService
from app.marking_scheme.alignment import TestAlignmentService
from app.orchestration.interfaces import (
    InputValidator,
    MarkingSchemePipeline,
    StudentPaperPipeline,
)


def build_marking_graph(
    *,
    input_validator: InputValidator,
    student_pipeline: StudentPaperPipeline,
    scheme_pipeline: MarkingSchemePipeline,
    alignment: TestAlignmentService,
    marking: CombinedMarkingService,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[
    PaperMarkingState,
    None,
    PaperMarkingState,
    PaperMarkingState,
]:
    """Compile the complete deterministic marking path through score aggregation."""

    builder = StateGraph(PaperMarkingState)
    builder.add_node(
        "validate_documents",
        RunnableLambda(create_document_validation_node(input_validator)),
    )
    builder.add_node(
        "process_student_paper",
        RunnableLambda(create_student_paper_node(student_pipeline)),
    )
    builder.add_node(
        "process_marking_scheme",
        RunnableLambda(create_marking_scheme_node(scheme_pipeline)),
    )
    builder.add_node(
        "align_tests",
        RunnableLambda(create_alignment_node(alignment)),
    )
    builder.add_node(
        "mark_objective",
        RunnableLambda(create_objective_marking_node(marking)),
    )
    builder.add_node(
        "mark_subjective",
        RunnableLambda(create_subjective_marking_node(marking)),
    )
    builder.add_node(
        "aggregate_scores",
        RunnableLambda(create_aggregation_node(marking)),
    )
    builder.add_node("completed", RunnableLambda(create_completed_node()))
    builder.add_node(
        "review_required",
        RunnableLambda(create_review_required_node()),
    )
    builder.add_node("human_review", RunnableLambda(create_human_review_node()))

    builder.add_edge(START, "validate_documents")
    builder.add_edge("validate_documents", "process_student_paper")
    builder.add_edge("process_student_paper", "process_marking_scheme")
    builder.add_edge("process_marking_scheme", "align_tests")
    builder.add_edge("align_tests", "mark_objective")
    builder.add_edge("mark_objective", "mark_subjective")
    builder.add_edge("mark_subjective", "aggregate_scores")
    builder.add_conditional_edges(
        "aggregate_scores",
        route_after_aggregation,
        {
            "completed": "completed",
            "review_required": "human_review",
        },
    )
    builder.add_conditional_edges(
        "human_review",
        route_after_aggregation,
        {
            "completed": "completed",
            "review_required": "review_required",
        },
    )
    builder.add_edge("completed", END)
    builder.add_edge("review_required", END)
    return builder.compile(checkpointer=checkpointer)

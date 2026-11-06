# LangGraph Workflow Foundation

## Phase 15: deterministic graph

The compiled graph runs document validation, student and scheme processing,
Test alignment, objective marking, subjective marking, and score aggregation.
It routes fully resolved results to `completed` and unresolved results to
`completed_with_review`. Routing is deterministic and does not ask the model to
choose workflow actions.

The existing `PaperMarkingService` remains the production entry point for the
CLI and Streamlit application. This graph is injectable and tested separately;
application adoption is deferred to a later integration phase.

## State

`app.graph.state.PaperMarkingState` is a `TypedDict` accepted by LangGraph's
`StateGraph`. It contains string file references and a run ID, pipeline
artifacts, exact Test alignment, objective/subjective/combined result models,
current stage/question, review flags, and safe warning/error records. Nodes
return partial updates using the same state schema. PDF and image bytes are not
copied into graph state.

The structured student and scheme artifacts can contain private OCR text.
Phase 16 adds PostgreSQL checkpointing; use it only with a private,
access-controlled database and the retention protections described below.

## Nodes

`app.graph.builder.build_marking_graph` wires these dependency-injected nodes in
order:

- validating both input paths;
- processing the student paper and marking scheme through existing pipelines;
- aligning Tests by exact number;
- running deterministic objective marking and guarded subjective marking;
- combining the two results with existing score-safety rules;
- routing the result to completed or review-required status.

Each node returns a partial state update and delegates business logic to the
existing service boundary. Missing prerequisite state raises an explicit
`ValueError`; service exceptions currently propagate to the graph caller and
are not converted to `FinalPaperMarkingResult` errors by this graph.

Unit tests invoke individual nodes with fakes and run the complete compiled
graph through both resolved and review-required outcomes.

## Phase 16: PostgreSQL checkpoint persistence

`app.graph.checkpoints.create_postgres_checkpointer` opens a
`PostgresSaver.from_conn_string` context, calls `setup()` to create or migrate
the LangGraph checkpoint tables, and closes the connection when the context
ends. It requires `DATABASE_URL` with a `postgresql://` or `postgres://` scheme.
The URL is not logged. The application database and checkpoint tables have
separate responsibilities even when hosted by the same PostgreSQL server.

Compile the graph with the saver and pass the run-specific configuration to
every invoke/resume operation:

```python
with create_postgres_checkpointer(settings) as checkpointer:
    graph = build_marking_graph(..., checkpointer=checkpointer)
    config = graph_config_for_run(run_id)
    result = graph.invoke(input_state, config=config)
```

`graph_config_for_run` uses the opaque run UUID as the checkpoint `thread_id`;
do not use student names, admission numbers, or other identifying values.
Checkpoints contain the full graph state, including student OCR and answer
text. PostgreSQL must therefore be access-controlled, protected at rest, and
covered by an explicit retention/deletion policy. Do not enable checkpointing
against an untrusted or shared database.

Tests use LangGraph's in-memory saver for graph behavior and mocks for
PostgreSQL connection setup. No PostgreSQL server or live database URL was
available for an integration test. The existing CLI and Streamlit entry points
still use `PaperMarkingService`; wiring persistent graph execution into those
entry points, retries, and application-level failure mapping remain future
phases.

## Phase 17: human-in-the-loop marking review

When aggregation cannot produce a final score, the graph routes to
`human_review`. The node raises a dynamic LangGraph interrupt containing the run
UUID and unresolved criteria, including student answer text, rubric
description, current status, and maximum marks. Criteria with no known maximum
are shown as non-assignable. This can include private student content; expose
the interrupt only to an authorized reviewer.

Resume the same run using its existing thread ID and a `Command(resume=...)`:

```python
config = graph_config_for_run(run_id)
paused = graph.invoke(input_state, config=config)

result = graph.invoke(
    Command(
        resume={
            "reviewer_id": "opaque-local-reviewer-id",
            "decisions": [
                {
                    "test_number": 14,
                    "criterion_id": "test-14-criterion-001",
                    "awarded_marks": "4",
                    "rationale": "Relevant response with minor language errors.",
                }
            ],
        }
    ),
    config=config,
)
```

`HumanReviewResponse` rejects extra fields and duplicate decisions.
`apply_human_review` accepts decisions only for unresolved criteria with a
known maximum and enforces `0 <= awarded_marks <= maximum_marks`. Unknown
criteria and over-maximum marks fail explicitly. A valid decision changes only
the final combined criterion result; original objective and subjective model
outputs remain unchanged for audit. The reviewer ID, rationale, and marks are
retained in graph state. The aggregate is recalculated from criterion results.
The graph reaches `completed` only when all marks are resolved; an empty or
partial response leaves the paper in `completed_with_review` with no fabricated
final score.

This phase provides the graph interrupt/resume contract, not a Streamlit review
screen or user authentication. Those belong to the later UI integration phase.
Pausing requires a checkpointer and a stable `thread_id`; use PostgreSQL for
persistent workflows. Unit tests use `InMemorySaver` and validate pause,
resume, final routing, score bounds, and preservation of unresolved results.
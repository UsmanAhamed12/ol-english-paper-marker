# AGENTS.md

## Project Architecture

                    O/L ENGLISH AI MARKING SYSTEM
             LangChain + LangGraph + Ollama + RAG

                         STREAMLIT
                            │
               ┌────────────┴────────────┐
               │                         │
        Student Answer PDF       Marking Scheme PDF
               │                         │
               └────────────┬────────────┘
                            ▼
                    LANGGRAPH WORKFLOW
                            │
                            ▼
                  [validate_documents]
                            │
                            ▼
                    [render_pages]
                            │
                            ▼
                       [run_ocr]
                            │
                            ▼
                  [validate_ocr]
                      │           │
                confident      uncertain
                      │           │
                      │           ▼
                      │     HUMAN REVIEW
                      │       interrupt()
                      │           │
                      └─────◄─────┘
                            │
                            ▼
                 [segment_questions]
                            │
                            ▼
                [ingest_marking_scheme]
                            │
                            ▼
                 LANGCHAIN DOCUMENTS
                            │
                            ▼
               Semantic/Question Chunking
                            │
                            ▼
                    Local Embeddings
                            │
                            ▼
                       ChromaDB
                            │
                            ▼
                 [retrieve_rubric]
                            │
                  metadata + similarity
                            │
                            ▼
                   [build_prompt]
                            │
                            ▼
               LangChain ChatOllama
                       Llama 2
                            │
                            ▼
                  structured output
                            │
                            ▼
                   [validate_grade]
                      │           │
                    valid     low confidence
                      │           │
                      │           ▼
                      │      HUMAN REVIEW
                      │       interrupt()
                      │           │
                      └────◄──────┘
                            │
                            ▼
                 [aggregate_scores]
                            │
                    Python arithmetic
                            │
                            ▼
                      SCORE /100
                            │
                            ▼
                     [persist_run]
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
          PostgreSQL                  Streamlit
              │
       papers / answers
       OCR / graph runs
       grading results
       evaluation data


        LANGGRAPH CHECKPOINTER
                 │
            PostgreSQL
                 │
     Resume / Retry / HITL / State

## 1. Project Identity

Project:

**O/L English Automated Paper Marking System**

This repository contains a local-first AI-assisted system for marking scanned Sri Lankan G.C.E. O/L English examination answer scripts.

The system accepts:

1. an unmarked scanned student answer-script PDF,
2. an official marking-scheme PDF,
3. optionally syllabus/reference material,

and produces:

- total mark out of 100,
- question-by-question marks,
- criterion-level marks,
- grading rationale,
- student feedback,
- OCR confidence,
- grading confidence,
- marking-scheme references,
- human-review warnings.

The application must be engineered as a maintainable AI software system, not as a notebook/demo.

---

# 2. Core Engineering Principle

Correctness, traceability and evaluation are more important than adding features quickly.

Never generate the entire application in one pass.

Development MUST proceed incrementally.

Every phase follows:

**Inspect → Plan → Implement → Test → Validate → Document → Stop**

Do not proceed to another phase until the current phase passes its acceptance criteria.

When a phase is complete, STOP and wait for:

`CONTINUE`

---

# 3. Development Environment

Primary development machine:

- macOS
- Apple Silicon / MacBook Air M1
- local development

Primary runtime:

- Python 3.12+

Package manager:

- uv

Do not use Poetry or Conda.

Do not use `pip` directly when `uv` can perform the operation.

Examples:

```bash
uv add package-name
uv add --dev pytest
uv sync
uv run pytest
uv run python ...
```

---

# 4. Technology Stack

Use the following technologies unless an Architecture Decision Record explicitly changes them.

## Language

Python 3.12+

## Package Management

uv

## AI Application Framework

LangChain

Use LangChain for:

- Ollama integration
- prompt templates
- document abstractions
- embeddings
- Chroma integration
- retriever composition
- structured model interaction where useful

Do NOT put the complete business architecture inside LangChain chains.

Core domain/business logic must remain framework-independent where practical.

## Workflow Orchestration

LangGraph

Use LangGraph for:

- marking workflow orchestration
- explicit workflow state
- conditional routing
- retries
- checkpointing
- resumability
- human-in-the-loop review
- failure recovery
- streaming workflow progress

Do NOT implement the marking process as an uncontrolled autonomous agent.

This application should primarily use a deterministic LangGraph workflow.

## Local LLM Runtime

Ollama

## Grading Model

Llama 2 through Ollama.

Access Ollama through the maintained LangChain Ollama integration where appropriate.

The grading model MUST be configurable.

Never scatter the literal model name throughout the codebase.

Use configuration such as:

`OLLAMA_GRADING_MODEL=llama2`

## OCR

OCR is a separate subsystem.

Llama 2 is NOT the OCR engine.

Create an OCR provider abstraction.

A local vision-capable model or another local OCR implementation may be used.

The OCR provider must be replaceable.

## Vector Database

ChromaDB.

Use the dedicated LangChain Chroma integration.

Chroma stores:

- marking-scheme embeddings,
- syllabus embeddings if required,
- retrieval metadata.

Chroma must NOT become the application's transactional database.

## Relational Database

PostgreSQL.

Use:

- SQLAlchemy 2.x
- Alembic

PostgreSQL stores application state and structured records.

## Frontend

Streamlit.

Streamlit is a presentation layer.

Business logic must NOT live inside Streamlit page files.

## Validation

Pydantic v2.

Use Pydantic models at application boundaries, especially for:

- OCR output,
- grading results,
- graph input/output,
- configuration,
- external model responses.

## Testing

pytest

pytest-cov where useful.

## Code Quality

Ruff

mypy

pre-commit may be introduced when useful.

---

# 5. Local-First Requirement

The project should run locally without requiring paid APIs.

Avoid introducing:

- paid LLM APIs,
- paid OCR APIs,
- unnecessary SaaS services,
- unnecessary cloud infrastructure.

Cloud deployment can be considered later.

Local development is the primary target.

---

# 6. High-Level Architecture

Maintain the following architectural boundaries:

```text
Streamlit
    │
    ▼
Application Services
    │
    ▼
LangGraph Workflow
    │
    ├── PDF ingestion
    ├── OCR
    ├── OCR validation
    ├── question segmentation
    ├── marking-scheme ingestion
    ├── retrieval
    ├── grading
    ├── grade validation
    ├── human review
    ├── aggregation
    └── persistence
           │
     ┌─────┴─────┐
     ▼           ▼
PostgreSQL     ChromaDB
```

LangChain integrations live behind appropriate service boundaries.

---

# 7. Target Repository Architecture

Use this as the target architecture.

Do NOT create every empty file immediately.

Files/directories should appear when required by a development phase.

```text
ol-english-paper-marker/
│
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .env.example
├── .gitignore
├── docker-compose.yml
│
├── app/
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── exceptions.py
│   │   └── constants.py
│   │
│   ├── domain/
│   │   ├── models/
│   │   ├── schemas/
│   │   └── enums/
│   │
│   ├── ingestion/
│   │   ├── pdf_loader.py
│   │   ├── pdf_renderer.py
│   │   ├── image_processor.py
│   │   └── validators.py
│   │
│   ├── ocr/
│   │   ├── base.py
│   │   ├── service.py
│   │   ├── preprocessing.py
│   │   ├── normalizer.py
│   │   └── providers/
│   │
│   ├── segmentation/
│   │   ├── question_segmenter.py
│   │   └── models.py
│   │
│   ├── marking_scheme/
│   │   ├── parser.py
│   │   ├── chunker.py
│   │   ├── models.py
│   │   └── service.py
│   │
│   ├── embeddings/
│   │   └── service.py
│   │
│   ├── vectorstore/
│   │   ├── base.py
│   │   └── chroma_store.py
│   │
│   ├── retrieval/
│   │   ├── retriever.py
│   │   ├── filters.py
│   │   └── models.py
│   │
│   ├── llm/
│   │   ├── factory.py
│   │   └── models.py
│   │
│   ├── grading/
│   │   ├── prompt.py
│   │   ├── schemas.py
│   │   ├── grader.py
│   │   ├── validator.py
│   │   └── aggregator.py
│   │
│   ├── graph/
│   │   ├── state.py
│   │   ├── builder.py
│   │   ├── routing.py
│   │   ├── checkpoints.py
│   │   └── nodes/
│   │       ├── validate_documents.py
│   │       ├── render_pages.py
│   │       ├── run_ocr.py
│   │       ├── validate_ocr.py
│   │       ├── segment_questions.py
│   │       ├── ingest_marking_scheme.py
│   │       ├── retrieve_rubric.py
│   │       ├── grade_question.py
│   │       ├── validate_grade.py
│   │       ├── human_review.py
│   │       ├── aggregate_scores.py
│   │       └── persist_results.py
│   │
│   ├── database/
│   │   ├── base.py
│   │   ├── session.py
│   │   ├── models/
│   │   └── repositories/
│   │
│   ├── services/
│   │   ├── marking_service.py
│   │   ├── document_service.py
│   │   └── evaluation_service.py
│   │
│   ├── evaluation/
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   ├── dataset.py
│   │   └── reports.py
│   │
│   └── ui/
│       ├── Home.py
│       ├── pages/
│       ├── components/
│       └── state.py
│
├── prompts/
│   ├── grading/
│   ├── ocr/
│   └── segmentation/
│
├── data/
│   ├── raw/
│   │   └── marked_papers/
│   ├── evaluation/
│   ├── processed/
│   ├── runtime/
│   └── chroma/
│
├── scripts/
│   ├── check_environment.py
│   ├── inspect_dataset.py
│   ├── ingest_marking_scheme.py
│   ├── evaluate.py
│   └── run_sample.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   ├── fixtures/
│   └── conftest.py
│
├── alembic/
│   └── versions/
│
└── docs/
    ├── architecture.md
    ├── setup.md
    ├── development.md
    ├── data.md
    ├── ocr.md
    ├── rag.md
    ├── langgraph.md
    ├── grading.md
    ├── evaluation.md
    ├── testing.md
    └── decisions/
```

---

# 8. LangGraph Architecture

The complete paper-marking workflow must eventually be represented as an explicit LangGraph workflow.

Target conceptual graph:

```text
START
  │
  ▼
validate_documents
  │
  ▼
render_pages
  │
  ▼
run_ocr
  │
  ▼
validate_ocr
  │
  ├── low confidence ──→ human_ocr_review
  │                         │
  │                         ▼
  │                       resume
  │                         │
  └─────────────────────────┘
  │
  ▼
segment_questions
  │
  ▼
ingest_marking_scheme
  │
  ▼
retrieve_rubric
  │
  ▼
grade_questions
  │
  ▼
validate_grades
  │
  ├── low confidence ──→ human_grading_review
  │                         │
  │                         ▼
  │                       resume
  │                         │
  └─────────────────────────┘
  │
  ▼
aggregate_scores
  │
  ▼
persist_results
  │
  ▼
END
```

Use conditional edges for routing.

The graph should remain deterministic.

Do not give the LLM control over the overall workflow.

---

# 9. LangGraph State

Graph state must be explicit and typed.

Do not use a giant unstructured dictionary.

Create a state model appropriate for LangGraph.

Conceptually it may contain:

```python
paper_id
run_id

student_pdf_path
marking_scheme_path

pages
ocr_results
questions
rubric_chunks
retrieval_results
question_grades

current_question
current_stage

ocr_review_required
grading_review_required

warnings
errors

final_score
```

Do not place heavyweight binary PDF/image contents directly into persistent graph state when file references are sufficient.

Persist references/identifiers where possible.

---

# 10. LangGraph Node Rules

Every graph node should perform one clear responsibility.

Good:

`run_ocr`

`retrieve_rubric`

`grade_question`

`aggregate_scores`

Bad:

`process_everything`

Nodes should call application/domain services.

Do not implement hundreds of lines of business logic directly inside node functions.

Conceptually:

```python
def run_ocr_node(state, ocr_service):
    result = ocr_service.process(...)
    return {"ocr_results": result}
```

Nodes should return state updates rather than mutate arbitrary global state.

---

# 11. LangGraph Human-in-the-Loop

Human review is a core feature, not an afterthought.

Use LangGraph interrupts where appropriate.

Possible reasons:

```text
OCR confidence below threshold
missing question number
ambiguous handwriting
retrieval produced no suitable rubric
grading confidence below threshold
invalid model response after retries
score anomaly
```

Example workflow:

```text
grade_question
       ↓
confidence = 0.41
       ↓
interrupt
       ↓
Streamlit review screen
       ↓
teacher accepts/edits result
       ↓
resume graph
```

Do not silently invent a mark when confidence is too low.

---

# 12. LangGraph Persistence

During unit tests or very early development, an in-memory checkpointer may be used.

For persistent application workflows, use PostgreSQL-backed LangGraph checkpointing when introduced in the appropriate phase.

Graph execution should eventually survive:

- UI refresh,
- process interruption,
- human-review pauses,
- recoverable failures.

Do not confuse:

**application database records**

with

**LangGraph checkpoints**.

They serve different responsibilities even if both use PostgreSQL.

---

# 13. LangChain Rules

Use modern dedicated integration packages.

Prefer package-specific integrations instead of old/deprecated monolithic imports.

Examples include:

```python
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
```

Before introducing a LangChain/LangGraph API, check the installed version and current official documentation.

Do NOT copy old tutorials without verification.

Avoid deprecated imports.

---

# 14. LLM Factory

Centralize model construction.

Do not write:

```python
ChatOllama(...)
```

inside random services.

Create something similar to:

```text
app/llm/factory.py
```

Application code should request the configured grading model from this layer.

Configuration should determine:

```text
model
temperature
timeout
retry behavior
base URL
```

Use deterministic settings where grading requires reproducibility.

---

# 15. Structured LLM Output

Never trust raw model prose.

Grading must produce a structured domain result.

Example:

```python
class CriterionScore(BaseModel):
    criterion: str
    awarded: float
    maximum: float
    reason: str


class QuestionGrade(BaseModel):
    question_number: str
    awarded_marks: float
    maximum_marks: float
    criteria: list[CriterionScore]
    feedback: str
    confidence: float
    requires_human_review: bool
```

All LLM output must be validated.

Never directly write raw LLM output into final scoring logic.

---

# 16. Score Safety

Always enforce:

```text
0 <= awarded <= maximum
```

The LLM does NOT calculate the final paper total.

Python application code performs aggregation.

Correct:

```python
total = sum(grade.awarded_marks for grade in grades)
```

Incorrect:

```text
"LLM, calculate the student's final mark."
```

The official exam structure determines the maximum possible score.

Do not normalize unless the actual marking scheme requires normalization.

---

# 17. OCR Architecture

OCR must remain independent from grading.

Required abstraction:

```python
class OCRProvider(Protocol):
    def extract_page(self, ...) -> OCRPageResult:
        ...
```

OCR results should contain where practical:

```text
page_number
raw_text
normalized_text
confidence
provider
model/version
warnings
```

Preserve raw OCR output.

Never replace raw OCR with corrected text.

---

# 18. OCR Evaluation

Handwritten answer scripts are difficult.

OCR accuracy must be evaluated separately from grading accuracy.

Do not conclude:

`AI grading is wrong`

until determining whether the failure came from:

```text
OCR
segmentation
retrieval
grading
aggregation
```

Evaluation reports should classify errors by pipeline stage where possible.

---

# 19. Question Segmentation

OCR output must be transformed into structured student answers.

Conceptual model:

```python
StudentAnswer(
    question_number="14",
    sub_question="a",
    text="...",
    page_numbers=[7],
    ocr_confidence=0.84,
)
```

Support answers spanning multiple pages.

Do not assume one page equals one question.

---

# 20. Marking-Scheme RAG

The marking scheme is structured information.

Do NOT blindly split it by character count.

Prefer chunks corresponding to:

```text
Paper
Section
Question
Sub-question
Criterion
```

Each chunk should have metadata.

Example:

```json
{
  "document_type": "marking_scheme",
  "year": 2025,
  "paper": "English Language II",
  "question_number": "14",
  "sub_question": "a",
  "page_number": 4,
  "maximum_marks": 10
}
```

---

# 21. Retrieval Strategy

Do not rely only on semantic similarity.

Preferred retrieval:

```text
student question
       ↓
metadata filter
       ↓
question-specific candidate chunks
       ↓
semantic retrieval
       ↓
top-k relevant rubric chunks
```

Question 14 should retrieve Question 14 rubric material whenever structured metadata makes that possible.

Prevent irrelevant marking criteria from leaking between questions.

---

# 22. ChromaDB Rules

Chroma is used for vector retrieval only.

Store:

```text
embedding
text
document ID
question metadata
page metadata
rubric metadata
```

Do not store transactional application state in Chroma.

Chroma persistence path must be configurable.

---

# 23. PostgreSQL Rules

PostgreSQL is the system of record for structured application data.

Potential entities include:

```text
papers
paper_pages
student_scripts
student_answers
ocr_results
marking_schemes
rubric_chunks
grading_runs
question_grades
criterion_scores
final_scores
human_reviews
evaluation_runs
evaluation_metrics
```

Do not create all tables before they are needed.

Use migrations.

Never manually modify production schema.

---

# 24. Repository Pattern

Database operations should be isolated.

UI code must never contain raw SQL.

LangGraph nodes must not contain raw SQL.

Services should interact with repositories.

Example:

```text
GradingService
      ↓
GradingResultRepository
      ↓
SQLAlchemy
      ↓
PostgreSQL
```

---

# 25. Dataset Rules

Raw historical papers are immutable.

Expected location:

```text
data/raw/marked_papers/
```

Never:

- overwrite,
- rename without explicit instruction,
- modify,
- annotate in-place,
- delete

raw dataset documents.

Generated artifacts belong under processed/evaluation/runtime locations.

---

# 26. Evaluation Data Leakage

The teacher's existing marks are ground truth.

They MUST remain hidden from the grading model during evaluation.

Correct:

```text
student answer
     ↓
AI grading
     ↓
freeze prediction
     ↓
teacher score revealed
     ↓
comparison
```

Never include teacher annotations/scores in the LLM grading context when measuring AI-vs-human performance.

Create architectural separation between:

```text
runtime grading data
calibration examples
evaluation ground truth
```

---

# 27. Evaluation Metrics

The project must eventually calculate:

- MAE
- RMSE
- mean signed error/bias
- percentage within ±1 mark
- percentage within ±2 marks
- percentage within ±5 marks
- per-question MAE
- OCR quality metrics where ground truth exists
- retrieval success metrics
- human-review rate

Correlation may be reported as a secondary metric.

Do not use correlation alone as evidence of marking accuracy.

---

# 28. Prompt Management

Do not hide giant prompts inside Python modules.

Reusable/versioned prompts should live under:

```text
prompts/
```

Prompts should have identifiable versions.

Store the prompt version used for each grading run.

Changing a grading prompt must be treated as an evaluation-relevant change.

---

# 29. Grading Prompt Rules

A grading prompt should contain only the information required to mark the current question:

```text
question
student response
maximum marks
official rubric
grading criteria
output schema
```

Do not provide the entire exam when grading one question unless explicitly required.

Do not provide historical teacher marks during evaluation.

---

# 30. Hallucination Protection

The model must not create grading criteria that do not exist.

If no valid marking criteria are retrieved:

do NOT guess.

Return an appropriate failure/review state.

Examples:

```text
RUBRIC_NOT_FOUND

HUMAN_REVIEW_REQUIRED
```

---

# 31. Confidence

Confidence must not be presented as mathematically calibrated probability unless it has actually been calibrated.

Treat model-provided confidence as a workflow signal.

Use deterministic checks alongside model confidence.

Examples:

```text
OCR confidence
retrieval quality
schema validity
rubric availability
score bounds
```

---

# 32. Error Handling

Create meaningful domain exceptions.

Examples:

```text
InvalidPDFError
OCRProcessingError
QuestionSegmentationError
RubricNotFoundError
RetrievalError
InvalidGradeError
ModelUnavailableError
PersistenceError
```

Do not catch `Exception` everywhere and silently continue.

Fail explicitly when correctness cannot be guaranteed.

---

# 33. Retry Policy

Retries are appropriate for transient failures.

Possible examples:

- temporary Ollama connection error,
- transient database connection issue.

Retries are NOT a solution for:

- invalid architecture,
- consistently malformed prompts,
- missing marking criteria,
- corrupted PDFs.

Retries must be bounded.

---

# 34. Logging

Use structured logging.

Do not use production `print()` calls.

Useful context:

```text
run_id
paper_id
question_number
page_number
graph_node
ocr_provider
llm_model
prompt_version
duration_ms
status
```

Never log sensitive student information unnecessarily.

---

# 35. Configuration

Use environment-driven configuration with Pydantic settings.

Potential configuration:

```text
APP_ENV
DATABASE_URL

OLLAMA_BASE_URL
OLLAMA_GRADING_MODEL
OLLAMA_OCR_MODEL

CHROMA_PERSIST_DIR
CHROMA_COLLECTION

DATA_DIR

OCR_CONFIDENCE_THRESHOLD
GRADING_CONFIDENCE_THRESHOLD

RETRIEVAL_TOP_K

LOG_LEVEL
```

Provide `.env.example`.

Never commit `.env`.

Never commit passwords or secrets.

---

# 36. Streamlit Rules

Streamlit is the presentation layer.

Streamlit may:

- upload files,
- display workflow progress,
- display results,
- collect human-review decisions,
- show evaluation metrics.

Streamlit must NOT:

- perform raw SQL,
- directly call Chroma,
- construct grading prompts,
- directly call Ollama,
- calculate business scoring logic.

Use services.

---

# 37. Final Streamlit Experience

Target navigation:

```text
Dashboard

Mark Paper

Results

Human Review

Evaluation

About
```

Main flow:

```text
Upload Student Paper
       +
Upload Marking Scheme
       ↓
Validate
       ↓
Start Marking
       ↓
OCR progress
       ↓
Question detection
       ↓
RAG retrieval
       ↓
AI grading
       ↓
Human review if required
       ↓
Final Results
```

Results should clearly display:

```text
Final Score

72 / 100

Question Breakdown

Q1     4 / 5
Q2     3 / 5
...
Q16   12 / 15
```

Each question should expose:

- student transcription,
- mark,
- maximum mark,
- rubric/criteria,
- rationale,
- feedback,
- confidence/review status.

---

# 38. UI Quality

Create a clean academic/professional interface.

Prefer native Streamlit functionality.

Use custom CSS only where it materially improves usability.

Do not build a flashy dashboard that hides grading uncertainty.

Accuracy and clarity are more important than decoration.

---

# 39. Testing Architecture

Maintain:

```text
tests/
├── unit/
├── integration/
├── e2e/
└── fixtures/
```

Unit tests:

- fast,
- deterministic,
- isolated.

Do not require live Ollama/PostgreSQL/Chroma unless explicitly testing an integration.

Integration tests can cover real infrastructure.

E2E tests cover complete workflows.

---

# 40. LangGraph Testing

Test graph nodes independently.

Test routing separately.

Required eventual routing tests include:

```text
valid OCR
→ segmentation
```

```text
low OCR confidence
→ human review
```

```text
valid grade
→ aggregation
```

```text
invalid grade
→ retry/review
```

```text
missing rubric
→ review/error
```

Do not rely only on a full E2E graph test.

---

# 41. Model Testing

Unit tests must mock LLM responses.

Do not make every pytest run execute Llama 2.

Separate tests using markers where appropriate:

```text
unit

integration

ollama

slow

e2e
```

Normal development tests should remain fast.

---

# 42. Required Quality Commands

At the end of each appropriate phase run:

```bash
uv run pytest
uv run ruff check .
uv run mypy app
```

Where formatting is configured:

```bash
uv run ruff format --check .
```

Do not claim success without actually running the commands.

---

# 43. Documentation

Documentation is part of implementation.

Maintain:

```text
README.md

docs/architecture.md
docs/setup.md
docs/development.md
docs/data.md
docs/ocr.md
docs/rag.md
docs/langgraph.md
docs/grading.md
docs/evaluation.md
docs/testing.md
```

Documentation must reflect the real implementation.

Never document features that do not exist as completed features.

Clearly label planned features.

---

# 44. Architecture Decision Records

Important decisions should use ADRs.

Location:

```text
docs/decisions/
```

Examples:

```text
ADR-001-use-uv.md
ADR-002-use-langchain.md
ADR-003-use-langgraph.md
ADR-004-use-ollama.md
ADR-005-use-postgresql.md
ADR-006-use-chromadb.md
ADR-007-separate-ocr-from-grading.md
```

ADR format:

```text
Title

Status

Context

Decision

Alternatives Considered

Consequences
```

---

# 45. Dependency Discipline

Before adding a package:

1. determine why it is required,
2. check whether an existing dependency already solves the problem,
3. use a maintained package,
4. avoid deprecated integrations,
5. add it through uv.

Do not install libraries merely because tutorials use them.

Keep dependencies minimal.

---

# 46. Latest API Rule

LangChain and LangGraph evolve quickly.

Before implementing unfamiliar or potentially changed APIs:

1. inspect installed package versions,
2. consult current official documentation when available,
3. prefer current package-specific imports,
4. avoid deprecated APIs,
5. record significant compatibility decisions.

Do not blindly reproduce old LangChain tutorials.

---

# 47. Code Quality

Write code appropriate for a developer with roughly two years of professional Python experience.

Code should be:

- readable,
- typed,
- testable,
- modular,
- explicit,
- maintainable.

Do not over-engineer.

Avoid:

- unnecessary factories,
- unnecessary inheritance,
- god classes,
- 1,000-line modules,
- giant `utils.py`,
- global mutable state,
- duplicated logic,
- deeply nested control flow.

---

# 48. Dependency Direction

Prefer dependency flow:

```text
UI
 ↓
Application Services
 ↓
Domain
 ↑
Infrastructure adapters
```

Domain logic should not depend directly on Streamlit.

Core scoring rules should not depend directly on LangChain.

This allows framework changes without rewriting the entire project.

---

# 49. Git Discipline

Keep changes focused.

Suggested commit style:

```text
chore: initialize uv project

feat: add PDF ingestion pipeline

feat: add OCR provider abstraction

feat: add question segmentation

feat: add marking scheme ingestion

feat: add Chroma retrieval

feat: add LangGraph marking workflow

feat: add structured Llama2 grader

feat: add human review workflow

feat: add evaluation pipeline
```

Never commit:

```text
.env
raw student data unnecessarily
database passwords
generated runtime uploads
local Chroma data
```

---

# 50. Development Phases

Follow this sequence.

## Phase 0

Repository and dataset inspection.

## Phase 1

Project foundation and tooling.

## Phase 2

PDF ingestion.

## Phase 3

OCR provider architecture.

## Phase 4

OCR baseline and evaluation.

## Phase 5

Question segmentation.

## Phase 6

PostgreSQL + SQLAlchemy + Alembic.

## Phase 7

Marking-scheme parser.

## Phase 8

Question-aware chunking.

## Phase 9

Embeddings + Chroma.

## Phase 10

Retriever.

## Phase 11

LangChain + Ollama integration.

## Phase 12

Structured question grader.

## Phase 13

Score validation and aggregation.

## Phase 14

LangGraph state and nodes.

## Phase 15

LangGraph complete workflow.

## Phase 16

LangGraph checkpoint persistence.

## Phase 17

Human-in-the-loop review.

## Phase 18

One-paper E2E pipeline.

## Phase 19

Historical evaluation framework.

## Phase 20

40-paper evaluation.

## Phase 21

Error analysis and calibration.

## Phase 22

Streamlit UI.

## Phase 23

Streamlit + LangGraph HITL integration.

## Phase 24

Complete E2E tests.

## Phase 25

Performance/reliability improvements.

## Phase 26

Security/privacy hardening.

## Phase 27

Final documentation.

## Phase 28

Release candidate.

Do NOT skip directly to UI.

---

# 51. Phase Execution Protocol

At the beginning of EVERY phase:

1. Read this `AGENTS.md`.
2. Inspect current repository state.
3. Read relevant existing documentation.
4. Run the existing test suite.
5. Identify exactly what the current phase requires.
6. Present a short implementation plan.

Then implement ONLY that phase.

---

# 52. Phase Completion Protocol

Before declaring a phase complete:

1. implementation must exist,
2. tests must exist,
3. tests must pass,
4. lint must pass,
5. type checking must pass where configured,
6. relevant documentation must be updated,
7. no unrelated refactoring,
8. acceptance criteria must be checked.

Then report:

```text
PHASE <number> COMPLETION REPORT

Implemented:
- ...

Files created:
- ...

Files modified:
- ...

Tests:
- X passed
- 0 failed

Quality:
- Ruff: PASS
- mypy: PASS

Documentation:
- ...

Acceptance criteria:
[x] ...
[x] ...
[x] ...

Known limitations:
- ...

Suggested commit:
<commit message>

STATUS: PHASE <number> COMPLETE

Waiting for CONTINUE.
```

---

# 53. Failure Protocol

If tests fail:

do NOT proceed.

If lint fails:

do NOT proceed.

If type checking fails because of newly introduced code:

do NOT proceed.

If acceptance criteria are incomplete:

do NOT proceed.

Report:

```text
STATUS: PHASE <number> NOT COMPLETE
```

Fix the phase.

---

# 54. No Fake Success

Never write:

`All tests pass`

unless tests were actually executed.

Never write:

`Ollama works`

unless the relevant health/integration check was actually executed.

Never write:

`PostgreSQL works`

unless connectivity was actually tested.

Never write:

`OCR is accurate`

without evaluation evidence.

Never write:

`AI grading is accurate`

without evaluation evidence.

---

# 55. Definition of Done

The project is not complete merely when the Streamlit interface produces a number.

The final system should demonstrate:

```text
PDF ingestion
✓

OCR
✓

Question segmentation
✓

Marking-scheme ingestion
✓

Question-aware RAG
✓

Structured grading
✓

Score validation
✓

Deterministic aggregation
✓

LangGraph orchestration
✓

Persistence
✓

Human review
✓

PostgreSQL
✓

ChromaDB
✓

Streamlit
✓

Automated tests
✓

Historical evaluation
✓

Documentation
✓
```

---

# 56. Final Accuracy Principle

The objective is NOT:

"make Llama 2 give marks."

The objective is:

**Build an auditable AI-assisted examination marking pipeline whose predictions can be compared objectively against human teacher markings.**

The system must make it possible to answer:

- What text did OCR extract?
- Which question was detected?
- Which marking criteria were retrieved?
- Which model produced the grade?
- Which prompt version was used?
- Why was each mark awarded?
- Was human review required?
- How close was the AI mark to the teacher's mark?

Traceability is mandatory.

---

# 57. Current Instruction to Codex

If this repository is newly initialized, begin with:

**PHASE 0 — Repository and Dataset Inspection**

Do NOT build the application yet.

Inspect:

```text
data/raw/marked_papers/
```

Determine:

- number of PDFs,
- page count distribution,
- whether PDFs contain extractable text or scanned images,
- image dimensions/resolution where practical,
- representative paper structure,
- handwriting characteristics,
- teacher-annotation characteristics,
- observable question numbering,
- potential OCR challenges.

Inspect at least three representative PDFs.

Do not modify raw PDFs.

Create/update:

```text
docs/data.md
```

Record findings.

If useful, create:

```text
scripts/inspect_dataset.py
```

but keep Phase 0 minimal.

At the end provide the Phase 0 completion report and STOP.

Wait for:

`CONTINUE`
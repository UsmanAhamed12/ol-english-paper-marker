# Local RAG knowledge base

## Current scope

The local Grade 10 English syllabus is extracted from its selectable PDF text,
split into page-bounded chunks, embedded through Ollama, and stored in the
persistent Chroma collection configured by `CHROMA_PERSIST_DIR` and
`CHROMA_COLLECTION_NAME`. Each chunk records its source type, stable source ID,
source SHA-256, source filename, grade, subject, page, and chunk position.

The Chroma adapter supports metadata-filtered similarity search and source
replacement. Re-indexing the same syllabus is idempotent; replacing it removes
obsolete chunks only after the new chunks have been added. PDF pages without
extractable text are skipped, and a document with no extractable text fails
explicitly rather than silently producing an empty index.

The existing marking-scheme ingestion remains a separate OCR-backed pipeline.
Syllabus retrieval is not yet wired into marking prompts or the Streamlit
workflow, and no claim about grading accuracy is made from indexing alone.

## Index the syllabus

Install/start Ollama and pull the configured embedding model:

```bash
ollama pull nomic-embed-text
```

Then, from the project root:

```bash
uv run python -m scripts.ingest_syllabus
```

The command uses `SYLLABUS_PDF_PATH` by default. A different local PDF may be
provided as the positional argument. The index is written under `data/chroma/`,
which is local runtime data and must not be committed. The command reports page
and chunk counts, the source hash, collection, persistence path, and embedding
model; it does not print syllabus contents.

The embedding model is configurable with `OLLAMA_EMBEDDING_MODEL`. The Chroma
collection and persistence path are configurable with `CHROMA_COLLECTION_NAME`
and `CHROMA_PERSIST_DIR`.

## Retrieval contract

Application code uses `ChromaKnowledgeBaseStore.search` with a query, `k`, and
optional metadata filters such as `{"source_type": "syllabus"}`. Future
question-aware retrieval should combine marking-scheme chunks and syllabus
chunks, constrain the scheme by exact Test/question metadata where available,
and preserve the returned source/page metadata in grading results. Missing or
irrelevant rubric evidence must continue to trigger review rather than guessed
criteria.
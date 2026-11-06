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

Parsed marking schemes now have a separate question-aware chunking boundary.
Each parsed criterion becomes an independently filterable chunk with exact
Test, criterion, source-page, and mark metadata. A Test without criteria keeps
its source lines in a Test-level fallback chunk. They can now be embedded and
indexed in the same Chroma collection as the syllabus, preserving exact Test
and scheme metadata. Retrieval is not yet connected to grading.

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

## Index a marking scheme

The marking-scheme command remains parse-only by default. To also create local
Ollama embeddings and add its question-aware chunks to Chroma, pull the
configured embedding model and pass `--index`:

```bash
ollama pull nomic-embed-text
uv run python -m scripts.ingest_marking_scheme /private/path/to/scheme.pdf --index
```

Indexing reports only safe counts and identifiers; OCR text and chunk content
remain in local runtime storage. Scheme chunks use their scheme ID as the
replacement key, so re-indexing the same scheme replaces stale chunks without
changing syllabus records. This phase provides indexing only; exact-Test
retrieval and use in grading prompts are later work.

## Retrieval contract

`MarkingSchemeRetriever.retrieve` accepts a query, scheme ID, exact Test number,
and `k`. It applies metadata filters for `source_type`, `source_id`, `scheme_id`,
and numeric `test_number` before Chroma ranks semantic matches. Returned chunks
are checked again for matching scheme/Test identity and valid hash, criterion,
and page provenance, then returned as typed `MarkingSchemeEvidence` records.

No matches return an empty tuple. Callers must treat missing rubric evidence as
a review condition, never as permission to guess criteria. The retriever is
implemented but is not yet wired into grading prompts or the Streamlit marking
workflow. Syllabus searches continue to use their separate
`source_type="syllabus"` and stable syllabus `source_id` filters.
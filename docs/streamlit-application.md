# Streamlit FYP application

## Status and scope

Phase 7 adds a local Streamlit presentation layer to the accepted Phase 6
orchestrator. It is a final-year project prototype, not a claim of validated
official examination-marking accuracy. The UI contains no OCR, alignment,
scheme parsing, or marking rules.

```text
Streamlit upload and presentation
              |
              v
     PaperMarkingService
              |
              v
   FinalPaperMarkingResult
              |
              v
 deterministic ResultViewModel
              |
              v
 summary / Tests / feedback / review / download
```

## Prerequisites and installation

Use Python 3.12 and install the locked environment:

```bash
uv sync
```

Real processing also requires:

- local Tesseract with the configured `eng` language data;
- a running local Ollama endpoint at `OLLAMA_BASE_URL`;
- the configured `OLLAMA_GRADING_MODEL` installed locally;
- one student answer PDF and its corresponding official marking-scheme PDF.

The application never installs system software automatically. Missing
Tesseract is shown as a safe setup error. Ollama/model failures remain failed or
review-required backend outcomes; the UI never substitutes a cloud model.

## Launch

From the repository root:

```bash
uv run streamlit run streamlit_app.py
```

Streamlit serves the application on `127.0.0.1`. Repository configuration
disables Streamlit usage-stat telemetry; no external file service is configured.

## Upload workflow and privacy

The page accepts exactly two PDFs. Uploaded bytes are validated for extension,
content type, size, and PDF signature before the existing production validator
runs. Each click creates a UUID run directory beneath ignored
`data/runtime/streamlit-uploads/` and writes only:

```text
student-paper.pdf
marking-scheme.pdf
```

Original filenames are never used for materialized paths. Writes use a
same-directory temporary file and atomic replacement. Upload copies are removed
after the synchronous service call and on reset where applicable. The validated
`FinalPaperMarkingResult` remains in the current Streamlit session only unless
the user explicitly downloads it.

Private uploaded PDFs, OCR text, model responses, and marking results are not
globally cached. The page calls `PaperMarkingService` exactly once per explicit
Mark Paper action and disables another run until Mark Another Paper resets the
session.

## Processing and results

Processing is sequential. The active status explains that OCR and local model
marking can take several minutes on the M1; it does not invent percentages.
After completion, measured backend stage statuses and durations are available
in an expander.

The result mapper preserves Phase 6 score semantics:

- a **Final score** and percentage appear only for `COMPLETED` with no
  unresolved marks;
- otherwise the page says **Confirmed marks**, displays unresolved marks, and
  suppresses a definitive percentage;
- `PARTIALLY_COMPLETED` retains every usable Test result;
- `FAILED` displays only safe mapped errors and no traceback.

Each Test expander shows confirmed, maximum, unresolved, objective, and
subjective marks plus its criterion decisions and existing criterion-grounded
feedback. The review queue shows the Test, criterion, reason, status, and marks
affected. No additional model call summarizes or rewrites feedback.

The JSON download uses the Phase 6 deterministic serializer. It can contain
private answer/marking evidence and must be protected by the user.

## Synthetic demo mode

There is currently no official marking-scheme PDF in repository private
storage. Presentation QA can be enabled explicitly:

```bash
UI_DEMO_ENABLED=true uv run streamlit run streamlit_app.py
```

The Developer / Synthetic Demo expander provides completed, review-required,
partial, and failed synthetic results. Every synthetic view displays:

> Synthetic demonstration result — not produced from a real examination paper.

Demo mode never activates automatically and never becomes a fallback after a
real processing failure.

## Phase 7 verification

Synthetic, private-data-free QA covered the desktop upload layout, one- and
two-file readiness states, active processing status, completed, review-required,
partially completed, and failed results, Test/criterion expanders, feedback,
the review queue, JSON download, and reset. A separate synthetic-PDF handoff
reached the real `PaperMarkingService`; its deliberately unusable scheme was
reported as a safe scheme-ingestion failure, and temporary upload copies were
removed. This verifies application wiring, not marking accuracy.

**REAL END-TO-END SMOKE BLOCKED: NO OFFICIAL MARKING-SCHEME PDF AVAILABLE.**

## Known limitations

- A real official marking-scheme smoke has not been completed.
- Streamlit execution is synchronous; a browser refresh cannot resume a run.
- Local Tesseract/Ollama processing may take several minutes.
- Review items are displayed but not editable in Phase 7.
- Results are not persisted to a database.
- OCR, evidence separation, and subjective marking retain their documented FYP
  limitations.

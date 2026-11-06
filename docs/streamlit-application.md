# Streamlit FYP application

## Status and scope

The Streamlit interface offers the local marking pipeline and a Groq vision
marking path. The project is a prototype, not a claim of validated official
examination-marking accuracy.

```text
Streamlit upload and presentation
              |
              v
       ┌──────┴───────┐
       v              v
 Local marking    Groq vision
 pipeline         transcription + grading
       |              |
       └──────┬───────┘
              v
      question-level results
              |
      score / 100 + review
```

 ## Navigation

 The application provides five focused pages:

 - **Overview** — current browser-session status and direct links to the main tasks.
 - **Mark a Paper** — upload both PDFs, choose an engine, and explicitly start.
 - **Results** — question marks, explanations, improvement tips, OCR text, and a
   JSON export.
 - **Review Queue** — items flagged for examiner attention; review is read-only.
 - **About & Readiness** — configured local runtime values, data handling, and
   known limitations.

 Results and review items are shared across pages through the active Streamlit
 session. They are not a persistent run history and are unavailable after that
 session ends.

 ## Prerequisites and installation

Use Python 3.12 and install the locked environment:

```bash
uv sync
```

Local processing requires:

- local Tesseract with the configured `eng` language data;
- a running local Ollama endpoint at `OLLAMA_BASE_URL`;
- the configured `OLLAMA_GRADING_MODEL` installed locally;
- one student answer PDF and its corresponding official marking-scheme PDF.

Groq processing requires a Groq API key in `.env` and a vision-capable model
enabled for the account. The application never installs system software
automatically. It does not silently switch between local and cloud providers.

## Launch

From the repository root:

```bash
uv run python -m streamlit run streamlit_app.py
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
after the synchronous service call and on reset where applicable. Results
remain in the current Streamlit session only unless the user explicitly
downloads them.

Private uploaded PDFs, OCR text, model responses, and marking results are not
globally cached. Groq mode displays an explicit cloud-processing notice and
requires per-run confirmation before sending either PDF to Groq. The local mode
continues to use the existing on-device pipeline.

## Processing and results

Processing is sequential. Groq mode renders each PDF page to a bounded JPEG,
sends it to the configured vision model for separate printed/handwritten
transcription, then asks the same model to grade the full paper against the
extracted scheme. The application validates every question score and computes
the total and normalized score out of 100 in Python. If rubric maxima are
missing or a question is flagged, the result is marked provisional or the
normalized score is withheld. Local mode uses the existing pipeline.

The interface uses a restrained academic visual style, grouped page navigation,
a compact local-workspace sidebar, page-specific headings, a four-step
workflow guide, paired upload cards, and consistent score and review panels.
It uses native Streamlit controls and inline styling only; no remote fonts,
images, or frontend packages are loaded. Visual styling does not change
marking, confidence, or review decisions. Surfaces and accents adapt to
Streamlit's System, Light, and Dark appearance choices; use the top-right
Streamlit menu's **Theme** options to switch.

Local results preserve the existing score semantics:

- a **Normalized score out of 100** appears only for `COMPLETED` with no
  unresolved marks; the original scheme score is also shown;
- otherwise the page says **Confirmed marks**, displays unresolved marks, and
  suppresses a definitive normalized score;
- `PARTIALLY_COMPLETED` retains every usable Test result;
- `FAILED` displays only safe mapped errors and no traceback.

Groq results show each question's transcription, awarded and maximum marks,
explanation, feedback, and improvement tips. OCR text for both PDFs and the
result JSON can contain sensitive student information. Model output is
unverified; teachers must check it against the original paper and scheme.

The JSON download uses the Phase 6 deterministic serializer. It can contain
private answer/marking evidence and must be protected by the user.

## Synthetic demo mode

There is currently no official marking-scheme PDF in repository private
storage. Presentation QA can be enabled explicitly:

```bash
UI_DEMO_ENABLED=true uv run python -m streamlit run streamlit_app.py
```

The Developer / Synthetic Demo expander provides completed, review-required,
partial, and failed synthetic results. Every synthetic view displays:

> Synthetic demonstration result — not produced from a real examination paper.

Demo mode never activates automatically and never becomes a fallback after a
real processing failure.

## Phase 7 verification

Synthetic, private-data-free QA covers the Overview, Mark a Paper, Results, and
Review Queue routes, their empty states, one- and two-file readiness states,
active processing status, completed, review-required, partially completed, and
failed result presentation, Test/criterion expanders, feedback, the review
queue, JSON download, and reset. A separate synthetic-PDF handoff
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

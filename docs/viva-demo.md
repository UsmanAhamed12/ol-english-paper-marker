# Viva demonstration guide

## Important distinction

The application has two demonstration paths:

1. **Synthetic UI demo** — shows result presentation, status handling, and
   examiner-review messaging using fabricated data. It is not model output and
   must be introduced as synthetic.
2. **Local paper smoke** — processes the private paper and scheme in
   `data/test/`. It exercises the real local pipeline, but it does not have
   verified teacher marks and must not be presented as an accuracy result.

Do not use the teacher-annotated PDFs in `data/raw/marked_papers/` as student
inputs. Do not claim that the system is ready to issue official marks.

## Start the application on Windows

From the repository root, enable the explicitly synthetic scenario and launch
the app:

```powershell
$env:UI_DEMO_ENABLED = 'true'
.\run_app.ps1
```

Open the local URL printed by Streamlit, normally `http://127.0.0.1:8501`.
Keep Ollama/Tesseract available only if demonstrating real PDF processing; the
synthetic scenario does not require either provider.

## Suggested presentation flow

1. On **Mark a Paper**, expand **Developer / Synthetic Demo**.
2. Point out the warning that the result is synthetic.
3. Load **Completed**, then open **Results** to show the synthetic normalized
   `/100` score, original scheme score, Test breakdown, feedback, and
   processing-stage details. In a real run, a normalized score is shown only
   when all marks are resolved.
4. Return to the demo controls and load **Review required** or **Partially
   completed**. Show that unresolved marks are not represented as a definitive
   final score. The **Review Queue** shows that examiner review is required;
   this prototype displays the queue but does not submit examiner edits.
5. If time permits, upload `data/test/unmarked_paper.pdf` and
   `data/test/marking scheme.pdf` on **Mark a Paper** and start a real local
   run. This may take several minutes.

The recorded local smoke aligned 12 Tests, confirmed 0 marks, left 69 marks
unresolved, produced 88 review items, and withheld the final score. Explain
that this is evidence of a conservative pipeline execution, not evidence of
grading accuracy.

A later rerun of the same private pair failed before Test alignment after
approximately 593 seconds, so it produced no real-paper feedback or score.
The configured local `llama2` model was independently exercised through the
feedback service using synthetic answer/rubric evidence; it returned validated
question advice with prompt version `question-feedback-v2` in approximately
119 seconds. This verifies the local feedback integration, not feedback
quality on a real script. Do not promise that the current real-paper smoke will
finish during the viva.

## What the project demonstrates

- Local PDF ingestion and OCR with explicit stage reporting.
- Test/criterion structure and deterministic score accounting.
- A replaceable local Ollama grading integration.
- Separate local-Ollama learning feedback and improvement tips that cannot
  change marks.
- A normalized `/100` display for fully resolved results, with the original
  scheme score retained.
- Mark bounds and review fallbacks that prevent unsupported final scores.
- A Streamlit result and review interface.
- A reproducible evaluation framework whose metrics require separate verified
  teacher references.

## Claims to avoid

- Do not describe the synthetic score as a prediction.
- Do not report the one-paper smoke as a 40-paper or accuracy evaluation.
- Do not describe confidence as calibrated probability.
- Do not say examiner edits resume the graph through this UI; review is
  currently read-only in the Streamlit prototype.
- Do not use teacher-annotated historical scans as model input.

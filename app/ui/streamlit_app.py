"""Streamlit entry point for the local O/L English marking prototype."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from uuid import uuid4

import streamlit as st

from app.core.config import Settings, get_settings
from app.core.exceptions import ApplicationError
from app.core.logging import configure_logging
from app.groq_marking.factory import create_groq_paper_marking_service
from app.groq_marking.models import GroqPaperMarkingResult
from app.orchestration.factory import create_local_paper_marking_service
from app.orchestration.models import (
    FinalPaperMarkingResult,
    FinalPaperStatus,
    PaperMarkingRequest,
)
from app.ui.components.review import render_review_queue
from app.ui.components.summary import render_summary
from app.ui.components.test_results import render_feedback, render_test_results
from app.ui.components.upload import render_upload_metadata
from app.ui.components.warnings import (
    render_errors,
    render_stage_history,
    render_warnings,
)
from app.ui.demo import DemoScenario, synthetic_demo_result
from app.ui.presentation.mapper import map_final_result
from app.ui.presentation.models import UIRunState
from app.ui.runtime import PrivateUploadStore, UploadedPDF

_RESULT = "marking_result"
_GROQ_RESULT = "groq_marking_result"
_RUN_STATE = "run_state"
_SAFE_ERROR = "safe_error"
_ACTIVE_RUN = "active_run_id"
_UPLOAD_DIR = "upload_run_dir"
_IS_DEMO = "is_demo_result"
_UPLOAD_GENERATION = "upload_generation"


def main() -> None:
    """Render the multipage application and delegate marking to Phase 6."""

    settings = get_settings()
    configure_logging(settings.log_level)
    st.set_page_config(
        page_title="O/L English Paper Marking",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _initialize_state()
    _render_style()
    st.sidebar.markdown("## O/L Marking")
    st.sidebar.caption("Local-first · Examiner-led")
    st.sidebar.divider()
    st.sidebar.caption(
        "Prototype for examiner-assisted marking. Results are held in the "
        "current browser session."
    )
    st.sidebar.caption("Change appearance from the top-right menu → Theme.")

    marking_page = st.Page(
        "app/ui/pages/marking.py",
        title="Mark a Paper",
        icon=":material/upload_file:",
        url_path="marking",
    )
    results_page = st.Page(
        "app/ui/pages/results.py",
        title="Results",
        icon=":material/assessment:",
        url_path="results",
    )
    review_page = st.Page(
        "app/ui/pages/review.py",
        title="Review Queue",
        icon=":material/rule:",
        url_path="review",
    )
    overview_page = st.Page(
        "app/ui/pages/dashboard.py",
        title="Overview",
        icon=":material/dashboard:",
        url_path="dashboard",
        default=True,
    )
    about_page = st.Page(
        "app/ui/pages/about.py",
        title="About & Readiness",
        icon=":material/info:",
        url_path="about",
    )
    navigation = st.navigation(
        {
            "WORKSPACE": [overview_page, marking_page],
            "ASSESSMENT": [results_page, review_page],
            "PROJECT": [about_page],
        },
        position="hidden",
    )
    st.sidebar.markdown("**WORKSPACE**")
    st.sidebar.page_link(
        overview_page,
        label="Overview",
        icon=":material/dashboard:",
        use_container_width=True,
    )
        else:
            st.write("Latest run: **Groq vision assessment**")
            st.write(
    f"{len(groq_result.assessment.questions)} questions · "
    f"{sum(question.review_required for question in groq_result.assessment.questions)} "
    "questions flagged for review"
            )
            st.page_link(
    "app/ui/pages/results.py",
    label="Open results",
    icon=":material/assessment:",
    use_container_width=True,
            )
            st.page_link(
    "app/ui/pages/review.py",
    label="Open review queue",
    icon=":material/rule:",
    use_container_width=True,
            )
    st.sidebar.page_link(
        marking_page,
        label="Mark a Paper",
        icon=":material/upload_file:",
        use_container_width=True,
    )
    st.sidebar.markdown("**ASSESSMENT**")
    st.sidebar.page_link(
        results_page,
        label="Results",
        icon=":material/assessment:",
        use_container_width=True,
    )
    st.sidebar.page_link(
        review_page,
        label="Review Queue",
        icon=":material/rule:",
        use_container_width=True,
    )
    st.sidebar.markdown("**PROJECT**")
    st.sidebar.page_link(
        about_page,
        label="About & Readiness",
        icon=":material/info:",
        use_container_width=True,
    )
    st.sidebar.divider()
    navigation.run()


def prepare_page() -> None:
    """Prepare a direct multipage script and preserve session defaults."""

    _initialize_state()
    _render_style()


def render_dashboard_page() -> None:
    """Show session status and the shortest paths to core tasks."""

    _render_page_header(
        "WORKSPACE",
        "Marking workspace",
        "Prepare a paper, inspect its result, and keep examiner review visible.",
    )
    result = st.session_state[_RESULT]
    groq_result = st.session_state[_GROQ_RESULT]
    state = UIRunState(st.session_state[_RUN_STATE])
    view = (
        map_final_result(result, is_demo=bool(st.session_state[_IS_DEMO]))
        if isinstance(result, FinalPaperMarkingResult)
        else None
    )
    metrics = st.columns(3)
    latest_score = (
        view.score_value
        if view is not None
        else (
            f"{groq_result.score_out_of_100} / 100"
            if isinstance(groq_result, GroqPaperMarkingResult)
            and groq_result.score_out_of_100 is not None
            else "Needs review"
            if isinstance(groq_result, GroqPaperMarkingResult)
            else "—"
        )
    )
    review_count = (
        len(view.review_items)
        if view is not None
        else sum(
            question.review_required for question in groq_result.assessment.questions
        )
        if isinstance(groq_result, GroqPaperMarkingResult)
        else "—"
    )
    run_status = (
        "Not started"
        if state is UIRunState.READY
        else state.value.replace("_", " ").title()
    )
    metrics[0].metric(
        "Current run",
        run_status,
    )
    metrics[1].metric(
        view.score_heading if view is not None else "Latest score",
        latest_score,
    )
    metrics[2].metric(
        "Items for review",
        str(review_count),
    )

    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        st.markdown("### Start a marking run")
        st.write("Upload a student script and its matching official marking scheme.")
        st.page_link(
            "app/ui/pages/marking.py",
            label="Open marking workspace",
            icon=":material/arrow_forward:",
            use_container_width=True,
        )
    with right, st.container(border=True):
        st.markdown("### Review this session")
        if view is None and not isinstance(groq_result, GroqPaperMarkingResult):
            st.write("Results and review items will appear here after a run.")
            st.caption("There is no saved run history in this prototype.")
        elif view is not None:
            st.write(f"Latest run status: **{view.status}**")
            st.write(f"{len(view.tests)} Tests · {len(view.review_items)} review items")
            st.page_link(
                "app/ui/pages/results.py",
                label="Open results",
                icon=":material/assessment:",
                use_container_width=True,
            )
            if view.review_items:
                st.page_link(
                    "app/ui/pages/review.py",
                    label="Open review queue",
                    icon=":material/rule:",
                    use_container_width=True,
                )

    st.subheader("How marking works")
    steps = st.columns(4, gap="small")
    for column, number, label in zip(
        steps,
        ("01", "02", "03", "04"),
        ("Upload PDFs", "Extract & assess", "Inspect evidence", "Confirm with care"),
        strict=True,
    ):
        column.markdown(
            f'<div class="workflow-step"><span>{number}</span>{label}</div>',
            unsafe_allow_html=True,
        )
    st.info(
        "This is an AI-assisted prototype, not a validated official marking "
        "authority. Unresolved marks require examiner attention."
    )


def render_marking_page() -> None:
    """Collect the two PDFs and start the existing marking service."""

    settings = get_settings()
    _render_page_header(
        "NEW ASSESSMENT",
        "Mark a paper",
        "Use the matching official scheme so every assessment has a clear rubric.",
    )
    generation = int(st.session_state[_UPLOAD_GENERATION])
    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        st.markdown("### 01 · Student answer paper")
        st.caption("The scanned script you want to assess.")
        student = st.file_uploader(
            "Upload student answer paper",
            type=("pdf",),
            key=f"student_upload_{generation}",
            on_change=_clear_groq_consent,
            help=(
                "Select one local PDF. The generated runtime filename is identity-safe."
            ),
        )
        render_upload_metadata(student)
    with right, st.container(border=True):
        st.markdown("### 02 · Official marking scheme")
        st.caption("The rubric used to ground the assessment.")
        scheme = st.file_uploader(
            "Upload marking scheme",
            type=("pdf",),
            key=f"scheme_upload_{generation}",
            on_change=_clear_groq_consent,
            help="Select the corresponding official marking-scheme PDF.",
        )
        render_upload_metadata(scheme)

    state = UIRunState(st.session_state[_RUN_STATE])
    engine = st.radio(
        "Marking engine",
        ("Groq vision (cloud)", "Local marking pipeline"),
        horizontal=True,
        key="marking_engine",
    )
    is_groq = engine == "Groq vision (cloud)"
    consent = True
    has_groq_key = settings.groq_api_key is not None and bool(
        settings.groq_api_key.get_secret_value().strip()
    )
    if is_groq:
        st.info(
            f"Cloud marking uses `{settings.groq_model}`. Both PDFs, including "
            "student handwriting, will be sent to Groq for processing."
        )
        consent = st.checkbox(
            "I have permission to send these documents to Groq for OCR and marking.",
            key="groq_data_consent",
        )
        if not has_groq_key:
            st.info("Set GROQ_API_KEY in your local .env file to enable Groq marking.")
    else:
        st.caption(
            "The local pipeline uses Tesseract and Ollama. Uploaded PDFs stay on "
            "this machine."
        )

    has_result = isinstance(
        st.session_state[_RESULT], FinalPaperMarkingResult
    ) or isinstance(st.session_state[_GROQ_RESULT], GroqPaperMarkingResult)
    if student is not None and scheme is not None:
        st.success("Both documents are ready. Start marking when you are ready.")
    else:
        st.info("Add both PDFs to prepare a marking run.")
    mark_disabled = (
        student is None
        or scheme is None
        or has_result
        or state in {UIRunState.VALIDATING, UIRunState.PROCESSING}
        or (is_groq and (not consent or not has_groq_key))
    )
    if st.button(
        "Start marking",
        type="primary",
        disabled=mark_disabled,
        use_container_width=True,
    ):
        if is_groq:
            _run_groq_marking(student, scheme, settings)
        else:
            _run_real_marking(student, scheme, settings)
        st.rerun()

    if st.session_state[_SAFE_ERROR]:
        st.error(str(st.session_state[_SAFE_ERROR]))
    if has_result:
        st.success("This run is complete. Open Results to inspect the breakdown.")
        if settings.ui_demo_enabled and st.session_state[_IS_DEMO]:
            st.warning(
                "Synthetic demonstration result — not produced from a real "
                "examination paper."
            )
    elif settings.ui_demo_enabled:
        _render_demo_controls(has_result)

    st.caption("Temporary upload copies are removed after processing.")


def render_results_page() -> None:
    """Present the active session result and its safe export."""

    settings = get_settings()
    _render_page_header(
        "ASSESSMENT",
        "Results",
        "Inspect question marks, explanations, feedback, and processing evidence.",
    )
    result = st.session_state[_RESULT]
    groq_result = st.session_state[_GROQ_RESULT]
    if isinstance(groq_result, GroqPaperMarkingResult):
        _render_groq_result(groq_result)
        if st.button("Mark another paper", use_container_width=True):
            _reset(settings)
        return
    if not isinstance(result, FinalPaperMarkingResult):
        st.info("No result is available in this browser session yet.")
        st.page_link(
            "app/ui/pages/marking.py",
            label="Start a marking run",
            icon=":material/arrow_forward:",
        )
        return

    view = map_final_result(result, is_demo=bool(st.session_state[_IS_DEMO]))
    render_summary(view)
    render_errors(view)
    if result.status is not FinalPaperStatus.FAILED:
        render_test_results(view)
        render_feedback(view)
        render_review_queue(view)
    render_warnings(view)
    render_stage_history(view)
    st.download_button(
        "Download result (JSON)",
        data=view.download_json,
        file_name=f"marking-result-{view.run_id}.json",
        mime="application/json",
        use_container_width=True,
    )
    if st.button("Mark another paper", use_container_width=True):
        _reset(settings)


def render_review_page() -> None:
    """Make the current run's examiner-review state easy to find."""

    _render_page_header(
        "EXAMINER WORKSPACE",
        "Review queue",
        "See which items need human attention before treating marks as final.",
    )
    result = st.session_state[_RESULT]
    groq_result = st.session_state[_GROQ_RESULT]
    if isinstance(groq_result, GroqPaperMarkingResult):
        flagged = tuple(
            question
            for question in groq_result.assessment.questions
            if question.review_required or question.maximum_marks == 0
        )
        if not flagged:
            st.success("No questions were flagged by the model for review.")
        for question in flagged:
            with st.expander(
                f"Question {question.question_number} · Review required",
                expanded=True,
            ):
                st.write(question.explanation)
                st.write(
                    f"**Student answer:** {question.student_answer or 'Not detected'}"
                )
        st.caption(
            "AI-generated review flags are advisory. Review the original script "
            "and official scheme before confirming any mark."
        )
        return
    if not isinstance(result, FinalPaperMarkingResult):
        st.info("The review queue is empty because no paper has been marked.")
        st.page_link(
            "app/ui/pages/marking.py",
            label="Start a marking run",
            icon=":material/arrow_forward:",
        )
        return
    view = map_final_result(result, is_demo=bool(st.session_state[_IS_DEMO]))
    if view.is_demo:
        st.warning(
            "Synthetic demonstration result — not produced from a real "
            "examination paper."
        )
    render_review_queue(view)
    if not view.review_items:
        st.success("No items are currently flagged for review in this run.")
    st.caption(
        "Review decisions are read-only in this prototype; no examiner edits "
        "are submitted to the marking workflow."
    )


def render_about_page() -> None:
    """Explain current capabilities, local prerequisites, and limitations."""

    settings = get_settings()
    _render_page_header(
        "PROJECT",
        "About & readiness",
        "A transparent look at what this prototype does and what it still needs.",
    )
    st.subheader("Designed for traceable, examiner-assisted marking")
    st.write(
        "Choose Groq vision processing or the existing local OCR and marking "
        "pipeline. Both paths show question-level reasoning and require examiner "
        "judgment; neither is validated for official marking."
    )
    left, right = st.columns(2, gap="large")
    with left, st.container(border=True):
        st.markdown("### Configured models")
        st.write(f"**Groq vision model:** `{settings.groq_model}`")
        st.write(f"**Local grading model:** `{settings.ollama_grading_model}`")
        st.write(f"**OCR language:** `{settings.tesseract_language}`")
        st.caption("Configured values do not verify live service availability.")
    with right, st.container(border=True):
        st.markdown("### Data handling")
        st.write(
            "Groq mode sends both PDFs to Groq. Local mode keeps the processing "
            "on this machine."
        )
        st.write(
            "Temporary upload copies are removed after processing. Results are "
            "held in the current browser session, not a permanent database."
        )

    st.subheader("Important limitations")
    st.warning(
        "This is a project prototype, not a validated official examination "
        "marking system. Do not rely on its output without examiner review."
    )
    st.markdown(
        "- Handwriting OCR accuracy is not an established official baseline.\n"
        "- Real end-to-end accuracy needs evaluation against teacher-marked papers.\n"
        "- Review items are visible but cannot yet be edited in the interface.\n"
        "- Saved run history and database persistence are not available."
    )


def _render_page_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="brand-kicker">{kicker}'
        "<span>LOCAL-FIRST · EXAMINER-LED</span></div>",
        unsafe_allow_html=True,
    )
    st.title(title)
    st.markdown(
        f'<p class="hero-subtitle">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def _run_real_marking(
    student: UploadedPDF | None,
    scheme: UploadedPDF | None,
    settings: Settings,
) -> None:
    if student is None or scheme is None:
        st.session_state[_SAFE_ERROR] = "Both PDF files are required."
        return
    run_id = uuid4()
    store = _upload_store(settings)
    st.session_state[_ACTIVE_RUN] = str(run_id)
    st.session_state[_RUN_STATE] = UIRunState.VALIDATING.value
    st.session_state[_SAFE_ERROR] = None
    st.session_state[_RESULT] = None
    st.session_state[_GROQ_RESULT] = None
    st.session_state[_IS_DEMO] = False
    uploads = None
    try:
        with st.status("Validating uploaded PDFs…", expanded=True) as status:
            uploads = store.materialize(student, scheme, run_id=run_id)
            st.session_state[_UPLOAD_DIR] = str(uploads.run_dir)
            st.session_state[_RUN_STATE] = UIRunState.PROCESSING.value
            status.update(
                label="Processing locally — OCR and marking are running…",
                state="running",
            )
            st.write("The backend is processing each stage sequentially.")
            result = create_local_paper_marking_service(settings).mark_paper(
                PaperMarkingRequest(
                    student_pdf=uploads.student_pdf,
                    marking_scheme_pdf=uploads.marking_scheme_pdf,
                    run_id=run_id,
                )
            )
            st.session_state[_RESULT] = result
            st.session_state[_RUN_STATE] = (
                UIRunState.FAILED.value
                if result.status is FinalPaperStatus.FAILED
                else UIRunState.COMPLETED.value
            )
            status.update(
                label=(
                    "Paper marking could not be completed safely."
                    if result.status is FinalPaperStatus.FAILED
                    else "Paper marking completed."
                ),
                state=(
                    "error" if result.status is FinalPaperStatus.FAILED else "complete"
                ),
            )
    except ApplicationError as error:
        st.session_state[_RUN_STATE] = UIRunState.FAILED.value
        st.session_state[_SAFE_ERROR] = _safe_runtime_message(error)
    except (OSError, ValueError):
        st.session_state[_RUN_STATE] = UIRunState.FAILED.value
        st.session_state[_SAFE_ERROR] = (
            "Paper marking could not start. Check the local runtime prerequisites."
        )
    finally:
        store.cleanup(uploads.run_dir if uploads is not None else None)
        st.session_state[_UPLOAD_DIR] = None


def _run_groq_marking(
    student: UploadedPDF | None,
    scheme: UploadedPDF | None,
    settings: Settings,
) -> None:
    if student is None or scheme is None:
        st.session_state[_SAFE_ERROR] = "Both PDF files are required."
        return
    run_id = uuid4()
    store = _upload_store(settings)
    st.session_state[_ACTIVE_RUN] = str(run_id)
    st.session_state[_RUN_STATE] = UIRunState.VALIDATING.value
    st.session_state[_SAFE_ERROR] = None
    st.session_state[_RESULT] = None
    st.session_state[_GROQ_RESULT] = None
    st.session_state[_IS_DEMO] = False
    uploads = None
    try:
        with st.status("Preparing secure temporary copies…", expanded=True) as status:
            uploads = store.materialize(student, scheme, run_id=run_id)
            st.session_state[_UPLOAD_DIR] = str(uploads.run_dir)
            st.session_state[_RUN_STATE] = UIRunState.PROCESSING.value
            status.update(
                label=f"Sending pages to Groq `{settings.groq_model}`…",
                state="running",
            )
            st.write(
                "Groq is transcribing both PDFs and grading each detected question "
                "against the uploaded scheme."
            )
            result = create_groq_paper_marking_service(settings).mark_paper(
                uploads.student_pdf,
                uploads.marking_scheme_pdf,
                run_id=run_id,
            )
            st.session_state[_GROQ_RESULT] = result
            st.session_state[_RUN_STATE] = UIRunState.COMPLETED.value
            status.update(
                label=(
                    "Assessment is ready for examiner review."
                    if result.requires_human_review
                    else "Assessment completed. Verify the marks against the originals."
                ),
                state="complete",
            )
    except ApplicationError as error:
        st.session_state[_RUN_STATE] = UIRunState.FAILED.value
        st.session_state[_SAFE_ERROR] = _safe_runtime_message(error)
    except (OSError, ValueError):
        st.session_state[_RUN_STATE] = UIRunState.FAILED.value
        st.session_state[_SAFE_ERROR] = (
            "Cloud marking could not start. Check the PDF files and Groq settings."
        )
    finally:
        store.cleanup(uploads.run_dir if uploads is not None else None)
        st.session_state[_UPLOAD_DIR] = None


def _render_groq_result(result: GroqPaperMarkingResult) -> None:
    """Render the question-by-question cloud assessment and safe export."""

    st.caption(
        f"Vision model: `{result.model}` · AI-generated · examiner review required"
    )
    if result.requires_human_review:
        st.warning(
            "This result is provisional. One or more questions need review, or "
            "the full-paper rubric maximum could not be established."
        )
    else:
        st.info(
            "AI-generated assessment only. Confirm every mark against the "
            "original script and official marking scheme."
        )

    metrics = st.columns(3)
    score_label = (
        "Provisional score / 100" if result.requires_human_review else "Score / 100"
    )
    score_value = (
        f"{result.score_out_of_100} / 100"
        if result.score_out_of_100 is not None
        else "Cannot safely calculate"
    )
    metrics[0].metric(score_label, score_value)
    metrics[1].metric(
        "Marks from scheme",
        f"{result.total_awarded_marks} / {result.total_maximum_marks}",
    )
    metrics[2].metric("Questions assessed", str(len(result.assessment.questions)))

    st.subheader("Question-by-question assessment")
    for index, question in enumerate(result.assessment.questions):
        mark = (
            f"{question.awarded_marks} / {question.maximum_marks}"
            if question.maximum_marks > 0
            else "Rubric maximum unavailable"
        )
        with st.expander(
            f"{question.question_number}  ·  {mark}",
            expanded=index == 0,
        ):
            if question.review_required or question.maximum_marks == 0:
                st.warning("Examiner review is required for this question.")
            if question.question_summary:
                st.markdown(f"**Question**  \n{question.question_summary}")
            st.markdown(
                "**Student answer**  \n"
                + (question.student_answer or "No answer was confidently detected.")
            )
            st.markdown(f"**Why this mark**  \n{question.explanation}")
            if question.feedback:
                st.markdown(f"**Feedback**  \n{question.feedback}")
            if question.improvement_tips:
                st.markdown("**Ways to improve**")
                for tip in question.improvement_tips:
                    st.markdown(f"- {tip}")

    st.subheader("Overall feedback")
    st.write(result.assessment.overall_feedback)
    if result.assessment.improvement_tips:
        st.markdown("**Priority improvement tips**")
        for tip in result.assessment.improvement_tips:
            st.markdown(f"- {tip}")

    with st.expander("OCR evidence from both PDFs"):
        st.markdown("#### Student paper")
        for page in result.student_pages:
            st.markdown(f"**Page {page.page_number} · Printed**")
            st.text(page.printed_text or "No printed text detected.")
            st.markdown("**Handwritten**")
            st.text(page.handwritten_text or "No handwritten text detected.")
            st.caption(page.understanding)
        st.markdown("#### Marking scheme")
        for page in result.marking_scheme_pages:
            st.markdown(f"**Page {page.page_number} · Printed**")
            st.text(page.printed_text or "No printed text detected.")
            if page.handwritten_text:
                st.markdown("**Handwritten**")
                st.text(page.handwritten_text)
            st.caption(page.understanding)

    st.download_button(
        "Download assessment (JSON)",
        data=result.model_dump_json(indent=2),
        file_name=f"groq-marking-result-{result.run_id}.json",
        mime="application/json",
        use_container_width=True,
    )


def _render_demo_controls(has_result: bool) -> None:
    with st.expander("Developer / Synthetic Demo"):
        st.warning(
            "Synthetic demonstration result — not produced from a real "
            "examination paper."
        )
        scenario = cast(
            DemoScenario,
            st.selectbox(
                "Synthetic result state",
                tuple(DemoScenario),
                format_func=lambda value: value.value,
            ),
        )
        if st.button("Load Synthetic Demo", disabled=has_result):
            st.session_state[_RESULT] = synthetic_demo_result(scenario)
            st.session_state[_IS_DEMO] = True
            st.session_state[_RUN_STATE] = (
                UIRunState.FAILED.value
                if scenario is DemoScenario.FAILED
                else UIRunState.COMPLETED.value
            )
            st.session_state[_SAFE_ERROR] = None
            st.rerun()


def _safe_runtime_message(error: ApplicationError) -> str:
    message = str(error).casefold()
    if "groq" in message or "api key" in message:
        return (
            "Groq marking could not be completed. Check GROQ_API_KEY, GROQ_MODEL, "
            "and the network connection."
        )
    if "tesseract" in message:
        return (
            "Tesseract was not found. Install it or set TESSERACT_EXECUTABLE "
            "to its executable path."
        )
    if "ollama" in message or "provider" in message:
        return (
            "Local Ollama is unavailable. Start Ollama and verify the configured model."
        )
    if "pdf" in message:
        return "One of the uploaded files could not be processed as a valid PDF."
    return "Paper marking could not start safely. Review the local setup."


def _clear_groq_consent() -> None:
    """Require fresh cloud consent whenever either uploaded PDF changes."""

    st.session_state["groq_data_consent"] = False


def _initialize_state() -> None:
    defaults: dict[str, object] = {
        _RESULT: None,
        _GROQ_RESULT: None,
        _RUN_STATE: UIRunState.READY.value,
        _SAFE_ERROR: None,
        _ACTIVE_RUN: None,
        _UPLOAD_DIR: None,
        _IS_DEMO: False,
        _UPLOAD_GENERATION: 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset(settings: Settings) -> None:
    run_dir = st.session_state.get(_UPLOAD_DIR)
    _upload_store(settings).cleanup(Path(run_dir) if isinstance(run_dir, str) else None)
    generation = int(st.session_state[_UPLOAD_GENERATION]) + 1
    for key in (_RESULT, _GROQ_RESULT, _SAFE_ERROR, _ACTIVE_RUN, _UPLOAD_DIR):
        st.session_state[key] = None
    st.session_state[_RUN_STATE] = UIRunState.READY.value
    st.session_state[_IS_DEMO] = False
    st.session_state[_UPLOAD_GENERATION] = generation
    st.session_state.pop("groq_data_consent", None)
    st.rerun()


def _upload_store(settings: Settings) -> PrivateUploadStore:
    return PrivateUploadStore(
        settings.runtime_data_dir / "streamlit-uploads",
        maximum_bytes=settings.max_pdf_size_mb * 1024 * 1024,
    )


def _render_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
          --app-ink: light-dark(#193447, #e7eef2);
          --app-muted: light-dark(#607681, #a9bac4);
          --app-primary: light-dark(#146b61, #65c8ae);
          --app-accent: light-dark(#a87319, #edc978);
          --app-line: light-dark(#dce7e4, #34434f);
          --app-card: light-dark(rgba(255,255,255,.9), #18232e);
          --app-card-soft: light-dark(#f3f7f6, #202d38);
          --app-surface: light-dark(#f7f9f8, #10171f);
          background:
            radial-gradient(
              ellipse at 50% -12%,
              light-dark(rgba(20,107,97,.09), rgba(101,200,174,.09)),
              transparent 42%
            ),
            var(--app-surface) !important;
          color: var(--app-ink);
        }
        .block-container {max-width: 1120px; padding-top: 2.4rem; padding-bottom: 3rem;}
        h1 {
          color: var(--app-ink);
          font-size: clamp(2.2rem, 4.2vw, 3.25rem) !important;
          letter-spacing: -.045em;
          line-height: 1.08 !important;
          margin: .35rem 0 .6rem !important;
        }
        h2, h3, h4 {color: var(--app-ink); letter-spacing: -.02em;}
        p, [data-testid="stCaption"] {color: var(--app-muted);}
        .brand-kicker {
          color: var(--app-primary);
          font-size: .72rem;
          font-weight: 800;
          letter-spacing: .14em;
        }
        .brand-kicker span {
          color: var(--app-muted);
          margin-left: .8rem;
          font-size: .65rem;
          letter-spacing: .11em;
        }
        .hero-subtitle {
          color: var(--app-muted);
          font-size: 1.18rem;
          font-weight: 500;
          margin: 0 0 .35rem;
        }
        .workflow-step {
          align-items: center;
          background: var(--app-card);
          border: 1px solid var(--app-line);
          border-radius: 10px;
          color: var(--app-ink);
          display: flex;
          font-size: .8rem;
          font-weight: 650;
          gap: .55rem;
          min-height: 2.7rem;
          padding: .45rem .65rem;
        }
        .workflow-step span {
          align-items: center;
          background: var(--app-card-soft);
          border-radius: 50%;
          color: var(--app-primary);
          display: inline-flex;
          flex: 0 0 1.55rem;
          font-size: .67rem;
          height: 1.55rem;
          justify-content: center;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
          background: var(--app-card);
          border-color: var(--app-line);
          border-radius: 14px;
          box-shadow: 0 8px 24px light-dark(rgba(20,43,59,.035), rgba(0,0,0,.18));
        }
        [data-testid="stFileUploaderDropzone"] {
          background: var(--app-card-soft);
          border: 1px dashed var(--app-line);
          border-radius: 10px;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
          border-color: var(--app-primary);
        }
        [data-testid="stMetric"] {
          background: var(--app-card-soft);
          border: 1px solid var(--app-line);
          border-radius: 11px;
          padding: .9rem 1rem;
        }
        [data-testid="stMetricLabel"] {color: var(--app-muted);}
        [data-testid="stAlert"] {border-radius: 11px;}
        .stButton > button, .stDownloadButton > button {
          border-radius: 9px;
          font-weight: 700;
          min-height: 2.8rem;
          transition: transform .15s ease, box-shadow .15s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
          box-shadow: 0 5px 14px light-dark(rgba(20,43,59,.1), rgba(0,0,0,.32));
          transform: translateY(-1px);
        }
        .stButton > button[kind="primary"] {
          background: var(--app-primary);
          border-color: var(--app-primary);
          color: light-dark(#ffffff, #10231f);
        }
        button:focus-visible, a:focus-visible {
          outline: 3px solid var(--app-accent) !important;
          outline-offset: 2px;
        }
        hr {border-color: var(--app-line);}
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

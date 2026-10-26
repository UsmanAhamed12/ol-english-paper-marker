"""Streamlit entry point for the local O/L English marking prototype."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from uuid import uuid4

import streamlit as st

from app.core.config import Settings, get_settings
from app.core.exceptions import ApplicationError
from app.core.logging import configure_logging
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
_RUN_STATE = "run_state"
_SAFE_ERROR = "safe_error"
_ACTIVE_RUN = "active_run_id"
_UPLOAD_DIR = "upload_run_dir"
_IS_DEMO = "is_demo_result"
_UPLOAD_GENERATION = "upload_generation"


def main() -> None:
    """Render the thin application and delegate all marking to Phase 6."""

    settings = get_settings()
    configure_logging(settings.log_level)
    st.set_page_config(
        page_title="O/L English Paper Marking",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _initialize_state()
    _render_style()
    _render_header()

    generation = int(st.session_state[_UPLOAD_GENERATION])
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Student answer paper")
        student = st.file_uploader(
            "Upload Student Answer Paper",
            type=("pdf",),
            key=f"student_upload_{generation}",
            help=(
                "Select one local PDF. The generated runtime filename is identity-safe."
            ),
        )
        render_upload_metadata(student)
    with right:
        st.markdown("### Marking scheme")
        scheme = st.file_uploader(
            "Upload Marking Scheme",
            type=("pdf",),
            key=f"scheme_upload_{generation}",
            help="Select the corresponding official marking-scheme PDF.",
        )
        render_upload_metadata(scheme)

    st.info(
        "Processing locally. Handwriting OCR and descriptive marking may take "
        "several minutes. Files are not sent to an external service."
    )
    state = UIRunState(st.session_state[_RUN_STATE])
    has_result = isinstance(st.session_state[_RESULT], FinalPaperMarkingResult)
    mark_disabled = (
        student is None
        or scheme is None
        or has_result
        or state
        in {
            UIRunState.VALIDATING,
            UIRunState.PROCESSING,
        }
    )
    if st.button(
        "Mark Paper",
        type="primary",
        disabled=mark_disabled,
        use_container_width=True,
    ):
        _run_real_marking(student, scheme, settings)
        st.rerun()

    if settings.ui_demo_enabled:
        _render_demo_controls(has_result)

    result = st.session_state[_RESULT]
    if isinstance(result, FinalPaperMarkingResult):
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
            "Download Result (JSON)",
            data=view.download_json,
            file_name=f"marking-result-{view.run_id}.json",
            mime="application/json",
            use_container_width=True,
        )
        if st.button("Mark Another Paper", use_container_width=True):
            _reset(settings)
    elif st.session_state[_SAFE_ERROR]:
        st.error(str(st.session_state[_SAFE_ERROR]))

    st.divider()
    st.caption(
        "Final-year project prototype · Local processing · Review flags must be "
        "resolved by an examiner before an unresolved score is treated as final."
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
    if "tesseract" in message:
        return "Local Tesseract is unavailable. Install and configure Tesseract first."
    if "ollama" in message or "provider" in message:
        return (
            "Local Ollama is unavailable. Start Ollama and verify the configured model."
        )
    if "pdf" in message:
        return "One of the uploaded files could not be processed as a valid PDF."
    return "Paper marking could not start safely. Review the local setup."


def _initialize_state() -> None:
    defaults: dict[str, object] = {
        _RESULT: None,
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
    for key in (_RESULT, _SAFE_ERROR, _ACTIVE_RUN, _UPLOAD_DIR):
        st.session_state[key] = None
    st.session_state[_RUN_STATE] = UIRunState.READY.value
    st.session_state[_IS_DEMO] = False
    st.session_state[_UPLOAD_GENERATION] = generation
    st.rerun()


def _upload_store(settings: Settings) -> PrivateUploadStore:
    return PrivateUploadStore(
        settings.runtime_data_dir / "streamlit-uploads",
        maximum_bytes=settings.max_pdf_size_mb * 1024 * 1024,
    )


def _render_header() -> None:
    st.title("Automated O/L English Paper Marking System")
    st.markdown("### AI-assisted local marking with structured review and feedback")
    st.caption(
        "Prototype automated marking pipeline — not a claim of validated official "
        "examination-marking accuracy."
    )
    st.markdown(
        "**Workflow:** Upload student paper → Upload marking scheme → Validate → "
        "Mark → Review results"
    )


def _render_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 2rem;}
        [data-testid="stMetric"] {
          background: #f7f8fa; padding: 1rem; border-radius: .6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

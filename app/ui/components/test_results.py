"""Readable Test and criterion breakdown components."""

from __future__ import annotations

import streamlit as st

from app.ui.presentation.models import ResultViewModel


def render_test_results(view: ResultViewModel) -> None:
    st.subheader("Test-by-Test Results")
    if not view.tests:
        st.caption("No Test-level results are available for this run.")
        return
    for test in view.tests:
        with st.expander(test.heading):
            columns = st.columns(4)
            columns[0].metric("Confirmed", test.confirmed_marks)
            columns[1].metric("Maximum", test.maximum_marks)
            columns[2].metric("Objective", test.objective_marks)
            columns[3].metric("Subjective", test.subjective_marks)
            st.caption(
                f"Status: {test.status} · Unresolved: {test.unresolved_marks} marks"
            )
            if test.criteria:
                st.markdown("**Criterion breakdown**")
            for criterion in test.criteria:
                st.markdown(
                    f"**{criterion.criterion_id}** — {criterion.criterion_type}  \n"
                    f"Marks: {criterion.marks} · Status: {criterion.status}  \n"
                    f"Decision: {criterion.decision} · Reason: {criterion.reason}"
                )
                if criterion.feedback:
                    st.caption(f"Feedback: {criterion.feedback}")
                if criterion.improvement_tips:
                    st.markdown(
                        "**Improvement tips** — grounded in the retrieved local "
                        "syllabus and official marking scheme"
                    )
                    for tip in criterion.improvement_tips:
                        st.write(f"- {tip}")
                    if criterion.syllabus_references:
                        st.caption(
                            "Syllabus references: "
                            + "; ".join(criterion.syllabus_references)
                        )
                elif criterion.improvement_advice_status in {
                    "syllabus_not_indexed",
                    "syllabus_retrieval_failed",
                    "invalid_model_tips",
                    "review_required",
                }:
                    status_message = {
                        "syllabus_not_indexed": (
                            "No indexed syllabus evidence was available; "
                            "syllabus-based tips were withheld."
                        ),
                        "syllabus_retrieval_failed": (
                            "Local syllabus retrieval failed; syllabus-based "
                            "tips were withheld."
                        ),
                        "invalid_model_tips": (
                            "The model's improvement tips did not pass validation "
                            "and were withheld."
                        ),
                        "review_required": (
                            "Improvement tips were withheld because this assessment "
                            "requires examiner review."
                        ),
                    }[criterion.improvement_advice_status]
                    st.caption(status_message)
                elif (
                    criterion.improvement_advice_status == "no_improvements_identified"
                ):
                    st.caption(
                        "No additional improvement tip was supported by the "
                        "retrieved syllabus evidence."
                    )
                st.divider()
            if test.feedback:
                st.markdown("**Feedback**")
                for feedback in test.feedback:
                    st.write(feedback)
            if test.formative_feedback is not None:
                st.markdown("**Local Ollama learning feedback**")
                st.caption(
                    "Educational guidance only; it does not award or change marks."
                )
                if test.formative_feedback_status == "provisional":
                    st.warning(
                        "Provisional feedback: verify the OCR transcription before "
                        "relying on this advice."
                    )
                st.write(test.formative_feedback)
                if test.improvement_tips:
                    st.markdown("**Improvement tips**")
                    for tip in test.improvement_tips:
                        st.write(f"- {tip}")
            elif test.formative_feedback_status in {"failed", "skipped"}:
                st.caption(
                    "Local learning feedback unavailable for this Test "
                    f"({test.formative_feedback_status})."
                )
            if test.warnings:
                with st.expander("Test warnings"):
                    for warning in test.warnings:
                        st.write(f"• {warning.message} (`{warning.code}`)")


def render_feedback(view: ResultViewModel) -> None:
    if not view.feedback_by_test:
        return
    st.subheader("Feedback")
    for test_label, items in view.feedback_by_test:
        st.markdown(f"**{test_label}**")
        for item in items:
            st.write(item)

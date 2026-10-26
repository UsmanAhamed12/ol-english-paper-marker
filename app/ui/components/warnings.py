"""Safe warning, error, and measured-stage presentation."""

from __future__ import annotations

import streamlit as st

from app.ui.presentation.models import ResultViewModel


def render_errors(view: ResultViewModel) -> None:
    if not view.errors:
        return
    st.error("Paper marking could not be completed.")
    for error in view.errors:
        st.write(f"• {error.message}")
        st.caption(f"Reference code: {error.code}")


def render_warnings(view: ResultViewModel) -> None:
    if not view.warnings:
        return
    with st.expander("Warnings and limitations"):
        for warning in view.warnings:
            st.write(f"• {warning.message} (`{warning.code}`)")


def render_stage_history(view: ResultViewModel) -> None:
    with st.expander("Processing stages", expanded=False):
        for stage in view.stages:
            st.write(f"**{stage.label}** — {stage.status} · {stage.duration}")

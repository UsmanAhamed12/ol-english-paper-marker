"""Paper-level score and metric cards."""

from __future__ import annotations

import streamlit as st

from app.ui.presentation.models import ResultViewModel, ViewTone


def render_summary(view: ResultViewModel) -> None:
    """Render confirmed/final semantics exactly as mapped."""

    st.subheader("Result summary")
    if view.is_demo:
        st.warning(
            "Synthetic demonstration result — not produced from a real "
            "examination paper."
        )
    message = f"**{view.score_heading.upper()}**  \n# {view.score_value}"
    if view.tone is ViewTone.SUCCESS:
        st.success(message)
    elif view.tone is ViewTone.ERROR:
        st.error("Paper marking could not be completed.")
    else:
        st.warning(message)
    details: list[str] = [f"Status: **{view.status}**"]
    if view.raw_score_value is not None:
        details.append(f"Raw scheme score: **{view.raw_score_value}**")
    if view.percentage is not None:
        details.append(f"Normalized: **{view.percentage}**")
    if view.unresolved_value is not None:
        details.append(f"Unresolved: **{view.unresolved_value}**")
    st.markdown(" · ".join(details))
    columns = st.columns(3)
    for index, metric in enumerate(view.metrics):
        columns[index % 3].metric(metric.label, metric.value)

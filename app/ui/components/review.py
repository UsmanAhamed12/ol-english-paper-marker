"""Highly visible examiner-review queue."""

from __future__ import annotations

import streamlit as st

from app.ui.presentation.models import ResultViewModel


def render_review_queue(view: ResultViewModel) -> None:
    if not view.review_items:
        return
    st.subheader("Review Required")
    st.warning("One or more items require examiner review before the score is final.")
    for item in view.review_items:
        heading = item.test_label
        if item.criterion_label is not None:
            heading += f" · {item.criterion_label}"
        with st.container(border=True):
            st.markdown(f"**{heading}**")
            st.write(f"Reason: {item.reason}")
            st.write(f"Marks affected: {item.marks_affected}")
            st.caption(f"Status: {item.status} · Action: Manual review recommended")

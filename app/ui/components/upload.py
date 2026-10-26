"""Upload metadata presentation without exposing materialized paths."""

from __future__ import annotations

import streamlit as st

from app.ui.runtime import UploadedPDF


def render_upload_metadata(upload: UploadedPDF | None) -> None:
    """Show only user-visible filename, size, and success state."""

    if upload is None:
        st.caption("PDF required")
        return
    size_mb = upload.size / (1024 * 1024)
    st.success("PDF selected")
    st.caption(f"{upload.name} · {size_mb:.2f} MB")

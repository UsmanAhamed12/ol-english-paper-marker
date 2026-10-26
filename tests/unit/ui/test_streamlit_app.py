"""Streamlit application boundary tests with no OCR, Ollama, or private PDFs."""

from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch
from streamlit.testing.v1 import AppTest

from app.core.config import get_settings

_APP_PATH = Path(__file__).parents[3] / "streamlit_app.py"


def _app() -> AppTest:
    return AppTest.from_file(_APP_PATH, default_timeout=20)


def test_initial_screen_requires_both_uploads() -> None:
    get_settings.cache_clear()
    app = _app().run()

    assert not app.exception
    assert app.title[0].value == "Automated O/L English Paper Marking System"
    mark_button = next(button for button in app.button if button.label == "Mark Paper")
    assert mark_button.disabled
    assert not app.warning


def test_synthetic_demo_is_explicit_and_downloadable(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("UI_DEMO_ENABLED", "true")
    get_settings.cache_clear()
    app = _app().run()

    load_button = next(
        button for button in app.button if button.label == "Load Synthetic Demo"
    )
    app = load_button.click().run()

    assert not app.exception
    assert any("Synthetic demonstration result" in item.value for item in app.warning)
    assert any("FINAL SCORE" in item.value for item in app.success)
    assert len(app.get("download_button")) == 1
    assert any(button.label == "Mark Another Paper" for button in app.button)

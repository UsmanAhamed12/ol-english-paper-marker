"""Streamlit application boundary tests with no OCR, Ollama, or private PDFs."""

from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch
from streamlit.testing.v1 import AppTest

from app.core.config import get_settings

_APP_PATH = Path(__file__).parents[3] / "streamlit_app.py"


def _app() -> AppTest:
    return AppTest.from_file(_APP_PATH, default_timeout=20)


def test_overview_and_marking_page_are_available() -> None:
    get_settings.cache_clear()
    app = _app().run()

    assert not app.exception
    assert app.title[0].value == "Marking workspace"
    assert any("no saved run history" in item.value.casefold() for item in app.caption)

    app = app.switch_page("app/ui/pages/marking.py").run()
    assert not app.exception
    assert app.title[0].value == "Mark a paper"
    mark_button = next(
        button for button in app.button if button.label == "Start marking"
    )
    assert mark_button.disabled
    assert app.radio[0].value == "Groq vision (cloud)"
    assert any("sent to Groq" in item.value for item in app.info)
    assert app.checkbox[0].value is False
    assert not app.warning


def test_results_and_review_pages_explain_empty_state() -> None:
    get_settings.cache_clear()
    app = _app().switch_page("app/ui/pages/results.py").run()

    assert not app.exception
    assert app.title[0].value == "Results"
    assert any("No result is available" in item.value for item in app.info)

    app = app.switch_page("app/ui/pages/review.py").run()
    assert not app.exception
    assert app.title[0].value == "Review queue"
    assert any("no paper has been marked" in item.value for item in app.info)


def test_synthetic_demo_is_explicit_and_downloadable(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("UI_DEMO_ENABLED", "true")
    get_settings.cache_clear()
    app = _app().switch_page("app/ui/pages/marking.py").run()

    load_button = next(
        button for button in app.button if button.label == "Load Synthetic Demo"
    )
    app = load_button.click().run()

    assert not app.exception
    assert any("Synthetic demonstration result" in item.value for item in app.warning)

    app = app.switch_page("app/ui/pages/results.py").run()
    assert not app.exception
    assert any("NORMALIZED SCORE OUT OF 100" in item.value for item in app.success)
    assert len(app.get("download_button")) == 1
    assert any(button.label == "Mark another paper" for button in app.button)

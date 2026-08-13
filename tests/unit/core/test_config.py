"""Tests for environment-driven application settings."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import AppEnvironment, LogLevel, Settings


def test_settings_load_safe_local_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings()

    assert settings.app_env is AppEnvironment.DEVELOPMENT
    assert settings.log_level is LogLevel.INFO
    assert settings.data_dir == Path("data")
    assert settings.runtime_data_dir == Path("data/runtime")
    assert settings.database_url is None
    assert str(settings.ollama_base_url) == "http://localhost:11434/"
    assert settings.ocr_confidence_threshold == 0.80
    assert settings.retrieval_top_k == 5
    assert settings.max_pdf_size_mb == 50
    assert settings.max_pdf_pages == 100
    assert settings.pdf_render_dpi == 150


def test_environment_variables_override_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DATA_DIR", "/tmp/ol-marker-test-data")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "8")
    monkeypatch.setenv("MAX_PDF_PAGES", "25")
    monkeypatch.setenv("PDF_RENDER_DPI", "200")

    settings = Settings()

    assert settings.app_env is AppEnvironment.TESTING
    assert settings.log_level is LogLevel.DEBUG
    assert settings.data_dir == Path("/tmp/ol-marker-test-data")
    assert settings.retrieval_top_k == 8
    assert settings.max_pdf_pages == 25
    assert settings.pdf_render_dpi == 200


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("APP_ENV", "staging"),
        ("OCR_CONFIDENCE_THRESHOLD", "1.1"),
        ("GRADING_CONFIDENCE_THRESHOLD", "-0.1"),
        ("RETRIEVAL_TOP_K", "0"),
        ("OLLAMA_BASE_URL", "not-a-url"),
        ("MAX_PDF_SIZE_MB", "0"),
        ("MAX_PDF_PAGES", "0"),
        ("PDF_RENDER_DPI", "601"),
    ],
)
def test_invalid_environment_configuration_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError):
        Settings()

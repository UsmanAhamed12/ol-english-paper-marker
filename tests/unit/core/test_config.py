"""Tests for environment-driven application settings."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import AppEnvironment, LogLevel, Settings


def test_settings_load_safe_local_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_env is AppEnvironment.DEVELOPMENT
    assert settings.log_level is LogLevel.INFO
    assert settings.data_dir == Path("data")
    assert settings.database_url is None
    assert str(settings.ollama_base_url) == "http://localhost:11434/"
    assert settings.ocr_confidence_threshold == 0.80
    assert settings.retrieval_top_k == 5


def test_environment_variables_override_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DATA_DIR", "/tmp/ol-marker-test-data")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "8")

    settings = Settings(_env_file=None)

    assert settings.app_env is AppEnvironment.TESTING
    assert settings.log_level is LogLevel.DEBUG
    assert settings.data_dir == Path("/tmp/ol-marker-test-data")
    assert settings.retrieval_top_k == 8


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("APP_ENV", "staging"),
        ("OCR_CONFIDENCE_THRESHOLD", "1.1"),
        ("GRADING_CONFIDENCE_THRESHOLD", "-0.1"),
        ("RETRIEVAL_TOP_K", "0"),
        ("OLLAMA_BASE_URL", "not-a-url"),
    ],
)
def test_invalid_environment_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)

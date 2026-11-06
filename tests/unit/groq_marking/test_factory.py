"""Tests for Groq provider configuration."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.groq_marking.factory import create_groq_paper_marking_service


def test_factory_requires_an_api_key_before_cloud_marking() -> None:
    settings = Settings(_env_file=None)

    with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
        create_groq_paper_marking_service(settings)

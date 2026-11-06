"""Tests for the configured LangChain Ollama model factory."""

from __future__ import annotations

import pytest
from langchain_ollama import ChatOllama

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.llm.factory import create_grading_chat_model


def test_factory_builds_local_deterministic_json_chat_model() -> None:
    settings = Settings(
        _env_file=None,
        ollama_base_url="http://127.0.0.1:11434",
        ollama_grading_model="synthetic:1b",
        ollama_grading_timeout_seconds=42.0,
    )

    model = create_grading_chat_model(settings)

    assert isinstance(model, ChatOllama)
    assert model.model == "synthetic:1b"
    assert model.base_url == "http://127.0.0.1:11434"
    assert model.format == "json"
    assert model.temperature == 0.0
    assert model.seed == 0
    assert model.num_predict == 1024
    assert model.keep_alive == "5m"
    assert model.sync_client_kwargs == {"timeout": 42.0}


@pytest.mark.parametrize(
    ("base_url", "model"),
    [
        ("http://example.com", "synthetic:1b"),
        ("http://localhost:11434", "synthetic:cloud"),
    ],
)
def test_factory_rejects_non_local_endpoint_or_cloud_model(
    base_url: str,
    model: str,
) -> None:
    settings = Settings(
        _env_file=None,
        ollama_base_url=base_url,
        ollama_grading_model=model,
    )

    with pytest.raises(ConfigurationError):
        create_grading_chat_model(settings)

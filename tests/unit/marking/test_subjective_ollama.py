"""Offline tests for the loopback-only subjective Ollama adapter."""

from __future__ import annotations

from typing import cast

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from ollama import ResponseError

from app.core.exceptions import (
    ConfigurationError,
    SubjectiveMarkingProviderError,
    SubjectiveMarkingTimeoutError,
)
from app.marking.subjective.ollama import (
    OllamaSubjectiveMarkingProvider,
)
from app.marking.subjective.provider import SubjectiveMarkingProvider


class RecordingChatModel:
    def __init__(self, response: str = '{"synthetic":true}') -> None:
        self.response = AIMessage(
            content=response,
            response_metadata={"total_duration": 2_000_000},
        )
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> AIMessage:
        self.prompts.append(prompt)
        return self.response


class FailingChatModel:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def invoke(self, _: str) -> AIMessage:
        raise self.error


def test_provider_uses_configured_model_schema_and_deterministic_options() -> None:
    client = RecordingChatModel(" exact raw response \n")
    provider = OllamaSubjectiveMarkingProvider(
        chat_model=cast(BaseChatModel, client),
        model="synthetic:1b",
        model_version="sha256:synthetic",
    )
    result = provider.assess("synthetic prompt")
    assert client.prompts == ["synthetic prompt"]
    assert result.raw_text == " exact raw response \n"
    assert result.processing_duration_ms == 2
    assert result.model_version == "sha256:synthetic"
    assert isinstance(provider, SubjectiveMarkingProvider)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx.ReadTimeout("private"), SubjectiveMarkingTimeoutError),
        (httpx.ConnectError("private"), SubjectiveMarkingProviderError),
        (ResponseError("private", status_code=500), SubjectiveMarkingProviderError),
    ],
)
def test_provider_maps_transport_errors_without_private_content(
    error: Exception,
    expected: type[Exception],
) -> None:
    provider = OllamaSubjectiveMarkingProvider(
        chat_model=cast(BaseChatModel, FailingChatModel(error)),
        model="synthetic:1b",
    )
    with pytest.raises(expected) as captured:
        provider.assess("private student evidence")
    assert "private student evidence" not in str(captured.value)
    assert "private" not in str(captured.value)


@pytest.mark.parametrize("model", ["grader-cloud", "grader:cloud", ""])
def test_provider_rejects_cloud_or_blank_models(model: str) -> None:
    with pytest.raises(ConfigurationError):
        OllamaSubjectiveMarkingProvider(
            chat_model=cast(BaseChatModel, RecordingChatModel()),
            model=model,
        )

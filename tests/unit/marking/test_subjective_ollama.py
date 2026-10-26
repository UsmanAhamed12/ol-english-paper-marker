"""Offline tests for the loopback-only subjective Ollama adapter."""

from __future__ import annotations

from typing import cast

import pytest

from app.core.exceptions import (
    ConfigurationError,
    SubjectiveMarkingProviderError,
    SubjectiveMarkingTimeoutError,
)
from app.marking.subjective.ollama import (
    DETERMINISTIC_GRADING_OPTIONS,
    OllamaSubjectiveMarkingProvider,
)
from app.marking.subjective.provider import SubjectiveMarkingProvider
from app.ocr.providers.ollama import (
    OllamaChatResponse,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTextClient,
    OllamaTimeoutError,
)


class RecordingTextClient:
    def __init__(self, response: str = '{"synthetic":true}') -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def chat_text(self, **kwargs: object) -> OllamaChatResponse:
        self.calls.append(dict(kwargs))
        return OllamaChatResponse(
            content=self.response,
            resolved_model="synthetic:1b",
            total_duration_ns=2_000_000,
        )


class FailingTextClient:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def chat_text(self, **_: object) -> OllamaChatResponse:
        raise self.error


def test_provider_uses_configured_model_schema_and_deterministic_options() -> None:
    client = RecordingTextClient(" exact raw response \n")
    provider = OllamaSubjectiveMarkingProvider(
        client=cast(OllamaTextClient, client),
        model="synthetic:1b",
        model_version="sha256:synthetic",
    )
    result = provider.assess("synthetic prompt")
    call = client.calls[0]
    assert call["model"] == "synthetic:1b"
    assert call["prompt"] == "synthetic prompt"
    assert call["options"] == DETERMINISTIC_GRADING_OPTIONS
    assert call["response_format"] == "json"
    assert result.raw_text == " exact raw response \n"
    assert result.processing_duration_ms == 2
    assert result.model_version == "sha256:synthetic"
    assert isinstance(provider, SubjectiveMarkingProvider)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (OllamaTimeoutError("private"), SubjectiveMarkingTimeoutError),
        (OllamaConnectionError("private"), SubjectiveMarkingProviderError),
        (OllamaModelNotFoundError("private"), SubjectiveMarkingProviderError),
        (OllamaResponseError("private"), SubjectiveMarkingProviderError),
    ],
)
def test_provider_maps_transport_errors_without_private_content(
    error: Exception,
    expected: type[Exception],
) -> None:
    provider = OllamaSubjectiveMarkingProvider(
        client=cast(OllamaTextClient, FailingTextClient(error)),
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
            client=cast(OllamaTextClient, RecordingTextClient()),
            model=model,
        )

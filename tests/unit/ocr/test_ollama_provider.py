"""Offline tests for the local Ollama OCR adapter."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest

from app.core.exceptions import ConfigurationError, OCRProviderError
from app.domain.models.paper import PaperPage
from app.ocr.base import OCRProvider
from app.ocr.prompts import OCR_TRANSCRIPTION_PROMPT
from app.ocr.providers.ollama import (
    DETERMINISTIC_OCR_OPTIONS,
    OllamaChatResponse,
    OllamaClient,
    OllamaConnectionError,
    OllamaHTTPClient,
    OllamaModelNotFoundError,
    OllamaOCRProvider,
    OllamaResponseError,
    OllamaTimeoutError,
    _validate_local_base_url,
)


class RecordingClient:
    """Synthetic client that records a provider request."""

    def __init__(self, response: str = "raw  text\n") -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def chat(
        self,
        *,
        model: str,
        prompt: str,
        image: bytes,
        options: object,
    ) -> OllamaChatResponse:
        self.calls.append(
            {"model": model, "prompt": prompt, "image": image, "options": options}
        )
        return OllamaChatResponse(content=self.response, resolved_model=model)


class FailingClient:
    """Synthetic client raising one sanitized transport error."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def chat(self, **_: object) -> OllamaChatResponse:
        raise self.error


def _page(tmp_path: Path, content: bytes = b"synthetic-image") -> PaperPage:
    image_path = tmp_path / "page.png"
    image_path.write_bytes(content)
    return PaperPage(
        paper_id=uuid4(),
        page_number=1,
        image_path=image_path.resolve(),
        width=10,
        height=10,
    )


def test_provider_sends_image_model_prompt_and_deterministic_options(
    tmp_path: Path,
) -> None:
    client = RecordingClient()
    provider = OllamaOCRProvider(
        client=cast(OllamaClient, client),
        model="synthetic-vision:1b",
        model_version="sha256:synthetic",
    )

    result = provider.extract_page(_page(tmp_path, b"private-image-bytes"))

    assert client.calls == [
        {
            "model": "synthetic-vision:1b",
            "prompt": OCR_TRANSCRIPTION_PROMPT,
            "image": b"private-image-bytes",
            "options": DETERMINISTIC_OCR_OPTIONS,
        }
    ]
    assert result.raw_text == "raw  text\n"
    assert result.confidence is None
    assert provider.model_version == "sha256:synthetic"
    assert isinstance(provider, OCRProvider)


@pytest.mark.parametrize(
    ("client_error", "safe_message"),
    [
        (
            OllamaModelNotFoundError("private payload"),
            "Configured local Ollama OCR model is unavailable",
        ),
        (OllamaTimeoutError("private payload"), "Local Ollama OCR request timed out"),
        (
            OllamaConnectionError("private payload"),
            "Local Ollama OCR service is unavailable",
        ),
        (
            OllamaResponseError("private payload"),
            "Local Ollama returned an invalid OCR response",
        ),
    ],
)
def test_provider_translates_client_errors_without_private_content(
    tmp_path: Path,
    client_error: Exception,
    safe_message: str,
) -> None:
    provider = OllamaOCRProvider(
        client=cast(OllamaClient, FailingClient(client_error)),
        model="synthetic-vision:1b",
    )

    with pytest.raises(OCRProviderError, match=safe_message) as captured:
        provider.extract_page(_page(tmp_path))

    assert "private payload" not in str(captured.value)


def test_provider_handles_unreadable_image_without_path_leak(tmp_path: Path) -> None:
    page = _page(tmp_path)
    page.image_path.unlink()
    provider = OllamaOCRProvider(
        client=cast(OllamaClient, RecordingClient()),
        model="synthetic-vision:1b",
    )

    with pytest.raises(
        OCRProviderError, match="source image could not be read"
    ) as error:
        provider.extract_page(page)

    assert str(page.image_path) not in str(error.value)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.test",
        "http://192.168.1.2:11434",
        "http://localhost:11434/path",
        "ftp://localhost:11434",
    ],
)
def test_remote_or_invalid_ollama_endpoints_are_rejected(url: str) -> None:
    with pytest.raises(ConfigurationError, match="local|loopback"):
        _validate_local_base_url(url)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:11434/", "http://localhost:11434"),
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("http://[::1]:11434", "http://[::1]:11434"),
    ],
)
def test_loopback_ollama_endpoints_are_accepted(url: str, expected: str) -> None:
    assert _validate_local_base_url(url) == expected


def test_http_client_rejects_malformed_chat_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OllamaHTTPClient(
        base_url="http://localhost:11434",
        timeout_seconds=1,
    )
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_args, **_kwargs: {"model": "x", "done": True},
    )

    with pytest.raises(OllamaResponseError, match="invalid OCR response"):
        client.chat(
            model="synthetic:1b",
            prompt="synthetic prompt",
            image=b"synthetic image",
            options=DETERMINISTIC_OCR_OPTIONS,
        )


@pytest.mark.parametrize("model", ["vision-cloud", "vision:cloud"])
def test_provider_rejects_ollama_cloud_models(model: str) -> None:
    with pytest.raises(ConfigurationError, match="cloud models"):
        OllamaOCRProvider(
            client=cast(OllamaClient, RecordingClient()),
            model=model,
        )


def test_http_client_rejects_nonpositive_timeout_update() -> None:
    client = OllamaHTTPClient(
        base_url="http://localhost:11434",
        timeout_seconds=1,
    )

    with pytest.raises(ConfigurationError, match="timeout must be positive"):
        client.set_timeout(0)

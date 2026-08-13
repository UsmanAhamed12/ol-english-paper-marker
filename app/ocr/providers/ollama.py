"""Local-only Ollama vision adapter implementing the OCR provider contract."""

from __future__ import annotations

import base64
import ipaddress
import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, cast
from urllib.parse import urlparse

from app.core.exceptions import ApplicationError, ConfigurationError, OCRProviderError
from app.domain.models.paper import PaperPage
from app.ocr.models import OCRExtraction
from app.ocr.prompts import OCR_TRANSCRIPTION_PROMPT

GenerationValue = int | float | bool | str

DETERMINISTIC_OCR_OPTIONS: dict[str, GenerationValue] = {
    "temperature": 0.0,
    "seed": 0,
    "num_predict": 2048,
}


class OllamaClientError(ApplicationError):
    """Base class for sanitized local Ollama transport failures."""


class OllamaModelNotFoundError(OllamaClientError):
    """Raised when the configured local model is unavailable."""


class OllamaTimeoutError(OllamaClientError):
    """Raised when local inference exceeds the configured timeout."""


class OllamaConnectionError(OllamaClientError):
    """Raised when the loopback Ollama service cannot be reached."""


class OllamaResponseError(OllamaClientError):
    """Raised when Ollama returns a malformed or unsuccessful response."""


@dataclass(frozen=True)
class OllamaChatResponse:
    """Validated response content required by the provider."""

    content: str
    resolved_model: str
    total_duration_ns: int | None = None


@dataclass(frozen=True)
class OllamaModelInfo:
    """Safe local model metadata exposed by Ollama."""

    model: str
    digest: str


class OllamaClient(Protocol):
    """Small client boundary used to keep provider unit tests offline."""

    def chat(
        self,
        *,
        model: str,
        prompt: str,
        image: bytes,
        options: Mapping[str, GenerationValue],
    ) -> OllamaChatResponse:
        """Run one local, non-streaming vision request."""


class OllamaHTTPClient:
    """Minimal Ollama REST client restricted to a loopback endpoint."""

    def __init__(self, *, base_url: str, timeout_seconds: float) -> None:
        self._base_url = _validate_local_base_url(base_url)
        self._timeout_seconds = timeout_seconds

    def version(self) -> str:
        """Return the reachable local Ollama service version."""

        payload = self._request("GET", "/api/version")
        version = payload.get("version")
        if not isinstance(version, str) or not version:
            raise OllamaResponseError("Local Ollama returned invalid version metadata")
        return version

    def model_info(self, model: str) -> OllamaModelInfo:
        """Resolve an installed model name and immutable digest."""

        # Show first verifies that Ollama can resolve the exact requested tag.
        self._request("POST", "/api/show", {"model": model})
        # /api/show does not consistently expose the manifest digest. The tags
        # endpoint does, so use it as the authoritative local identity.
        tags = self._request("GET", "/api/tags")
        models = tags.get("models")
        if not isinstance(models, list):
            raise OllamaResponseError("Local Ollama returned invalid model metadata")
        for item in models:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            digest = item.get("digest")
            if name == model and isinstance(digest, str) and digest:
                return OllamaModelInfo(model=model, digest=digest)
        raise OllamaModelNotFoundError("Configured local Ollama model is unavailable")

    def chat(
        self,
        *,
        model: str,
        prompt: str,
        image: bytes,
        options: Mapping[str, GenerationValue],
    ) -> OllamaChatResponse:
        """Send one image to the configured local Ollama service."""

        payload = self._request(
            "POST",
            "/api/chat",
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [base64.b64encode(image).decode("ascii")],
                    }
                ],
                "stream": False,
                "think": False,
                "keep_alive": "5m",
                "options": dict(options),
            },
        )
        message = payload.get("message")
        resolved_model = payload.get("model")
        done = payload.get("done")
        if (
            not isinstance(message, dict)
            or not isinstance(resolved_model, str)
            or done is not True
        ):
            raise OllamaResponseError("Local Ollama returned an invalid OCR response")
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaResponseError("Local Ollama returned an invalid OCR response")
        total_duration = payload.get("total_duration")
        if total_duration is not None and not isinstance(total_duration, int):
            raise OllamaResponseError("Local Ollama returned an invalid OCR response")
        return OllamaChatResponse(
            content=content,
            resolved_model=resolved_model,
            total_duration_ns=total_duration,
        )

    def unload(self, model: str) -> None:
        """Ask Ollama to release one model from memory after a sequential run."""

        self._request(
            "POST",
            "/api/generate",
            {"model": model, "keep_alive": 0, "stream": False},
        )

    def set_timeout(self, timeout_seconds: float) -> None:
        """Adjust the request deadline for a bounded benchmark invocation."""

        if timeout_seconds <= 0:
            raise ConfigurationError("Ollama request timeout must be positive")
        self._timeout_seconds = timeout_seconds

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(  # noqa: S310 - loopback URL validated above
                request,
                timeout=self._timeout_seconds,
            ) as response:
                raw_payload = response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise OllamaModelNotFoundError(
                    "Configured local Ollama model is unavailable"
                ) from error
            raise OllamaResponseError("Local Ollama request failed") from error
        except TimeoutError as error:
            raise OllamaTimeoutError("Local Ollama request timed out") from error
        except urllib.error.URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise OllamaTimeoutError("Local Ollama request timed out") from error
            raise OllamaConnectionError(
                "Local Ollama service is unavailable"
            ) from error
        except OSError as error:
            raise OllamaConnectionError(
                "Local Ollama service is unavailable"
            ) from error

        try:
            decoded = json.loads(raw_payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise OllamaResponseError("Local Ollama returned invalid JSON") from error
        if not isinstance(decoded, dict):
            raise OllamaResponseError("Local Ollama returned invalid JSON")
        return cast(dict[str, object], decoded)


class OllamaOCRProvider:
    """Transcribe one page through a configured local Ollama vision model."""

    def __init__(
        self,
        *,
        client: OllamaClient,
        model: str,
        model_version: str | None = None,
        prompt: str = OCR_TRANSCRIPTION_PROMPT,
    ) -> None:
        if not model.strip():
            raise ConfigurationError("Ollama OCR model must not be blank")
        if _is_cloud_model(model):
            raise ConfigurationError("Ollama cloud models are not permitted for OCR")
        self._client = client
        self._model = model
        self._model_version = model_version or model
        self._prompt = prompt

    @property
    def name(self) -> str:
        """Return stable provider provenance."""

        return "ollama"

    @property
    def model_version(self) -> str:
        """Return the resolved model digest when supplied by the runner."""

        return self._model_version

    def extract_page(self, page: PaperPage) -> OCRExtraction:
        """Return the exact model response without fabricating confidence."""

        try:
            image = page.image_path.read_bytes()
        except OSError as error:
            raise OCRProviderError("OCR source image could not be read") from error

        started = perf_counter()
        try:
            response = self._client.chat(
                model=self._model,
                prompt=self._prompt,
                image=image,
                options=DETERMINISTIC_OCR_OPTIONS,
            )
        except OllamaModelNotFoundError as error:
            raise OCRProviderError(
                "Configured local Ollama OCR model is unavailable"
            ) from error
        except OllamaTimeoutError as error:
            raise OCRProviderError("Local Ollama OCR request timed out") from error
        except OllamaConnectionError as error:
            raise OCRProviderError("Local Ollama OCR service is unavailable") from error
        except OllamaResponseError as error:
            raise OCRProviderError(
                "Local Ollama returned an invalid OCR response"
            ) from error

        wall_duration_ms = (perf_counter() - started) * 1000
        duration_ms = (
            response.total_duration_ns / 1_000_000
            if response.total_duration_ns is not None
            else wall_duration_ms
        )
        return OCRExtraction(
            raw_text=response.content,
            confidence=None,
            processing_duration_ms=duration_ms,
        )


def _validate_local_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or parsed.username is not None:
        raise ConfigurationError("OLLAMA_BASE_URL must be a local HTTP endpoint")
    if (
        parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path
        not in {
            "",
            "/",
        }
    ):
        raise ConfigurationError("OLLAMA_BASE_URL must be a local HTTP endpoint")
    host = parsed.hostname
    if host is None:
        raise ConfigurationError("OLLAMA_BASE_URL must be a local HTTP endpoint")
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ConfigurationError("OLLAMA_BASE_URL must use a loopback address")
        except ValueError as error:
            raise ConfigurationError(
                "OLLAMA_BASE_URL must use a loopback address"
            ) from error
    return base_url.rstrip("/")


def _is_cloud_model(model: str) -> bool:
    normalized = model.casefold()
    return normalized.endswith("-cloud") or normalized.endswith(":cloud")

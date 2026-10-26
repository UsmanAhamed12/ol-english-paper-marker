"""Loopback-only Ollama implementation of subjective marking."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter

from app.core.exceptions import (
    ConfigurationError,
    SubjectiveMarkingProviderError,
    SubjectiveMarkingTimeoutError,
)
from app.marking.subjective.provider import SubjectiveProviderResponse
from app.ocr.providers.ollama import (
    GenerationValue,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTextClient,
    OllamaTimeoutError,
    _is_cloud_model,
)

DETERMINISTIC_GRADING_OPTIONS: dict[str, GenerationValue] = {
    "temperature": 0.0,
    "seed": 0,
    "num_predict": 1024,
}


class OllamaSubjectiveMarkingProvider:
    """Request strict JSON from one configured local Ollama model."""

    def __init__(
        self,
        *,
        client: OllamaTextClient,
        model: str,
        model_version: str | None = None,
        options: Mapping[str, GenerationValue] | None = None,
    ) -> None:
        if not model.strip():
            raise ConfigurationError("Ollama grading model must not be blank")
        if _is_cloud_model(model):
            raise ConfigurationError(
                "Ollama cloud models are not permitted for grading"
            )
        self._client = client
        self._model = model
        self._model_version = model_version or model
        self._options = dict(options or DETERMINISTIC_GRADING_OPTIONS)

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_version(self) -> str:
        return self._model_version

    def assess(self, prompt: str) -> SubjectiveProviderResponse:
        started = perf_counter()
        try:
            response = self._client.chat_text(
                model=self._model,
                prompt=prompt,
                options=self._options,
                response_format="json",
            )
        except OllamaTimeoutError as error:
            raise SubjectiveMarkingTimeoutError(
                "Local subjective marking request timed out"
            ) from error
        except (OllamaModelNotFoundError, OllamaConnectionError) as error:
            raise SubjectiveMarkingProviderError(
                "Local subjective marking provider is unavailable"
            ) from error
        except OllamaResponseError as error:
            raise SubjectiveMarkingProviderError(
                "Local subjective marking provider returned an invalid response"
            ) from error
        duration = (
            response.total_duration_ns / 1_000_000
            if response.total_duration_ns is not None
            else (perf_counter() - started) * 1000
        )
        return SubjectiveProviderResponse(
            raw_text=response.content,
            provider=self.name,
            model_version=self._model_version,
            processing_duration_ms=duration,
        )

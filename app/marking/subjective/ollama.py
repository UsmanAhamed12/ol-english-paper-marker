"""Loopback-only Ollama implementation of subjective marking."""

from __future__ import annotations

from time import perf_counter

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from ollama import ResponseError

from app.core.exceptions import (
    ConfigurationError,
    SubjectiveMarkingProviderError,
    SubjectiveMarkingTimeoutError,
)
from app.core.ollama import is_ollama_cloud_model
from app.marking.subjective.provider import SubjectiveProviderResponse


class OllamaSubjectiveMarkingProvider:
    """Request strict JSON from one configured local Ollama model."""

    def __init__(
        self,
        *,
        chat_model: BaseChatModel,
        model: str,
        model_version: str | None = None,
    ) -> None:
        if not model.strip():
            raise ConfigurationError("Ollama grading model must not be blank")
        if is_ollama_cloud_model(model):
            raise ConfigurationError(
                "Ollama cloud models are not permitted for grading"
            )
        self._chat_model = chat_model
        self._model_version = model_version or model

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_version(self) -> str:
        return self._model_version

    def assess(self, prompt: str) -> SubjectiveProviderResponse:
        started = perf_counter()
        try:
            response = self._chat_model.invoke(prompt)
        except (httpx.TimeoutException, TimeoutError) as error:
            raise SubjectiveMarkingTimeoutError(
                "Local subjective marking request timed out"
            ) from error
        except (
            ResponseError,
            httpx.HTTPError,
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            raise SubjectiveMarkingProviderError(
                "Local subjective marking provider is unavailable"
            ) from error
        if not isinstance(response, AIMessage) or not isinstance(response.content, str):
            raise SubjectiveMarkingProviderError(
                "Local subjective marking provider returned an invalid response"
            )
        total_duration_ns = response.response_metadata.get("total_duration")
        duration = (
            total_duration_ns / 1_000_000
            if isinstance(total_duration_ns, int)
            and not isinstance(total_duration_ns, bool)
            else (perf_counter() - started) * 1000
        )
        return SubjectiveProviderResponse(
            raw_text=response.content,
            provider=self.name,
            model_version=self._model_version,
            processing_duration_ms=duration,
        )

"""Factory for the configured local grading chat model."""

from __future__ import annotations

from langchain_ollama import ChatOllama

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.core.ollama import is_ollama_cloud_model, validate_local_ollama_base_url


def create_grading_chat_model(settings: Settings) -> ChatOllama:
    """Build a local JSON-mode ChatOllama with bounded deterministic settings."""

    model = settings.ollama_grading_model
    if not model.strip() or is_ollama_cloud_model(model):
        raise ConfigurationError("A local Ollama grading model is required")
    base_url = validate_local_ollama_base_url(str(settings.ollama_base_url))
    return ChatOllama(
        model=model,
        base_url=base_url,
        format="json",
        temperature=0.0,
        seed=0,
        num_predict=1024,
        keep_alive="5m",
        sync_client_kwargs={"timeout": settings.ollama_grading_timeout_seconds},
    )

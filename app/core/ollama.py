"""Shared local-endpoint and model-name validation for Ollama integrations."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

from app.core.exceptions import ConfigurationError


def validate_local_ollama_base_url(base_url: str) -> str:
    """Require an HTTP(S) URL with a loopback host and no URL credentials."""

    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or parsed.username is not None:
        raise ConfigurationError("OLLAMA_BASE_URL must be a local HTTP endpoint")
    if (
        parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
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


def is_ollama_cloud_model(model: str) -> bool:
    """Return whether a model tag explicitly selects an Ollama cloud model."""

    normalized = model.casefold()
    return normalized.endswith("-cloud") or normalized.endswith(":cloud")

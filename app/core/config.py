"""Strongly typed, environment-driven application configuration."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import AnyHttpUrl, AnyUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnvironment(StrEnum):
    """Supported application runtime environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    """Log levels accepted by the standard-library logging setup."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveTopK = Annotated[int, Field(ge=1, le=100)]
PositiveMegabytes = Annotated[int, Field(ge=1, le=1024)]
PositivePageLimit = Annotated[int, Field(ge=1, le=1000)]
RenderDPI = Annotated[int, Field(ge=72, le=600)]


class Settings(BaseSettings):
    """Application settings loaded from environment variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: AppEnvironment = AppEnvironment.DEVELOPMENT
    log_level: LogLevel = LogLevel.INFO
    data_dir: Path = Path("data")
    runtime_data_dir: Path = Path("data/runtime")

    database_url: AnyUrl | None = None

    ollama_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:11434")
    ollama_grading_model: str = "llama2"
    ollama_ocr_model: str | None = None

    chroma_persist_dir: Path = Path("data/chroma")

    ocr_confidence_threshold: UnitInterval = 0.80
    grading_confidence_threshold: UnitInterval = 0.70
    retrieval_top_k: PositiveTopK = 5

    max_pdf_size_mb: PositiveMegabytes = 50
    max_pdf_pages: PositivePageLimit = 100
    pdf_render_dpi: RenderDPI = 150


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()

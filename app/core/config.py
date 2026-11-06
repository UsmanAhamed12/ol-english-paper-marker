"""Strongly typed, environment-driven application configuration."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import AnyHttpUrl, AnyUrl, Field, SecretStr
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
OllamaTimeoutSeconds = Annotated[float, Field(gt=0.0, le=3600.0)]
TesseractPSM = Annotated[int, Field(ge=0, le=13)]
TesseractTimeoutSeconds = Annotated[float, Field(gt=0.0, le=600.0)]


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
    syllabus_pdf_path: Path = Path(
        "data/syllabus/grade-10-english-syllabus-68d2609d3a1ea.pdf"
    )

    database_url: AnyUrl | None = None

    ollama_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:11434")
    ollama_grading_model: str = "llama2"
    ollama_ocr_model: str | None = None
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_ocr_timeout_seconds: OllamaTimeoutSeconds = 900.0
    ollama_grading_timeout_seconds: OllamaTimeoutSeconds = 300.0
    groq_api_key: SecretStr | None = None
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    groq_timeout_seconds: OllamaTimeoutSeconds = 300.0
    tesseract_executable: Path | None = None
    tesseract_language: str = "eng"
    tesseract_psm: TesseractPSM = 6
    tesseract_timeout_seconds: TesseractTimeoutSeconds = 120.0

    chroma_persist_dir: Path = Path("data/chroma")
    chroma_collection_name: str = "ol_english_knowledge"

    ocr_confidence_threshold: UnitInterval = 0.80
    grading_confidence_threshold: UnitInterval = 0.70
    retrieval_top_k: PositiveTopK = 5

    max_pdf_size_mb: PositiveMegabytes = 50
    max_pdf_pages: PositivePageLimit = 100
    pdf_render_dpi: RenderDPI = 300
    ui_demo_enabled: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()

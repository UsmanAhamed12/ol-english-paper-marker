"""Construction of the configured Groq vision marking service."""

from __future__ import annotations

from groq import Groq

from app.core.config import Settings
from app.core.exceptions import ConfigurationError
from app.groq_marking.service import GroqVisionMarkingService
from app.ingestion.validators import PDFValidator


def create_groq_paper_marking_service(
    settings: Settings,
) -> GroqVisionMarkingService:
    """Build the Groq client and bounded PDF validator from application settings."""

    api_key = (
        settings.groq_api_key.get_secret_value().strip()
        if settings.groq_api_key is not None
        else ""
    )
    if not api_key:
        raise ConfigurationError("GROQ_API_KEY is required for cloud marking")
    client = Groq(api_key=api_key, timeout=settings.groq_timeout_seconds)
    return GroqVisionMarkingService(
        completion_create=client.chat.completions.create,
        model=settings.groq_model,
        pdf_validator=PDFValidator(
            max_file_size_bytes=settings.max_pdf_size_mb * 1024 * 1024,
            max_pages=settings.max_pdf_pages,
        ),
        render_dpi=settings.pdf_render_dpi,
    )

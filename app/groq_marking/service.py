"""Vision extraction and rubric-grounded whole-paper assessment via Groq."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import httpx
import pymupdf
from groq import GroqError
from pydantic import ValidationError

from app.core.exceptions import GroqMarkingError, InvalidPDFError
from app.groq_marking.models import (
    GroqAssessment,
    GroqPaperMarkingResult,
    VisionPageText,
)
from app.ingestion.validators import PDFValidator

_MAX_IMAGE_BYTES = 3_000_000
_EXTRACTION_PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "grading"
    / "groq_page_extraction_v1.txt"
)
_ASSESSMENT_PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "grading"
    / "groq_paper_assessment_v1.txt"
)


class GroqVisionMarkingService:
    """Process both PDFs with one configured vision model and validate its output."""

    def __init__(
        self,
        *,
        completion_create: Callable[..., object],
        model: str,
        pdf_validator: PDFValidator,
        render_dpi: int,
    ) -> None:
        if not model.strip():
            raise ValueError("Groq model must not be blank")
        if not 72 <= render_dpi <= 600:
            raise ValueError("render_dpi must be between 72 and 600")
        self._completion_create = completion_create
        self._model = model
        self._pdf_validator = pdf_validator
        self._render_dpi = render_dpi
        self._extraction_prompt = _EXTRACTION_PROMPT_PATH.read_text(encoding="utf-8")
        self._assessment_prompt = _ASSESSMENT_PROMPT_PATH.read_text(encoding="utf-8")

    def mark_paper(
        self,
        student_pdf: Path,
        marking_scheme_pdf: Path,
        *,
        run_id: UUID | None = None,
    ) -> GroqPaperMarkingResult:
        """Validate, transcribe and assess both PDFs using the same model."""

        student_document = self._pdf_validator.validate(student_pdf)
        scheme_document = self._pdf_validator.validate(marking_scheme_pdf)
        if student_document.source_path == scheme_document.source_path:
            raise InvalidPDFError("Student paper and marking scheme must be different")

        student_pages = self._extract_document(
            student_document.source_path,
            role="student answer paper",
        )
        scheme_pages = self._extract_document(
            scheme_document.source_path,
            role="official marking scheme",
        )
        assessment = self._assess_paper(student_pages, scheme_pages)
        return GroqPaperMarkingResult.from_assessment(
            run_id=run_id or uuid4(),
            model=self._model,
            student_pages=student_pages,
            marking_scheme_pages=scheme_pages,
            assessment=assessment,
        )

    def _extract_document(self, path: Path, *, role: str) -> tuple[VisionPageText, ...]:
        try:
            with pymupdf.open(str(path)) as document:  # type: ignore[no-untyped-call]
                extracted = []
                for page_number, page in enumerate(document, start=1):
                    image = _render_jpeg(page, self._render_dpi)
                    image_data = base64.b64encode(image).decode("ascii")
                    prompt = (
                        f"{self._extraction_prompt}\n\n"
                        f"Document type: {role}. Page number: {page_number}."
                    )
                    response = self._request_json(
                        prompt,
                        image_data=image_data,
                        max_completion_tokens=2048,
                    )
                    extracted.append(
                        VisionPageText.model_validate(
                            {**response, "page_number": page_number}
                        )
                    )
        except ValidationError as error:
            raise GroqMarkingError(
                f"Groq returned invalid transcription data for the {role}."
            ) from error
        except (OSError, RuntimeError, ValueError) as error:
            raise GroqMarkingError(
                f"The {role} could not be rendered or read as a PDF."
            ) from error
        return tuple(extracted)

    def _assess_paper(
        self,
        student_pages: tuple[VisionPageText, ...],
        scheme_pages: tuple[VisionPageText, ...],
    ) -> GroqAssessment:
        prompt = (
            f"{self._assessment_prompt}\n\n"
            f"OFFICIAL MARKING SCHEME PAGES:\n"
            f"{_pages_as_json(scheme_pages)}\n\n"
            f"STUDENT ANSWER PAPER PAGES:\n"
            f"{_pages_as_json(student_pages)}"
        )
        response = self._request_json(prompt, max_completion_tokens=8192)
        try:
            return GroqAssessment.model_validate(response)
        except ValidationError as error:
            raise GroqMarkingError(
                "Groq returned marks that failed question and score validation."
            ) from error

    def _request_json(
        self,
        prompt: str,
        *,
        image_data: str | None = None,
        max_completion_tokens: int,
    ) -> dict[str, object]:
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        if image_data is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}",
                    },
                }
            )
        try:
            response = self._completion_create(
                model=self._model,
                temperature=0,
                max_completion_tokens=max_completion_tokens,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": content}],
            )
        except (GroqError, httpx.HTTPError) as error:
            raise GroqMarkingError(
                "The Groq request failed. Check the API key, model ID, and connection."
            ) from error

        text = _response_content(response)
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise GroqMarkingError(
                "Groq returned a response that was not valid JSON."
            ) from error
        if not isinstance(value, dict):
            raise GroqMarkingError(
                "Groq returned JSON with an invalid top-level shape."
            )
        return cast(dict[str, object], value)


def _render_jpeg(page: pymupdf.Page, render_dpi: int) -> bytes:
    """Render a page under the per-image request budget, lowering detail as needed."""

    dpi_values = tuple(dict.fromkeys((render_dpi, min(render_dpi, 120), 96, 72)))
    for dpi in dpi_values:
        pixmap = page.get_pixmap(dpi=dpi, alpha=False, annots=True)
        for quality in (85, 70, 55):
            image = bytes(
                pixmap.tobytes("jpeg", jpg_quality=quality)  # type: ignore[no-untyped-call]
            )
            if len(image) <= _MAX_IMAGE_BYTES:
                return image
    raise GroqMarkingError(
        "A rendered page exceeds the Groq image-size limit, even at reduced resolution."
    )


def _pages_as_json(pages: tuple[VisionPageText, ...]) -> str:
    return json.dumps(
        [page.model_dump(mode="json") for page in pages],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _response_content(response: object) -> str:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, (list, tuple)) or not choices:
        raise GroqMarkingError("Groq returned no completion choices.")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise GroqMarkingError("Groq returned an empty completion.")
    return content

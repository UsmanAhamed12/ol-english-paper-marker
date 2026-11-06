"""Tests for Groq page extraction, score validation, and aggregation."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pymupdf
import pytest
from pydantic import ValidationError

from app.core.exceptions import GroqMarkingError, InvalidPDFError
from app.groq_marking.models import (
    GroqAssessment,
    GroqPaperMarkingResult,
    GroqQuestionGrade,
)
from app.groq_marking.service import GroqVisionMarkingService
from app.ingestion.validators import PDFValidator

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def _pdf(path: Path, text: str) -> Path:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()
    return path


def _response(value: dict[str, object]) -> object:
    message = SimpleNamespace(content=json.dumps(value))
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _page_text(text: str) -> dict[str, object]:
    return {
        "printed_text": text,
        "handwritten_text": "The student response.",
        "understanding": "English examination page.",
    }


def _assessment(
    *,
    awarded: int = 8,
    maximum: int = 10,
    review_required: bool = False,
) -> dict[str, object]:
    return {
        "subject": "English",
        "overall_feedback": "Clear response.",
        "improvement_tips": ["Support the point with evidence."],
        "questions": [
            {
                "question_number": "Q1",
                "question_summary": "Respond to the passage.",
                "student_answer": "The student response.",
                "awarded_marks": awarded,
                "maximum_marks": maximum,
                "explanation": "Most rubric criteria are met.",
                "feedback": "Add more detail.",
                "improvement_tips": ["Explain the evidence."],
                "review_required": review_required,
            }
        ],
    }


def _service(
    completion_create: Callable[..., object],
) -> GroqVisionMarkingService:
    return GroqVisionMarkingService(
        completion_create=completion_create,
        model=MODEL,
        pdf_validator=PDFValidator(max_file_size_bytes=1_000_000, max_pages=10),
        render_dpi=96,
    )


def test_marks_both_pdfs_with_same_model_and_totals_in_python(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    responses = iter(
        (
            _response(_page_text("Question 1")),
            _response(_page_text("Maximum: 10")),
            _response(_assessment()),
        )
    )

    def complete(**kwargs: object) -> object:
        calls.append(kwargs)
        return next(responses)

    student_pdf = _pdf(tmp_path / "student.pdf", "Student answer")
    scheme_pdf = _pdf(tmp_path / "scheme.pdf", "Official scheme")
    result = _service(complete).mark_paper(student_pdf, scheme_pdf)

    assert len(calls) == 3
    assert all(call["model"] == MODEL for call in calls)
    assert result.total_awarded_marks == 8
    assert result.total_maximum_marks == 10
    assert result.score_out_of_100 == 80
    assert not result.requires_human_review
    assert result.student_pages[0].handwritten_text == "The student response."
    assert result.marking_scheme_pages[0].printed_text == "Maximum: 10"
    first_messages = calls[0]["messages"]
    assert isinstance(first_messages, list)
    first_content = first_messages[0]["content"]
    assert isinstance(first_content, list)
    assert first_content[1]["type"] == "image_url"


def test_invalid_model_score_fails_instead_of_producing_result(tmp_path: Path) -> None:
    responses = iter(
        (
            _response(_page_text("Question 1")),
            _response(_page_text("Maximum: 10")),
            _response(_assessment(awarded=11)),
        )
    )

    def complete(**kwargs: object) -> object:
        return next(responses)

    with pytest.raises(GroqMarkingError, match="failed question and score validation"):
        _service(complete).mark_paper(
            _pdf(tmp_path / "student.pdf", "Student answer"),
            _pdf(tmp_path / "scheme.pdf", "Official scheme"),
        )


def test_unavailable_rubric_suppresses_definitive_score() -> None:
    assessment = GroqAssessment.model_validate(
        _assessment(awarded=0, maximum=0, review_required=True)
    )

    result = GroqPaperMarkingResult.from_assessment(
        run_id=uuid4(),
        model=MODEL,
        student_pages=(),
        marking_scheme_pages=(),
        assessment=assessment,
    )

    assert result.score_out_of_100 is None
    assert result.requires_human_review


@pytest.mark.parametrize(
    ("awarded", "maximum"),
    [(11, 10), (1, 0)],
)
def test_question_model_rejects_out_of_range_marks(
    awarded: int,
    maximum: int,
) -> None:
    question = _assessment(awarded=awarded, maximum=maximum)["questions"]
    assert isinstance(question, list)
    with pytest.raises(ValidationError):
        GroqQuestionGrade.model_validate(question[0])


def test_invalid_upload_is_rejected_before_any_groq_request(tmp_path: Path) -> None:
    calls = 0

    def complete(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return _response(_page_text("never reached"))

    invalid_pdf = tmp_path / "invalid.pdf"
    invalid_pdf.write_text("not a PDF", encoding="utf-8")
    valid_pdf = _pdf(tmp_path / "valid.pdf", "Valid")

    with pytest.raises(InvalidPDFError, match="readable PDF"):
        _service(complete).mark_paper(invalid_pdf, valid_pdf)

    assert calls == 0

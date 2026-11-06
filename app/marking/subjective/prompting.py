"""Versioned prompt construction with explicit untrusted-data separation."""

from __future__ import annotations

import json
from pathlib import Path

from app.answers.models import StructuredStudentAnswer
from app.marking.subjective.models import SyllabusEvidence
from app.marking_scheme.models import MarkingCriterion

SUBJECTIVE_PROMPT_VERSION = "subjective-marking-v2"
_PROMPT_PATH = (
    Path(__file__).resolve().parents[3] / "prompts" / "grading" / "subjective-v2.txt"
)


def build_subjective_prompt(
    criterion: MarkingCriterion,
    answer: StructuredStudentAnswer,
    *,
    syllabus_evidence: tuple[SyllabusEvidence, ...] = (),
    template_path: Path = _PROMPT_PATH,
) -> str:
    """Render one criterion prompt without interpreting student text."""

    template = template_path.read_text(encoding="utf-8")
    criterion_payload = {
        "criterion_id": criterion.criterion_id,
        "test_number": criterion.test_number,
        "criterion_type": criterion.criterion_type.value,
        "description": criterion.description,
        "maximum_marks": (
            str(criterion.maximum_marks)
            if criterion.maximum_marks is not None
            else None
        ),
        "partial_credit_allowed": criterion.partial_credit_allowed,
        "source_page": criterion.provenance.source.page_number,
    }
    student_payload = {
        "text": answer.text,
        "extraction_status": answer.status.value,
        "ocr_confidence": answer.ocr_confidence,
        "ocr_warnings": sorted(
            {
                warning.value
                for item in answer.ocr_fragments
                for warning in item.warnings
            }
        ),
        "pages": list(answer.pages),
    }
    syllabus_payload = [
        {
            "source_id": item.reference.source_id,
            "source_name": item.reference.source_name,
            "source_sha256": item.reference.source_sha256,
            "page_number": item.reference.page_number,
            "text": item.text,
        }
        for item in syllabus_evidence
    ]
    return (
        template.replace(
            "{{CRITERION_JSON}}",
            json.dumps(criterion_payload, ensure_ascii=False, sort_keys=True),
        )
        .replace(
            "{{STUDENT_JSON}}",
            json.dumps(student_payload, ensure_ascii=False, sort_keys=True),
        )
        .replace(
            "{{SYLLABUS_JSON}}",
            json.dumps(syllabus_payload, ensure_ascii=False, sort_keys=True),
        )
    )

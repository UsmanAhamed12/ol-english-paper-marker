"""Deterministic spatial evidence separation within detected Tests."""

from app.evidence.answer_regions import AnswerRegionDetector
from app.evidence.models import (
    DocumentEvidence,
    EvidenceRegion,
    EvidenceType,
    StudentAnswerRegion,
)
from app.evidence.separator import EvidenceSeparator
from app.evidence.service import EvidenceSeparationService

__all__ = [
    "AnswerRegionDetector",
    "DocumentEvidence",
    "EvidenceRegion",
    "EvidenceSeparationService",
    "EvidenceSeparator",
    "EvidenceType",
    "StudentAnswerRegion",
]

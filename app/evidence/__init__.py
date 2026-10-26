"""Deterministic spatial evidence separation within detected Tests."""

from app.evidence.answer_regions import AnswerRegionDetector
from app.evidence.models import (
    DocumentEvidence,
    EvidenceRegion,
    EvidenceType,
    StudentAnswerRegion,
)
from app.evidence.safety_filter import (
    StudentAnswerRegionBuilder,
    StudentCandidateSafetyFilter,
)
from app.evidence.safety_models import (
    CandidateDisposition,
    SafeTestExtraction,
    TestExtractionStatus,
)
from app.evidence.separator import EvidenceSeparator
from app.evidence.service import EvidenceSeparationService

__all__ = [
    "AnswerRegionDetector",
    "CandidateDisposition",
    "DocumentEvidence",
    "EvidenceRegion",
    "EvidenceSeparationService",
    "EvidenceSeparator",
    "EvidenceType",
    "SafeTestExtraction",
    "StudentAnswerRegionBuilder",
    "StudentCandidateSafetyFilter",
    "StudentAnswerRegion",
    "TestExtractionStatus",
]

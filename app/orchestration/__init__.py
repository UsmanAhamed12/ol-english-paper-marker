"""End-to-end paper-marking application boundary."""

from app.orchestration.models import (
    FinalPaperMarkingResult,
    FinalPaperStatus,
    PaperMarkingRequest,
)
from app.orchestration.service import PaperMarkingService

__all__ = [
    "FinalPaperMarkingResult",
    "FinalPaperStatus",
    "PaperMarkingRequest",
    "PaperMarkingService",
]

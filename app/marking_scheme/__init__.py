"""Typed marking-scheme ingestion and Test-alignment boundaries."""

from app.marking_scheme.alignment import TestAlignmentService
from app.marking_scheme.ingestion import MarkingSchemeIngestionService
from app.marking_scheme.models import (
    AlignedPaper,
    AlignedTest,
    AlignmentStatus,
    CriterionComponent,
    CriterionType,
    MarkingCriterion,
    MarkingSchemeTest,
    StructuredMarkingScheme,
)

__all__ = [
    "AlignedPaper",
    "AlignedTest",
    "AlignmentStatus",
    "CriterionComponent",
    "CriterionType",
    "MarkingCriterion",
    "MarkingSchemeIngestionService",
    "MarkingSchemeTest",
    "StructuredMarkingScheme",
    "TestAlignmentService",
]

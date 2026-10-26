"""Injectable boundaries used by the end-to-end orchestration service."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.answers.models import StructuredStudentPaper
from app.marking_scheme.models import StructuredMarkingScheme
from app.orchestration.models import StageSummary


class StudentPipelineOutput(BaseModel):
    """Student artifact plus its internally measured stage summaries."""

    model_config = ConfigDict(frozen=True)

    paper: StructuredStudentPaper
    stages: tuple[StageSummary, ...]


class SchemePipelineOutput(BaseModel):
    """Scheme artifact plus its internally measured stage summaries."""

    model_config = ConfigDict(frozen=True)

    scheme: StructuredMarkingScheme
    stages: tuple[StageSummary, ...]


class InputValidator(Protocol):
    """Validate both untrusted document inputs before processing."""

    def validate(self, path: Path) -> None: ...


class StudentPaperPipeline(Protocol):
    """Produce the existing structured student-paper contract."""

    def process(self, path: Path, *, run_id: UUID) -> StudentPipelineOutput: ...


class MarkingSchemePipeline(Protocol):
    """Produce the existing structured marking-scheme contract."""

    def process(self, path: Path, *, run_id: UUID) -> SchemePipelineOutput: ...

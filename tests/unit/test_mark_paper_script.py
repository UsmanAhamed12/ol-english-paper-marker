"""Tests for the thin Phase 6 CLI delegation boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.orchestration.models import FinalPaperMarkingResult, PaperMarkingRequest
from scripts import mark_paper
from tests.unit.orchestration.test_service import _request, _service


class RecordingService:
    def __init__(self) -> None:
        self.requests: list[PaperMarkingRequest] = []
        self.result = _service()[0].mark_paper(_request())

    def mark_paper(self, request: PaperMarkingRequest) -> FinalPaperMarkingResult:
        self.requests.append(request)
        return self.result


def test_cli_delegates_and_prints_safe_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service = RecordingService()
    monkeypatch.setattr(
        mark_paper,
        "create_local_paper_marking_service",
        lambda _settings: service,
    )
    assert (
        mark_paper.main(
            [
                "--student",
                "student.pdf",
                "--scheme",
                "scheme.pdf",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"status": "completed_with_review"' in output
    assert "Synthetic content answer" not in output
    assert service.requests[0].student_pdf == Path("student.pdf")


def test_cli_parser_requires_both_inputs() -> None:
    with pytest.raises(SystemExit):
        mark_paper.main(["--student", "student.pdf"])

"""Tests for the privacy-safe subjective development smoke."""

import pytest

from scripts.evaluate_subjective_marking import main


def test_synthetic_subjective_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    # The smoke prints aggregate metadata only; no answer or rubric text.
    output = capsys.readouterr().out
    assert '"criteria": 4' in output
    assert '"marked": 4' in output
    assert "synthetic response" not in output

"""Safe CLI tests for the synthetic marking smoke."""

from __future__ import annotations

import json

import pytest

from scripts.evaluate_marking_engine import main, synthetic_summary


def test_synthetic_summary_contains_only_safe_aggregate_results() -> None:
    summary = synthetic_summary()

    assert summary == {
        "tests_evaluated": 6,
        "criteria_evaluated": 6,
        "automatically_marked": 4,
        "review_required": 1,
        "unsupported": 1,
        "confirmed_marks": "2",
        "unresolved_marks": "6",
        "known_maximum": "11",
    }
    assert "Colombo" not in json.dumps(summary)


def test_cli_prints_safe_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tests_evaluated"] == 6


def test_cli_rejects_arguments() -> None:
    with pytest.raises(SystemExit, match="accepts no arguments"):
        main(["private-path"])

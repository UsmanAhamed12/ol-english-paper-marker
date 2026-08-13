"""Tests for standard-library structured logging."""

from __future__ import annotations

import json
import logging

from app.core.config import LogLevel
from app.core.logging import JsonFormatter, configure_logging


def test_json_formatter_includes_structured_context() -> None:
    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="phase complete",
        args=(),
        exc_info=None,
    )
    record.run_id = "run-123"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.test"
    assert payload["message"] == "phase complete"
    assert payload["run_id"] == "run-123"
    assert payload["timestamp"].endswith("+00:00")


def test_configure_logging_sets_requested_root_level() -> None:
    configure_logging(LogLevel.WARNING)

    root_logger = logging.getLogger()
    assert root_logger.level == logging.WARNING
    assert len(root_logger.handlers) == 1
    assert isinstance(root_logger.handlers[0].formatter, JsonFormatter)

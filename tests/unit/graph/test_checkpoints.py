"""Tests for PostgreSQL checkpointer configuration and resource management."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock
from uuid import UUID

import pytest

from app.core.config import Settings
from app.graph.checkpoints import create_postgres_checkpointer, graph_config_for_run


def test_postgres_checkpointer_initializes_and_closes_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql://localhost/ol_english_marker",
    )
    checkpointer = Mock()
    context_manager = MagicMock()
    context_manager.__enter__.return_value = checkpointer
    from_conn_string = Mock(return_value=context_manager)
    monkeypatch.setattr(
        "app.graph.checkpoints.PostgresSaver.from_conn_string",
        from_conn_string,
    )

    with create_postgres_checkpointer(settings) as active_checkpointer:
        assert active_checkpointer is checkpointer

    from_conn_string.assert_called_once_with("postgresql://localhost/ol_english_marker")
    checkpointer.setup.assert_called_once_with()
    context_manager.__exit__.assert_called_once_with(None, None, None)


def test_postgres_checkpointer_requires_database_url() -> None:
    settings = Settings(_env_file=None)

    with (
        pytest.raises(ValueError, match="DATABASE_URL"),
        create_postgres_checkpointer(settings),
    ):
        pytest.fail("A checkpointer should not be created without DATABASE_URL")


def test_postgres_checkpointer_rejects_non_postgres_urls() -> None:
    settings = Settings(_env_file=None, database_url="https://localhost/database")

    with (
        pytest.raises(ValueError, match="PostgreSQL"),
        create_postgres_checkpointer(settings),
    ):
        pytest.fail("A checkpointer should not be created for a non-Postgres URL")


def test_graph_config_uses_run_uuid_as_thread_id() -> None:
    run_id = UUID("30000000-0000-0000-0000-000000000003")

    assert graph_config_for_run(run_id) == {"configurable": {"thread_id": str(run_id)}}

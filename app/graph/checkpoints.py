"""PostgreSQL-backed checkpoint lifecycle and per-run graph configuration."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver

from app.core.config import Settings, get_settings


def graph_config_for_run(run_id: UUID) -> RunnableConfig:
    """Use the opaque workflow run ID as LangGraph's checkpoint thread ID."""

    return {"configurable": {"thread_id": str(run_id)}}


@contextmanager
def create_postgres_checkpointer(
    settings: Settings | None = None,
) -> Generator[PostgresSaver, None, None]:
    """Open and initialize the PostgreSQL checkpointer for one graph lifetime."""

    configured_settings = settings or get_settings()
    database_url = configured_settings.database_url
    if database_url is None:
        raise ValueError("DATABASE_URL is required for PostgreSQL graph checkpoints")
    if database_url.scheme not in {"postgres", "postgresql"}:
        raise ValueError("DATABASE_URL must use a PostgreSQL connection URL")

    with PostgresSaver.from_conn_string(str(database_url)) as checkpointer:
        checkpointer.setup()
        yield checkpointer

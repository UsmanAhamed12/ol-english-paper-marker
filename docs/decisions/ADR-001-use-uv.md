# ADR-001: Use uv for Python project management

## Status

Accepted

## Context

The project needs reproducible dependency resolution, virtual-environment
management, Python runtime selection, and consistent command execution on a
local Apple Silicon development machine. Using separate tools for each concern
would increase setup variation and maintenance overhead.

## Decision

Use uv as the only project package and environment manager. Dependencies are
declared in `pyproject.toml`, resolved in `uv.lock`, and installed with
`uv sync`. Project commands run through `uv run`.

## Alternatives considered

- `pip` plus `venv`: standard and widely available, but requires separate lock
  and Python-version workflows.
- Poetry: provides integrated project management but conflicts with the chosen
  uv workflow and adds another project-specific tool.
- Conda: useful for mixed-language scientific environments but unnecessarily
  heavy for this Python-first project.

## Consequences

- Developers need uv installed.
- The lockfile and project metadata become the reproducibility source of truth.
- Direct `pip`, Poetry, and Conda workflows are unsupported.
- uv can provision the pinned Python runtime consistently across machines.

# ADR-002: Pin the project to Python 3.12

## Status

Accepted

## Context

Phase 0 transient checks ran under Python 3.14 and produced 3.14 cache files,
while the production target is Python 3.12. Allowing arbitrary newer runtimes
would make dependency compatibility and test evidence ambiguous, particularly
for the later AI and data integrations.

## Decision

Pin the repository development runtime to Python 3.12 in `.python-version` and
restrict `requires-python` to `>=3.12,<3.13`. All project commands and quality
checks run through uv and must report Python 3.12.x.

## Alternatives considered

- Python 3.14: available transiently during Phase 0, but not the declared
  production target and potentially ahead of third-party integration support.
- Python 3.13: newer than the target without a demonstrated project benefit.
- `>=3.12` with no upper bound: flexible, but permits accidental validation on
  an unsupported runtime.

## Consequences

- uv may need to download and manage Python 3.12 locally.
- CI and developer commands must use the pinned runtime.
- A deliberate ADR and compatibility validation will be needed before widening
  or changing the supported Python range.

"""Session-scoped human re-verification state for private annotations."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.exceptions import EvidenceSeparationError
from app.evaluation.evidence_benchmark.annotations import (
    AnswerAnnotationStatus,
    EvidenceAnnotation,
    EvidenceAnnotationRepository,
    EvidenceAnnotationStore,
    annotation_fingerprint,
    annotation_record_fingerprint,
    validate_annotation_dataset,
)
from app.evaluation.evidence_benchmark.models import EvidenceBenchmarkManifest


class ReverifiedAnnotation(BaseModel):
    """Fingerprint proving one explicit save in the current session."""

    model_config = ConfigDict(frozen=True)

    sample_id: Annotated[str, Field(pattern=r"^sample_[0-9]{3}$")]
    annotation_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class EvidenceReverificationSession(BaseModel):
    """Private progress that never inherits the old human-verified state."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    session_id: UUID
    started_from_annotation_fingerprint: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$")
    ]
    reverified_annotations: tuple[ReverifiedAnnotation, ...] = ()

    @model_validator(mode="after")
    def identifiers_are_unique(self) -> Self:
        identifiers = [item.sample_id for item in self.reverified_annotations]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Re-verification sample IDs must be unique")
        return self


class EvidenceReverificationRepository:
    """Atomically persist a private re-verification session ledger."""

    def __init__(self, path: Path, *, private_root: Path) -> None:
        root = private_root.resolve()
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise EvidenceSeparationError(
                "Evidence re-verification must remain in private storage"
            )
        self._path = resolved

    def initialize(
        self,
        annotations: EvidenceAnnotationStore,
        *,
        session_id: UUID | None = None,
    ) -> EvidenceReverificationSession:
        """Create a fresh empty ledger or resume the existing private session."""

        if self._path.exists():
            return self.load()
        session = EvidenceReverificationSession(
            session_id=session_id or uuid4(),
            started_from_annotation_fingerprint=annotation_fingerprint(annotations),
        )
        self._write(session)
        return session

    def load(self) -> EvidenceReverificationSession:
        try:
            return EvidenceReverificationSession.model_validate_json(
                self._path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise EvidenceSeparationError(
                "Private evidence re-verification state is invalid"
            ) from error

    def mark_reverified(
        self, annotation: EvidenceAnnotation
    ) -> EvidenceReverificationSession:
        """Record the exact annotation approved by the current human action."""

        session = self.load()
        records = {item.sample_id: item for item in session.reverified_annotations}
        records[annotation.sample_id] = ReverifiedAnnotation(
            sample_id=annotation.sample_id,
            annotation_fingerprint=annotation_record_fingerprint(annotation),
        )
        updated = session.model_copy(
            update={
                "reverified_annotations": tuple(records[key] for key in sorted(records))
            }
        )
        self._write(updated)
        return updated

    def current_ids(self, annotations: EvidenceAnnotationStore) -> frozenset[str]:
        """Return only approvals that still match current semantic content."""

        session = self.load()
        current = {
            item.sample_id: annotation_record_fingerprint(item)
            for item in annotations.annotations
        }
        return frozenset(
            item.sample_id
            for item in session.reverified_annotations
            if current.get(item.sample_id) == item.annotation_fingerprint
        )

    def _write(self, session: EvidenceReverificationSession) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(f".{self._path.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(session.model_dump(mode="json"), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(self._path)
        except OSError as error:
            raise EvidenceSeparationError(
                "Private evidence re-verification state could not be saved"
            ) from error


def validate_reverification_dataset(
    manifest: EvidenceBenchmarkManifest,
    annotations: EvidenceAnnotationStore,
    session: EvidenceReverificationSession,
    samples_root: Path,
) -> None:
    """Require every exact current annotation to be approved this session."""

    validate_annotation_dataset(manifest, annotations, samples_root)
    expected = {sample.sample_id for sample in manifest.samples}
    recorded = {item.sample_id for item in session.reverified_annotations}
    if recorded != expected:
        raise EvidenceSeparationError(
            "Evidence re-verification does not cover every expected sample"
        )
    current = {
        item.sample_id: annotation_record_fingerprint(item)
        for item in annotations.annotations
    }
    if any(
        current.get(item.sample_id) != item.annotation_fingerprint
        for item in session.reverified_annotations
    ):
        raise EvidenceSeparationError(
            "Evidence annotations changed after human re-verification"
        )


def freeze_reverified_annotations(
    manifest: EvidenceBenchmarkManifest,
    annotations: EvidenceAnnotationStore,
    session: EvidenceReverificationSession,
    *,
    samples_root: Path,
    frozen_root: Path,
    private_root: Path,
    timestamp_factory: Callable[[], datetime] | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    """Write a complete, immutable Phase 4C.4R snapshot and provenance."""

    validate_reverification_dataset(manifest, annotations, session, samples_root)
    root = private_root.resolve()
    destination = frozen_root.resolve()
    if not destination.is_relative_to(root):
        raise EvidenceSeparationError(
            "Frozen evidence annotations must remain in private storage"
        )
    fingerprint = annotation_fingerprint(annotations)
    snapshot = destination / f"annotations_{fingerprint}.json"
    provenance = destination / f"provenance_{fingerprint}.json"
    if snapshot.exists() or provenance.exists():
        raise EvidenceSeparationError(
            "Frozen evidence annotation snapshot already exists"
        )
    destination.mkdir(parents=True, exist_ok=True)
    created_at = (timestamp_factory or (lambda: datetime.now(UTC)))().astimezone(UTC)
    snapshot_payload = (
        json.dumps(
            annotations.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    distribution = Counter(
        annotation.evidence_type.value for annotation in annotations.annotations
    )
    provenance_payload: dict[str, object] = {
        "schema_version": "1.0",
        "baseline_version": "phase_4c_4r",
        "relationship": "complete_human_reverified_replacement_baseline",
        "annotation_fingerprint": fingerprint,
        "annotation_count": len(annotations.annotations),
        "reverification_count": len(session.reverified_annotations),
        "class_distribution": dict(sorted(distribution.items())),
        "human_answer_box_count": sum(
            len(item.answer_regions) for item in annotations.annotations
        ),
        "verified_empty_count": sum(
            item.answer_status is AnswerAnnotationStatus.VERIFIED_EMPTY
            for item in annotations.annotations
        ),
        "created_at_utc": created_at.isoformat().replace("+00:00", "Z"),
        "historical_original_fingerprint": (
            "a7007ef2e4887dd7c9b298de0dcb6809df7a222291a4a13104e3576f3c330f2a"
        ),
        "pre_reverification_drifted_fingerprint": (
            "bcb7dd2f85042b03c896d8dd49f1acd1c61cadc45b1a3e1a607378865ae6465e"
        ),
    }
    _write_new_file(snapshot, snapshot_payload)
    try:
        _write_new_file(
            provenance,
            json.dumps(provenance_payload, indent=2, sort_keys=True) + "\n",
        )
    except EvidenceSeparationError:
        snapshot.unlink(missing_ok=True)
        raise
    return snapshot, provenance, provenance_payload


def verify_frozen_annotations(
    manifest: EvidenceBenchmarkManifest,
    snapshot_path: Path,
    provenance_path: Path,
    *,
    samples_root: Path,
    private_root: Path,
) -> str:
    """Independently reload and validate a complete frozen snapshot."""

    snapshot = EvidenceAnnotationRepository(
        manifest,
        snapshot_path,
        private_root=private_root,
    ).load()
    validate_annotation_dataset(manifest, snapshot, samples_root)
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        recorded = provenance["annotation_fingerprint"]
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise EvidenceSeparationError(
            "Frozen evidence annotation provenance is invalid"
        ) from error
    fingerprint = annotation_fingerprint(snapshot)
    if not isinstance(recorded, str) or recorded != fingerprint:
        raise EvidenceSeparationError(
            "Frozen evidence annotation fingerprints do not match"
        )
    return fingerprint


def _write_new_file(path: Path, payload: str) -> None:
    """Atomically create a file while refusing every overwrite."""

    if path.exists():
        raise EvidenceSeparationError("Frozen evidence artifact already exists")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.link(temporary, path)
    except FileExistsError as error:
        raise EvidenceSeparationError(
            "Frozen evidence artifact already exists"
        ) from error
    except OSError as error:
        raise EvidenceSeparationError(
            "Frozen evidence artifact could not be written"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)

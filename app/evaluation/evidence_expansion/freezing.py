"""Immutable freezing and recovery validation for evidence-v2 annotations."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.exceptions import EvidenceSeparationError
from app.evaluation.evidence_benchmark.annotations import (
    AnswerAnnotationStatus,
    EvidenceAnnotationRepository,
    EvidenceAnnotationStore,
    annotation_fingerprint,
    validate_annotation_dataset,
)
from app.evaluation.evidence_expansion.models import EvidenceExpansionManifest


class EvidenceExpansionFrozenProvenance(BaseModel):
    """Private safe metadata linking a complete snapshot to its source manifest."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["2.0"] = "2.0"
    baseline_version: Literal["phase_4c_5b"] = "phase_4c_5b"
    annotation_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    manifest_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    annotation_count: Annotated[int, Field(gt=0)]
    class_distribution: dict[str, Annotated[int, Field(ge=0)]]
    samples_with_answer_regions: Annotated[int, Field(ge=0)]
    verified_empty_count: Annotated[int, Field(ge=0)]
    human_answer_box_count: Annotated[int, Field(ge=0)]
    created_at_utc: datetime
    phase_4c_4r_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    ocr_benchmark_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    corpus_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    canonical_snapshot_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def expansion_manifest_fingerprint(manifest: EvidenceExpansionManifest) -> str:
    """Hash canonical semantic manifest data for private source provenance."""

    payload = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def freeze_expansion_annotations(
    manifest: EvidenceExpansionManifest,
    annotations: EvidenceAnnotationStore,
    *,
    samples_root: Path,
    frozen_root: Path,
    private_root: Path,
    phase_4c_4r_fingerprint: str,
    ocr_benchmark_fingerprint: str,
    corpus_sha256: str,
    canonical_snapshot_sha256: str,
    timestamp_factory: Callable[[], datetime] | None = None,
) -> tuple[Path, Path, EvidenceExpansionFrozenProvenance]:
    """Atomically create a complete, non-overwriting evidence-v2 snapshot."""

    validate_annotation_dataset(manifest, annotations, samples_root)
    root = private_root.resolve()
    destination = frozen_root.resolve()
    if not destination.is_relative_to(root):
        raise EvidenceSeparationError(
            "Frozen evidence-v2 annotations must remain in private storage"
        )
    fingerprint = annotation_fingerprint(annotations)
    snapshot = destination / f"annotations_{fingerprint}.json"
    provenance_path = destination / f"provenance_{fingerprint}.json"
    if snapshot.exists() or provenance_path.exists():
        raise EvidenceSeparationError("Frozen evidence-v2 snapshot already exists")
    distribution = Counter(item.evidence_type.value for item in annotations.annotations)
    provenance = EvidenceExpansionFrozenProvenance(
        annotation_fingerprint=fingerprint,
        manifest_fingerprint=expansion_manifest_fingerprint(manifest),
        annotation_count=len(annotations.annotations),
        class_distribution=dict(sorted(distribution.items())),
        samples_with_answer_regions=sum(
            bool(item.answer_regions) for item in annotations.annotations
        ),
        verified_empty_count=sum(
            item.answer_status is AnswerAnnotationStatus.VERIFIED_EMPTY
            for item in annotations.annotations
        ),
        human_answer_box_count=sum(
            len(item.answer_regions) for item in annotations.annotations
        ),
        created_at_utc=(timestamp_factory or (lambda: datetime.now(UTC)))().astimezone(
            UTC
        ),
        phase_4c_4r_fingerprint=phase_4c_4r_fingerprint,
        ocr_benchmark_fingerprint=ocr_benchmark_fingerprint,
        corpus_sha256=corpus_sha256,
        canonical_snapshot_sha256=canonical_snapshot_sha256,
    )
    destination.mkdir(parents=True, exist_ok=True)
    _write_new_file(
        snapshot,
        json.dumps(
            annotations.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    try:
        _write_new_file(
            provenance_path,
            json.dumps(
                provenance.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
    except EvidenceSeparationError:
        snapshot.unlink(missing_ok=True)
        raise
    return snapshot, provenance_path, provenance


def verify_frozen_expansion_annotations(
    manifest: EvidenceExpansionManifest,
    snapshot_path: Path,
    provenance_path: Path,
    *,
    samples_root: Path,
    private_root: Path,
) -> tuple[EvidenceAnnotationStore, EvidenceExpansionFrozenProvenance]:
    """Independently recover, validate, and fingerprint a complete snapshot."""

    snapshot = EvidenceAnnotationRepository(
        manifest,
        snapshot_path,
        private_root=private_root,
    ).load()
    validate_annotation_dataset(manifest, snapshot, samples_root)
    try:
        provenance = EvidenceExpansionFrozenProvenance.model_validate_json(
            provenance_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise EvidenceSeparationError(
            "Frozen evidence-v2 provenance is invalid"
        ) from error
    if (
        annotation_fingerprint(snapshot) != provenance.annotation_fingerprint
        or expansion_manifest_fingerprint(manifest) != provenance.manifest_fingerprint
        or len(snapshot.annotations) != provenance.annotation_count
        or sum(len(item.answer_regions) for item in snapshot.annotations)
        != provenance.human_answer_box_count
    ):
        raise EvidenceSeparationError(
            "Frozen evidence-v2 snapshot provenance does not match"
        )
    return snapshot, provenance


def _write_new_file(path: Path, payload: str) -> None:
    """Atomically create a private artifact while refusing overwrite."""

    if path.exists():
        raise EvidenceSeparationError("Frozen evidence-v2 artifact already exists")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.link(temporary, path)
    except FileExistsError as error:
        raise EvidenceSeparationError(
            "Frozen evidence-v2 artifact already exists"
        ) from error
    except OSError as error:
        raise EvidenceSeparationError(
            "Frozen evidence-v2 artifact could not be written"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)

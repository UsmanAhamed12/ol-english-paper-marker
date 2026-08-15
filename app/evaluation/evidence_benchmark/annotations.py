"""Private human visual annotations for the evidence benchmark."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.exceptions import EvidenceSeparationError
from app.evaluation.evidence_benchmark.models import (
    EvidenceBenchmarkManifest,
    GroundTruthAnswerRegion,
)
from app.evaluation.evidence_expansion.models import EvidenceExpansionManifest
from app.evaluation.teacher_evidence.models import TeacherEvidenceManifest
from app.evidence.models import EvidenceType


class AnswerAnnotationStatus(StrEnum):
    """Explicitly distinguish drawn regions from verified-empty samples."""

    ANNOTATED = "annotated"
    VERIFIED_EMPTY = "verified_empty"


class EvidenceAnnotation(BaseModel):
    """One complete, explicit human visual-label decision."""

    model_config = ConfigDict(frozen=True)

    sample_id: Annotated[
        str, Field(pattern=r"^(?:sample|evidence_v2|evidence_teacher_v1)_[0-9]{3}$")
    ]
    evidence_type: EvidenceType
    answer_status: AnswerAnnotationStatus
    answer_regions: tuple[GroundTruthAnswerRegion, ...] = ()
    human_verified: Literal[True] = True

    @model_validator(mode="after")
    def answer_state_is_explicit(self) -> Self:
        if (
            self.answer_status is AnswerAnnotationStatus.ANNOTATED
            and not self.answer_regions
        ):
            raise ValueError("Annotated answer state requires at least one rectangle")
        if (
            self.answer_status is AnswerAnnotationStatus.VERIFIED_EMPTY
            and self.answer_regions
        ):
            raise ValueError("Verified-empty answer state cannot contain rectangles")
        return self


class EvidenceAnnotationStore(BaseModel):
    """Versioned private collection saved independently from the worksheet."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    annotations: tuple[EvidenceAnnotation, ...] = ()

    @model_validator(mode="after")
    def identifiers_are_unique(self) -> Self:
        identifiers = [annotation.sample_id for annotation in self.annotations]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Evidence annotation sample IDs must be unique")
        return self


class EvidenceAnnotationRepository:
    """Validate and atomically persist annotations beneath private storage."""

    def __init__(
        self,
        manifest: (
            EvidenceBenchmarkManifest
            | EvidenceExpansionManifest
            | TeacherEvidenceManifest
        ),
        output_path: Path,
        *,
        private_root: Path,
        timestamp_factory: Callable[[], datetime] | None = None,
    ) -> None:
        root = private_root.resolve()
        path = output_path.resolve()
        if not path.is_relative_to(root):
            raise EvidenceSeparationError(
                "Evidence annotations must remain in private evaluation storage"
            )
        self._manifest = manifest
        self._samples = {sample.sample_id: sample for sample in manifest.samples}
        self._path = path
        self._private_root = root
        self._timestamp_factory = timestamp_factory or (lambda: datetime.now(UTC))

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> EvidenceAnnotationStore:
        if not self._path.exists():
            return EvidenceAnnotationStore()
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            _validate_raw_annotation_geometry(payload)
            return EvidenceAnnotationStore.model_validate(payload)
        except (OSError, TypeError, ValueError) as error:
            raise EvidenceSeparationError(
                "Private evidence annotations are invalid"
            ) from error

    def save(self, annotation: EvidenceAnnotation) -> EvidenceAnnotationStore:
        sample = self._samples.get(annotation.sample_id)
        if sample is None:
            raise EvidenceSeparationError("Evidence annotation sample is unknown")
        if any(
            region.bbox.x + region.bbox.width > sample.region.width
            or region.bbox.y + region.bbox.height > sample.region.height
            for region in annotation.answer_regions
        ):
            raise EvidenceSeparationError(
                "Answer annotation exceeds the private sample crop"
            )
        previous = self.load()
        existing = {item.sample_id: item for item in previous.annotations}
        existing[annotation.sample_id] = annotation
        store = EvidenceAnnotationStore(
            annotations=tuple(existing[key] for key in sorted(existing))
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.is_file() and annotation_fingerprint(
            store
        ) != annotation_fingerprint(previous):
            self._backup_current_store()
        temporary = self._path.with_name(f".{self._path.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(store.model_dump(mode="json"), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(self._path)
        except OSError as error:
            raise EvidenceSeparationError(
                "Private evidence annotation could not be saved"
            ) from error
        return store

    def _backup_current_store(self) -> Path:
        """Preserve the current valid file before a semantic replacement."""

        current = self.load()
        fingerprint = annotation_fingerprint(current)
        timestamp = self._timestamp_factory().astimezone(UTC)
        stamp = timestamp.strftime("%Y%m%dT%H%M%S.%fZ")
        backup_root = self._private_root / "backups"
        backup_root.mkdir(parents=True, exist_ok=True)
        backup = backup_root / f"annotations_{stamp}_{fingerprint}.json"
        if backup.exists():
            raise EvidenceSeparationError("Annotation backup already exists")
        temporary = backup.with_name(f".{backup.name}.tmp")
        try:
            temporary.write_bytes(self._path.read_bytes())
            temporary.replace(backup)
        except OSError as error:
            raise EvidenceSeparationError(
                "Previous evidence annotations could not be backed up"
            ) from error
        return backup

    def completion(self) -> tuple[int, int]:
        return len(self.load().annotations), len(self._manifest.samples)


def validate_annotation_dataset(
    manifest: (
        EvidenceBenchmarkManifest | EvidenceExpansionManifest | TeacherEvidenceManifest
    ),
    store: EvidenceAnnotationStore,
    samples_root: Path,
) -> None:
    """Require a complete, exact, geometry-valid private annotation dataset."""

    expected = {sample.sample_id: sample for sample in manifest.samples}
    actual = {annotation.sample_id: annotation for annotation in store.annotations}
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing:
        raise EvidenceSeparationError(
            f"Evidence annotations are missing expected sample {missing[0]}"
        )
    if unexpected:
        raise EvidenceSeparationError(
            f"Evidence annotations contain unexpected sample {unexpected[0]}"
        )
    if len(store.annotations) != len(expected):
        raise EvidenceSeparationError("Evidence annotation count is invalid")
    for sample_id, annotation in actual.items():
        sample = expected[sample_id]
        if annotation.human_verified is not True:
            raise EvidenceSeparationError(
                f"Evidence annotation is not human verified: {sample_id}"
            )
        if any(
            region.bbox.x + region.bbox.width > sample.region.width
            or region.bbox.y + region.bbox.height > sample.region.height
            for region in annotation.answer_regions
        ):
            raise EvidenceSeparationError(
                f"Evidence answer rectangle exceeds crop: {sample_id}"
            )
        if not (samples_root / f"{sample_id}.png").is_file():
            raise EvidenceSeparationError(
                f"Private evidence sample image is missing: {sample_id}"
            )


def annotation_fingerprint(store: EvidenceAnnotationStore) -> str:
    """Hash a canonical representation of the finalized human labels."""

    canonical = EvidenceAnnotationStore(
        annotations=tuple(sorted(store.annotations, key=lambda item: item.sample_id))
    ).model_dump(mode="json")
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def annotation_record_fingerprint(annotation: EvidenceAnnotation) -> str:
    """Hash one canonical annotation for session-scoped re-verification."""

    payload = json.dumps(
        annotation.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def freeze_annotation_provenance(
    store: EvidenceAnnotationStore,
    output_path: Path,
    *,
    private_root: Path,
) -> dict[str, object]:
    """Persist safe frozen-label metadata without changing the annotation store."""

    root = private_root.resolve()
    path = output_path.resolve()
    if not path.is_relative_to(root):
        raise EvidenceSeparationError(
            "Evidence annotation provenance must remain in private storage"
        )
    distribution = Counter(
        annotation.evidence_type.value for annotation in store.annotations
    )
    with_regions = sum(bool(item.answer_regions) for item in store.annotations)
    verified_empty = sum(
        item.answer_status is AnswerAnnotationStatus.VERIFIED_EMPTY
        for item in store.annotations
    )
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "annotation_fingerprint": annotation_fingerprint(store),
        "sample_count": len(store.annotations),
        "class_distribution": dict(sorted(distribution.items())),
        "samples_with_answer_regions": with_regions,
        "verified_empty_samples": verified_empty,
        "human_answer_box_count": sum(
            len(item.answer_regions) for item in store.annotations
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError as error:
        raise EvidenceSeparationError(
            "Private annotation provenance could not be written"
        ) from error
    return payload


def _validate_raw_annotation_geometry(payload: object) -> None:
    """Reject textual, floating-point, or otherwise coercible box coordinates."""

    if not isinstance(payload, dict) or not isinstance(
        payload.get("annotations"), list
    ):
        raise TypeError("Annotation payload must contain an array")
    for record in payload["annotations"]:
        if not isinstance(record, dict) or not isinstance(
            record.get("answer_regions"), list
        ):
            raise TypeError("Annotation answer regions must be an array")
        for region in record["answer_regions"]:
            if not isinstance(region, dict) or not isinstance(region.get("bbox"), dict):
                raise TypeError("Annotation rectangle must contain a bounding box")
            box = region["bbox"]
            if set(box) != {"x", "y", "width", "height"} or any(
                type(box[key]) is not int for key in ("x", "y", "width", "height")
            ):
                raise TypeError("Annotation rectangle coordinates must be integers")

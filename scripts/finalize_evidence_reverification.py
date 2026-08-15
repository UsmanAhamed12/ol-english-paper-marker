"""Freeze and measure the complete private Phase 4C.4R human baseline."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.core.exceptions import EvidenceSeparationError
from app.evaluation.evidence_benchmark.annotations import (
    EvidenceAnnotationRepository,
    annotation_fingerprint,
)
from app.evaluation.evidence_benchmark.evaluation import evaluate_evidence_benchmark
from app.evaluation.evidence_benchmark.manifest import load_evidence_manifest
from app.evaluation.evidence_benchmark.reverification import (
    EvidenceReverificationRepository,
    freeze_reverified_annotations,
    validate_reverification_dataset,
    verify_frozen_annotations,
)
from app.evidence.models import TestEvidence
from scripts.prepare_evidence_benchmark import prepare_private_artifacts

DEFAULT_MANIFEST = Path("data/evaluation/evidence/benchmark_manifest.json")
PHASE_NAMESPACE = "phase4c4r"


def finalize_reverified_baseline(manifest_path: Path) -> dict[str, Any]:
    """Validate, freeze, independently reload, rerun, and measure Phase 4C.4R."""

    resolved_manifest = manifest_path.resolve(strict=True)
    root = resolved_manifest.parent
    manifest = load_evidence_manifest(resolved_manifest)
    annotation_repository = EvidenceAnnotationRepository(
        manifest,
        root / "annotations.json",
        private_root=root,
    )
    annotations = annotation_repository.load()
    reverification_repository = EvidenceReverificationRepository(
        root / "reverification_session.json",
        private_root=root,
    )
    session = reverification_repository.load()
    samples_root = root / "samples"
    validate_reverification_dataset(manifest, annotations, session, samples_root)
    live_fingerprint = annotation_fingerprint(annotations)

    snapshot_path, provenance_path, provenance = freeze_reverified_annotations(
        manifest,
        annotations,
        session,
        samples_root=samples_root,
        frozen_root=root / "frozen",
        private_root=root,
    )
    frozen_fingerprint = verify_frozen_annotations(
        manifest,
        snapshot_path,
        provenance_path,
        samples_root=samples_root,
        private_root=root,
    )
    if frozen_fingerprint != live_fingerprint:
        raise EvidenceSeparationError(
            "Live and frozen evidence annotation fingerprints do not match"
        )

    prepare_private_artifacts(
        resolved_manifest,
        artifact_namespace=PHASE_NAMESPACE,
    )
    if annotation_fingerprint(annotation_repository.load()) != live_fingerprint:
        raise EvidenceSeparationError(
            "Human annotations changed during Phase 4C.4R evaluation"
        )
    validate_reverification_dataset(
        manifest,
        annotation_repository.load(),
        reverification_repository.load(),
        samples_root,
    )
    frozen_annotations = EvidenceAnnotationRepository(
        manifest,
        snapshot_path,
        private_root=root,
    ).load()
    predictions_path = root / "results" / PHASE_NAMESPACE / "candidate_predictions.json"
    predictions = _load_predictions(predictions_path)
    report = evaluate_evidence_benchmark(manifest, frozen_annotations, predictions)
    if report.annotation_fingerprint != live_fingerprint:
        raise EvidenceSeparationError(
            "Phase 4C.4R evaluation does not match the frozen annotations"
        )
    result_path = root / "results" / PHASE_NAMESPACE / "baseline_evaluation.json"
    _write_new_json(
        result_path,
        {
            "schema_version": "1.0",
            "baseline_version": "phase_4c_4r",
            "annotation_provenance": provenance,
            "evaluation": report.model_dump(mode="json"),
        },
    )
    return {
        "annotation_fingerprint": live_fingerprint,
        "snapshot_verified": True,
        "reverification_count": len(session.reverified_annotations),
        "class_distribution": provenance["class_distribution"],
        "human_answer_box_count": provenance["human_answer_box_count"],
        "verified_empty_count": provenance["verified_empty_count"],
        "classification": report.classification.model_dump(mode="json"),
        "answer_localization": report.answer_localization.model_dump(mode="json"),
        "category_classification_errors": report.category_classification_errors,
        "category_answer_misses": report.category_answer_misses,
        "category_answer_extras": report.category_answer_extras,
        "private_snapshot_written": True,
        "private_result_written": True,
    }


def _load_predictions(path: Path) -> dict[str, TestEvidence]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload["samples"]
        if not isinstance(records, list):
            raise TypeError
        predictions = {
            record["sample_id"]: TestEvidence.model_validate(record["evidence"])
            for record in records
        }
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise EvidenceSeparationError(
            "Private Phase 4C.4R predictions are invalid"
        ) from error
    if len(predictions) != len(records):
        raise EvidenceSeparationError(
            "Private Phase 4C.4R predictions contain duplicates"
        )
    return predictions


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise EvidenceSeparationError("Phase 4C.4R result already exists")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.link(temporary, path)
    except FileExistsError as error:
        raise EvidenceSeparationError("Phase 4C.4R result already exists") from error
    except OSError as error:
        raise EvidenceSeparationError(
            "Private Phase 4C.4R result could not be written"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError("Phase 4C.4R finalization accepts no arguments")
    print(json.dumps(finalize_reverified_baseline(DEFAULT_MANIFEST), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

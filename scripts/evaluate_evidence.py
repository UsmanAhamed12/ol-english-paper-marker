"""Freeze and evaluate the private human-labeled evidence benchmark."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.core.exceptions import EvidenceSeparationError
from app.evaluation.evidence_benchmark.annotations import (
    EvidenceAnnotationRepository,
    annotation_fingerprint,
    freeze_annotation_provenance,
    validate_annotation_dataset,
)
from app.evaluation.evidence_benchmark.evaluation import (
    evaluate_evidence_benchmark,
)
from app.evaluation.evidence_benchmark.evaluation_overlay import (
    render_evaluation_overlay,
)
from app.evaluation.evidence_benchmark.manifest import load_evidence_manifest
from app.evidence.models import TestEvidence
from scripts.prepare_evidence_benchmark import prepare_private_artifacts

DEFAULT_MANIFEST = Path("data/evaluation/evidence/benchmark_manifest.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def run_evaluation(manifest_path: Path) -> dict[str, Any]:
    """Validate, freeze, rerun, evaluate, and persist private safe metrics."""

    resolved_manifest = manifest_path.resolve(strict=True)
    root = resolved_manifest.parent
    manifest = load_evidence_manifest(resolved_manifest)
    repository = EvidenceAnnotationRepository(
        manifest,
        root / "annotations.json",
        private_root=root,
    )
    annotations = repository.load()
    validate_annotation_dataset(manifest, annotations, root / "samples")
    provenance = freeze_annotation_provenance(
        annotations,
        root / "annotation_provenance.json",
        private_root=root,
    )
    frozen_fingerprint = annotation_fingerprint(annotations)

    prepare_private_artifacts(resolved_manifest)
    if annotation_fingerprint(repository.load()) != frozen_fingerprint:
        raise EvidenceSeparationError(
            "Human annotation fingerprint changed during evidence evaluation"
        )
    predictions = _load_predictions(root / "results" / "candidate_predictions.json")
    report = evaluate_evidence_benchmark(manifest, annotations, predictions)
    if report.annotation_fingerprint != frozen_fingerprint:
        raise EvidenceSeparationError(
            "Evidence evaluation does not match frozen human annotations"
        )
    comparison_root = root / "evaluation-overlays"
    annotation_by_id = {
        annotation.sample_id: annotation for annotation in annotations.annotations
    }
    for sample in manifest.samples:
        render_evaluation_overlay(
            sample,
            annotation_by_id[sample.sample_id],
            predictions[sample.sample_id],
            root / "samples" / f"{sample.sample_id}.png",
            comparison_root / f"{sample.sample_id}.png",
        )
    result_path = root / "results" / "evidence_baseline_evaluation.json"
    _write_json(
        result_path,
        {
            "schema_version": "1.0",
            "benchmark_evaluated": True,
            "annotation_provenance": provenance,
            "evaluation": report.model_dump(mode="json"),
        },
    )
    return {
        "sample_count": len(manifest.samples),
        "annotation_fingerprint": frozen_fingerprint,
        "class_distribution": provenance["class_distribution"],
        "samples_with_answer_regions": provenance["samples_with_answer_regions"],
        "verified_empty_samples": provenance["verified_empty_samples"],
        "human_answer_box_count": provenance["human_answer_box_count"],
        "classification": report.classification.model_dump(mode="json"),
        "answer_localization": report.answer_localization.model_dump(mode="json"),
        "category_classification_errors": report.category_classification_errors,
        "category_answer_misses": report.category_answer_misses,
        "category_answer_extras": report.category_answer_extras,
        "private_results_written": True,
        "private_overlays_written": len(manifest.samples),
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
            "Private evidence predictions are invalid"
        ) from error
    if len(predictions) != len(records):
        raise EvidenceSeparationError("Private evidence predictions contain duplicates")
    return predictions


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError as error:
        raise EvidenceSeparationError(
            "Private evidence evaluation could not be written"
        ) from error


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    print(json.dumps(run_evaluation(arguments.manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

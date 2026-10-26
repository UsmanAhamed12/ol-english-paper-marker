"""Run the fixed Phase 4C.6R post-V1 safety policy on frozen benchmarks."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from app.evaluation.evidence_benchmark.annotations import (
    AnswerAnnotationStatus,
    EvidenceAnnotation,
)
from app.evaluation.evidence_benchmark.evaluation_overlay import (
    render_evaluation_overlay,
)
from app.evaluation.evidence_benchmark.metrics import EvidenceClassMetrics
from app.evaluation.evidence_benchmark.models import EvidenceBenchmarkSample
from app.evaluation.evidence_expansion.models import EvidenceExpansionSample
from app.evaluation.evidence_redesign.runner import FrozenEvidenceDataset
from app.evaluation.evidence_safety.models import SafetyFilterBenchmarkResult
from app.evaluation.evidence_safety.runner import run_safety_filter
from app.evaluation.teacher_evidence.models import TeacherEvidenceSample
from app.evidence.models import EvidenceType, TestEvidence
from app.evidence.safety_filter import SAFETY_POLICY_VERSION
from app.evidence.safety_models import SafeTestExtraction
from scripts.evaluate_evidence_v2 import (
    _code_version,
    _load_datasets,
    _verify_all_integrity,
    _write_new_json,
)

OUTPUT_ROOT = Path("data/evaluation/evidence_safety/phase4c6r")
Sample = EvidenceBenchmarkSample | EvidenceExpansionSample | TeacherEvidenceSample


def run_experiment(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    """Persist one non-overwriting, privacy-safe experiment bundle."""

    integrity_before = _verify_all_integrity()
    datasets = _load_datasets()
    runs = tuple((dataset, *run_safety_filter(dataset)) for dataset in datasets)
    created = datetime.now(UTC)
    bundle = output_root.resolve() / created.strftime("%Y%m%dT%H%M%S.%fZ")
    bundle.mkdir(parents=True, exist_ok=False)
    for dataset, _result, extractions, predictions in runs:
        _write_new_json(
            bundle / f"{dataset.dataset_id}-private-predictions.json",
            {
                "schema_version": "1.0",
                "policy_version": SAFETY_POLICY_VERSION,
                "annotation_fingerprint": dataset.fingerprint,
                "samples": [
                    {
                        "sample_id": sample_id,
                        "status": extractions[sample_id].status.value,
                        "decisions": [
                            {
                                "evidence_index": decision.evidence_index,
                                "disposition": decision.disposition.value,
                                "reasons": [
                                    reason.value for reason in decision.reasons
                                ],
                            }
                            for decision in extractions[sample_id].decisions
                        ],
                        "answer_regions": [
                            region.model_dump(mode="json")
                            for region in predictions[sample_id].answer_regions
                        ],
                    }
                    for sample_id in sorted(predictions)
                ],
            },
        )
    overlay_count = _render_overlays(bundle / "overlays", runs)
    integrity_after = _verify_all_integrity()
    if integrity_before != integrity_after:
        raise RuntimeError("Frozen integrity changed during safety evaluation")
    report = {
        "schema_version": "1.0",
        "experiment": "phase_4c_6r",
        "policy_version": SAFETY_POLICY_VERSION,
        "created_at_utc": created.isoformat(),
        "code_version": _code_version(),
        "integrity": integrity_after,
        "overlay_count": overlay_count,
        "benchmarks": {
            result.dataset_id: _safe_summary(result) for _, result, _, _ in runs
        },
    }
    _write_new_json(bundle / "experiment.json", report)
    return {
        "bundle": str(bundle.relative_to(Path.cwd().resolve())),
        "policy_version": SAFETY_POLICY_VERSION,
        "overlay_count": overlay_count,
        "benchmarks": report["benchmarks"],
        "integrity": integrity_after,
    }


def _safe_summary(result: SafetyFilterBenchmarkResult) -> dict[str, Any]:
    classification = result.evaluation.classification
    localization = result.evaluation.answer_localization
    return {
        "accuracy": classification.overall_accuracy,
        "macro_f1": classification.macro_f1,
        "unknown_rate": classification.unknown_prediction_rate,
        "classes": _classes(classification.per_class),
        "safe_tests": result.safe_tests,
        "review_required_tests": result.review_required_tests,
        "empty_tests": result.empty_tests,
        "safe_candidates": result.safe_candidates,
        "quarantined_candidates": result.quarantined_candidates,
        "rejected_candidates": result.rejected_candidates,
        "teacher_as_student_count": result.teacher_as_student_count,
        "printed_as_student_count": result.printed_as_student_count,
        "teacher_samples_with_student_regions": (
            result.teacher_samples_with_student_regions
        ),
        "student_regions_on_teacher_samples": (
            result.student_regions_on_teacher_samples
        ),
        "v1_predicted_regions": result.v1_predicted_region_count,
        "filtered_predicted_regions": result.filtered_predicted_region_count,
        "iou_50_f1": localization.at_iou_50.f1,
        "iou_25_f1": localization.at_iou_25.f1,
        "correctly_empty": localization.correctly_predicted_empty,
        "verified_empty": localization.verified_empty_samples,
    }


def _classes(rows: tuple[EvidenceClassMetrics, ...]) -> dict[str, Any]:
    return {
        row.evidence_type.value: {
            "support": row.support,
            "predicted": row.predicted,
            "precision": row.precision,
            "recall": row.recall,
            "f1": row.f1,
        }
        for row in rows
    }


def _render_overlays(
    output: Path,
    runs: tuple[
        tuple[
            FrozenEvidenceDataset,
            SafetyFilterBenchmarkResult,
            dict[str, SafeTestExtraction],
            dict[str, TestEvidence],
        ],
        ...,
    ],
) -> int:
    requirements = (
        ("student", lambda a, p: a.evidence_type is EvidenceType.STUDENT_CANDIDATE),
        ("teacher", lambda a, p: a.evidence_type is EvidenceType.TEACHER_CANDIDATE),
        ("printed", lambda a, p: a.evidence_type is EvidenceType.PRINTED),
        (
            "verified-empty",
            lambda a, p: a.answer_status is AnswerAnnotationStatus.VERIFIED_EMPTY,
        ),
        ("review", lambda a, p: p.status.value == "review_required"),
    )
    used: set[str] = set()
    count = 0
    for label, predicate in requirements:
        found = _find_sample(runs, used, predicate)
        if found is None:
            continue
        dataset_id, sample, annotation, extraction, prediction = found
        used.add(sample.sample_id)
        sample_image = _private_sample_root(dataset_id) / f"{sample.sample_id}.png"
        render_evaluation_overlay(
            sample,
            annotation,
            prediction,
            sample_image,
            output / f"{label}-{sample.sample_id}.png",
        )
        count += 1
    return count


def _find_sample(
    runs: tuple[
        tuple[
            FrozenEvidenceDataset,
            SafetyFilterBenchmarkResult,
            dict[str, SafeTestExtraction],
            dict[str, TestEvidence],
        ],
        ...,
    ],
    used: set[str],
    predicate: Any,
) -> tuple[str, Sample, EvidenceAnnotation, SafeTestExtraction, TestEvidence] | None:
    for dataset, _, extractions, predictions in runs:
        annotations = {item.sample_id: item for item in dataset.annotations.annotations}
        samples = cast(tuple[Sample, ...], dataset.manifest.samples)
        for sample in samples:
            annotation = annotations[sample.sample_id]
            extraction = extractions[sample.sample_id]
            if sample.sample_id not in used and predicate(annotation, extraction):
                return (
                    dataset.dataset_id,
                    sample,
                    annotation,
                    extraction,
                    predictions[sample.sample_id],
                )
    return None


def _private_sample_root(dataset_id: str) -> Path:
    roots = {
        "phase_4c_4r": Path("data/evaluation/evidence/samples"),
        "evidence_v2": Path("data/evaluation/evidence_v2/samples"),
        "evidence_teacher_v1": Path("data/evaluation/evidence_teacher_v1/samples"),
    }
    return roots[dataset_id].resolve(strict=True)


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise SystemExit("This command accepts no arguments")
    print(json.dumps(run_experiment(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

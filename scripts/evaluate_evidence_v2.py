"""Run controlled private Phase 4C.6 experiments across all frozen benchmarks."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from app.core.exceptions import EvidenceSeparationError
from app.domain.models.paper import PaperPage
from app.evaluation.evidence_benchmark.annotations import (
    AnswerAnnotationStatus,
    EvidenceAnnotation,
    EvidenceAnnotationRepository,
    annotation_fingerprint,
    validate_annotation_dataset,
)
from app.evaluation.evidence_benchmark.manifest import load_evidence_manifest
from app.evaluation.evidence_benchmark.metrics import EvidenceClassMetrics
from app.evaluation.evidence_benchmark.models import (
    EvidenceBenchmarkManifest,
    EvidenceBenchmarkSample,
)
from app.evaluation.evidence_expansion.freezing import (
    verify_frozen_expansion_annotations,
)
from app.evaluation.evidence_expansion.manifest import (
    load_evidence_expansion_manifest,
)
from app.evaluation.evidence_expansion.models import (
    EvidenceExpansionManifest,
    EvidenceExpansionSample,
)
from app.evaluation.evidence_redesign.models import EvidenceV2Experiment
from app.evaluation.evidence_redesign.runner import (
    FrozenEvidenceDataset,
    VariantRun,
    combined_classification,
    controlled_variants,
    run_variant,
    select_variant,
)
from app.evaluation.teacher_evidence.freezing import (
    verify_frozen_teacher_annotations,
)
from app.evaluation.teacher_evidence.manifest import load_teacher_evidence_manifest
from app.evaluation.teacher_evidence.models import (
    TeacherEvidenceManifest,
    TeacherEvidenceSample,
)
from app.evidence.models import EvidenceType, TestEvidence
from app.evidence.overlay_v2 import render_evidence_overlay_v2
from app.evidence.v2_models import EvidenceSeparatorV2Config, TestEvidenceV2
from scripts.prepare_teacher_evidence import _integrity_gate

PHASE_4C_4R = "41d2364cbc0ac56269c30ef41473ccb67e9c08d7109e748f119f274f0671ab35"
EVIDENCE_V2 = "b28eb7ce4daa69bdaa89687cc905366e92d4ed351205c706a77bb16ffea2614b"
TEACHER_V1 = "405f8c0de7e73cbce13859457d2be28c3d615dc3d0ee2dd734a348b29ebbb7d9"
EXPERIMENT_ROOT = Path("data/evaluation/evidence_v2_experiments/phase4c6")

Manifest = (
    EvidenceBenchmarkManifest | EvidenceExpansionManifest | TeacherEvidenceManifest
)
Sample = EvidenceBenchmarkSample | EvidenceExpansionSample | TeacherEvidenceSample


def run_controlled_experiments(
    output_root: Path = EXPERIMENT_ROOT,
) -> dict[str, Any]:
    """Evaluate four predetermined variants and persist non-overwriting records."""

    integrity_before = _verify_all_integrity()
    datasets = _load_datasets()
    configs = controlled_variants()
    all_runs: dict[str, tuple[VariantRun, ...]] = {}
    for config in configs:
        all_runs[config.variant] = tuple(
            run_variant(dataset, config) for dataset in datasets
        )
    selected = select_variant(
        tuple(
            (config, tuple(run.result for run in all_runs[config.variant]))
            for config in configs
        )
    )
    diagnostic_variant = selected or "v2-b"
    created = datetime.now(UTC)
    bundle = output_root.resolve() / created.strftime("%Y%m%dT%H%M%S.%fZ")
    bundle.mkdir(parents=True, exist_ok=False)
    code_version = _code_version()
    fingerprints = {dataset.dataset_id: dataset.fingerprint for dataset in datasets}
    for config in configs:
        runs = all_runs[config.variant]
        experiment = EvidenceV2Experiment(
            configuration=config,
            configuration_hash=config.fingerprint(),
            code_version=code_version,
            benchmark_fingerprints=fingerprints,
            created_at_utc=created,
            selected=selected is not None and config.variant == selected,
            benchmark_results=tuple(run.result for run in runs),
            combined_classification=combined_classification(
                tuple(run.result for run in runs)
            ),
        )
        destination = bundle / f"{config.variant}-{config.fingerprint()[:12]}"
        destination.mkdir(exist_ok=False)
        _write_new_json(
            destination / "experiment.json", experiment.model_dump(mode="json")
        )
        for dataset, run in zip(datasets, runs, strict=True):
            _write_new_json(
                destination / f"{dataset.dataset_id}-predictions.json",
                {
                    "schema_version": "1.0",
                    "configuration_hash": config.fingerprint(),
                    "annotation_fingerprint": dataset.fingerprint,
                    "samples": [
                        {
                            "sample_id": sample_id,
                            "evidence": evidence.model_dump(mode="json"),
                        }
                        for sample_id, evidence in sorted(run.predictions.items())
                    ],
                },
            )
    overlay_count = _selected_overlays(
        bundle / "selected-overlays",
        diagnostic_variant,
        datasets,
        all_runs[diagnostic_variant],
    )
    integrity_after = _verify_all_integrity()
    if integrity_before != integrity_after:
        raise EvidenceSeparationError("Frozen integrity changed during V2 experiments")
    return {
        "bundle": str(bundle.relative_to(Path.cwd().resolve())),
        "selected_variant": selected,
        "selected_configuration_hash": (
            next(
                config.fingerprint() for config in configs if config.variant == selected
            )
            if selected is not None
            else None
        ),
        "diagnostic_variant": diagnostic_variant,
        "visual_overlays": overlay_count,
        "variants": {
            config.variant: _safe_variant_summary(
                config,
                all_runs[config.variant],
                selected=selected is not None and config.variant == selected,
            )
            for config in configs
        },
        "integrity": integrity_after,
    }


def _load_datasets() -> tuple[FrozenEvidenceDataset, ...]:
    old_root = Path("data/evaluation/evidence").resolve(strict=True)
    old_manifest = load_evidence_manifest(old_root / "benchmark_manifest.json")
    old_store = EvidenceAnnotationRepository(
        old_manifest,
        old_root / "frozen" / f"annotations_{PHASE_4C_4R}.json",
        private_root=old_root,
    ).load()
    validate_annotation_dataset(old_manifest, old_store, old_root / "samples")

    v2_root = Path("data/evaluation/evidence_v2").resolve(strict=True)
    v2_manifest = load_evidence_expansion_manifest(v2_root / "benchmark_manifest.json")
    v2_store, _ = verify_frozen_expansion_annotations(
        v2_manifest,
        v2_root / "frozen" / f"annotations_{EVIDENCE_V2}.json",
        v2_root / "frozen" / f"provenance_{EVIDENCE_V2}.json",
        samples_root=v2_root / "samples",
        private_root=v2_root,
    )

    teacher_root = Path("data/evaluation/evidence_teacher_v1").resolve(strict=True)
    teacher_manifest = load_teacher_evidence_manifest(
        teacher_root / "benchmark_manifest.json"
    )
    teacher_store, _ = verify_frozen_teacher_annotations(
        teacher_manifest,
        teacher_root / "frozen" / f"annotations_{TEACHER_V1}.json",
        teacher_root / "frozen" / f"provenance_{TEACHER_V1}.json",
        samples_root=teacher_root / "samples",
        private_root=teacher_root,
    )
    datasets = (
        FrozenEvidenceDataset(
            "phase_4c_4r",
            PHASE_4C_4R,
            old_manifest,
            old_store,
            _load_predictions(
                old_root / "results/phase4c4r/candidate_predictions.json"
            ),
        ),
        FrozenEvidenceDataset(
            "evidence_v2",
            EVIDENCE_V2,
            v2_manifest,
            v2_store,
            _load_predictions(v2_root / "results/phase4c5b/candidate_predictions.json"),
        ),
        FrozenEvidenceDataset(
            "evidence_teacher_v1",
            TEACHER_V1,
            teacher_manifest,
            teacher_store,
            _load_predictions(
                teacher_root / "results/phase4c5d/candidate_predictions.json"
            ),
        ),
    )
    if any(
        annotation_fingerprint(item.annotations) != item.fingerprint
        for item in datasets
    ):
        raise EvidenceSeparationError("V2 experiment annotation integrity failed")
    return datasets


def _load_predictions(path: Path) -> dict[str, TestEvidence]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            str(item["sample_id"]): TestEvidence.model_validate(item["evidence"])
            for item in payload["samples"]
        }
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise EvidenceSeparationError("Frozen V1 predictions are invalid") from error


def _selected_overlays(
    output: Path,
    diagnostic_variant: str,
    datasets: tuple[FrozenEvidenceDataset, ...],
    runs: tuple[VariantRun, ...],
) -> int:
    requirements = (
        ("printed-only", lambda a, c: a.evidence_type is EvidenceType.PRINTED),
        (
            "short-student",
            lambda a, c: (
                a.evidence_type is EvidenceType.STUDENT_CANDIDATE
                and "short_answer" in c
            ),
        ),
        (
            "paragraph-student",
            lambda a, c: (
                a.evidence_type is EvidenceType.STUDENT_CANDIDATE and "paragraph" in c
            ),
        ),
        (
            "teacher-tick-cross",
            lambda a, c: (
                a.evidence_type is EvidenceType.TEACHER_CANDIDATE
                and "tick_cross_geometry" in c
            ),
        ),
        (
            "teacher-score",
            lambda a, c: (
                a.evidence_type is EvidenceType.TEACHER_CANDIDATE
                and "score_like_geometry" in c
            ),
        ),
        ("correction", lambda a, c: "correction_stroke_risk" in c),
        ("mixed", lambda a, c: "mixed" in " ".join(c)),
        (
            "verified-empty",
            lambda a, c: a.answer_status is AnswerAnnotationStatus.VERIFIED_EMPTY,
        ),
        ("unknown", lambda a, c: a.evidence_type is EvidenceType.UNKNOWN),
        ("difficult", lambda a, c: "difficult" in " ".join(c)),
    )
    selected_ids: set[str] = set()
    written = 0
    for label, predicate in requirements:
        found = _find_overlay_candidate(datasets, runs, selected_ids, predicate)
        if found is None:
            continue
        sample, annotation, prediction, base = found
        selected_ids.add(sample.sample_id)
        page = _page_for_overlay(sample, base)
        render_evidence_overlay_v2(
            page,
            prediction,
            output / f"{label}-{sample.sample_id}.png",
            crop_bbox=sample.region,
        )
        written += 1
    _write_new_json(
        output / "visual-qa-manifest.json",
        {
            "schema_version": "1.0",
            "diagnostic_variant": diagnostic_variant,
            "selected_for_downstream": False,
            "requirements": [item[0] for item in requirements],
            "overlay_count": written,
        },
    )
    return written


def _find_overlay_candidate(
    datasets: tuple[FrozenEvidenceDataset, ...],
    runs: tuple[VariantRun, ...],
    used: set[str],
    predicate: Any,
) -> tuple[Sample, EvidenceAnnotation, TestEvidenceV2, TestEvidence] | None:
    for dataset, run in zip(datasets, runs, strict=True):
        annotations = {item.sample_id: item for item in dataset.annotations.annotations}
        samples = cast(tuple[Sample, ...], dataset.manifest.samples)
        for sample in samples:
            annotation = annotations[sample.sample_id]
            categories = _categories(sample)
            if sample.sample_id not in used and predicate(annotation, categories):
                return (
                    sample,
                    annotation,
                    run.predictions[sample.sample_id],
                    dataset.v1_predictions[sample.sample_id],
                )
    return None


def _categories(sample: Sample) -> tuple[str, ...]:
    values = tuple(str(item.value) for item in sample.categories)
    if isinstance(sample, (EvidenceExpansionSample, TeacherEvidenceSample)):
        return (sample.discovery_category.value, *values)
    return values


def _page_for_overlay(sample: Sample, base: TestEvidence) -> PaperPage:
    return PaperPage(
        paper_id=base.paper_id,
        page_number=base.page_number,
        image_path=sample.source_image_path.resolve(strict=True),
        width=sample.page_width,
        height=sample.page_height,
    )


def _safe_variant_summary(
    config: EvidenceSeparatorV2Config,
    runs: tuple[VariantRun, ...],
    *,
    selected: bool,
) -> dict[str, Any]:
    combined = combined_classification(tuple(run.result for run in runs))
    return {
        "selected": selected,
        "configuration_hash": config.fingerprint(),
        "combined": _classification_summary(combined.per_class, combined),
        "benchmarks": {
            run.result.dataset_id: _safe_benchmark_summary(run) for run in runs
        },
    }


def _safe_benchmark_summary(run: VariantRun) -> dict[str, Any]:
    classification = run.result.evaluation.classification
    localization = run.result.evaluation.answer_localization
    return {
        "accuracy": classification.overall_accuracy,
        "macro_f1": classification.macro_f1,
        "unknown_rate": classification.unknown_prediction_rate,
        "classes": {
            item.evidence_type.value: {
                "precision": item.precision,
                "recall": item.recall,
                "f1": item.f1,
            }
            for item in classification.per_class
        },
        "teacher_as_student_count": run.result.teacher_as_student_count,
        "teacher_as_student_rate": run.result.teacher_as_student_rate,
        "teacher_samples_with_student_regions": (
            run.result.teacher_samples_with_student_regions
        ),
        "student_regions_on_teacher_samples": (
            run.result.student_regions_on_teacher_samples
        ),
        "iou_50_f1": localization.at_iou_50.f1,
        "iou_25_f1": localization.at_iou_25.f1,
        "correctly_empty": localization.correctly_predicted_empty,
        "verified_empty": localization.verified_empty_samples,
        "mean_region_area_ratio": run.result.mean_predicted_region_area_ratio,
    }


def _classification_summary(
    rows: tuple[EvidenceClassMetrics, ...], metrics: Any
) -> dict[str, Any]:
    return {
        "accuracy": metrics.overall_accuracy,
        "macro_f1": metrics.macro_f1,
        "weighted_f1": metrics.weighted_f1,
        "unknown_rate": metrics.unknown_prediction_rate,
        "classes": {
            row.evidence_type.value: {
                "support": row.support,
                "precision": row.precision,
                "recall": row.recall,
                "f1": row.f1,
            }
            for row in rows
        },
    }


def _verify_all_integrity() -> dict[str, Any]:
    datasets = _load_datasets()
    historical = _integrity_gate(Path("data/evaluation/evidence_v2"))
    expected = {
        "phase_4c_4r": PHASE_4C_4R,
        "evidence_v2": EVIDENCE_V2,
        "evidence_teacher_v1": TEACHER_V1,
    }
    observed = {item.dataset_id: item.fingerprint for item in datasets}
    if observed != expected:
        raise EvidenceSeparationError("Frozen V2 benchmark integrity failed")
    return {**historical, **observed}


def _code_version() -> str:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], check=False).returncode != 0
    except (OSError, subprocess.SubprocessError) as error:
        raise EvidenceSeparationError(
            "Git code version could not be recorded"
        ) from error
    return f"{revision}+dirty" if dirty else revision


def _write_new_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise EvidenceSeparationError(
            "V2 experiment artifact already exists"
        ) from error


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise SystemExit("This command accepts no arguments")
    print(json.dumps(run_controlled_experiments(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

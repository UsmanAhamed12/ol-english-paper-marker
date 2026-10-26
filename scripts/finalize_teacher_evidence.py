"""Freeze and evaluate the private Phase 4C.5D teacher-v1 baseline."""

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
    validate_annotation_dataset,
)
from app.evaluation.evidence_benchmark.evaluation import (
    EvidenceBenchmarkEvaluation,
    SampleEvidenceEvaluation,
    evaluate_evidence_benchmark,
)
from app.evaluation.evidence_benchmark.evaluation_overlay import (
    render_evaluation_overlay,
)
from app.evaluation.evidence_benchmark.metrics import ClassificationMetrics
from app.evaluation.teacher_evidence.baseline import run_current_teacher_baseline
from app.evaluation.teacher_evidence.freezing import (
    freeze_teacher_annotations,
    verify_frozen_teacher_annotations,
)
from app.evaluation.teacher_evidence.manifest import load_teacher_evidence_manifest
from app.evidence.models import EvidenceType, TestEvidence
from scripts.prepare_teacher_evidence import _integrity_gate

DEFAULT_ROOT = Path("data/evaluation/evidence_teacher_v1")
EVIDENCE_V2_ROOT = Path("data/evaluation/evidence_v2")
PHASE_4C_4R_RESULT = Path(
    "data/evaluation/evidence/results/phase4c4r/baseline_evaluation.json"
)
EVIDENCE_V2_RESULT = Path(
    "data/evaluation/evidence_v2/results/phase4c5b/baseline_evaluation_v2.json"
)


def finalize_teacher_baseline(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Validate, freeze, recover, and measure unchanged detector behavior."""

    integrity = _integrity_gate(EVIDENCE_V2_ROOT)
    resolved = root.resolve(strict=True)
    manifest = load_teacher_evidence_manifest(resolved / "benchmark_manifest.json")
    repository = EvidenceAnnotationRepository(
        manifest, resolved / "annotations.json", private_root=resolved
    )
    live = repository.load()
    samples_root = resolved / "samples"
    validate_annotation_dataset(manifest, live, samples_root)
    live_fingerprint = annotation_fingerprint(live)
    snapshot_path, provenance_path, provenance = freeze_teacher_annotations(
        manifest,
        live,
        samples_root=samples_root,
        frozen_root=resolved / "frozen",
        private_root=resolved,
        phase_4c_4r_fingerprint=integrity["phase_4c_4r_fingerprint"],
        evidence_v2_fingerprint=integrity["evidence_v2_fingerprint"],
        ocr_benchmark_fingerprint=integrity["ocr_benchmark_fingerprint"],
        corpus_sha256=integrity["corpus_sha256"],
        canonical_snapshot_sha256=integrity["canonical_snapshot_sha256"],
    )
    frozen, recovered_provenance = verify_frozen_teacher_annotations(
        manifest,
        snapshot_path,
        provenance_path,
        samples_root=samples_root,
        private_root=resolved,
    )
    if (
        annotation_fingerprint(frozen) != live_fingerprint
        or recovered_provenance.annotation_fingerprint != live_fingerprint
    ):
        raise EvidenceSeparationError(
            "Live and frozen teacher annotation fingerprints do not match"
        )

    predictions = run_current_teacher_baseline(manifest)
    if annotation_fingerprint(repository.load()) != live_fingerprint:
        raise EvidenceSeparationError(
            "Human teacher annotations changed during evaluation"
        )
    report = evaluate_evidence_benchmark(manifest, frozen, predictions)
    if report.annotation_fingerprint != live_fingerprint:
        raise EvidenceSeparationError(
            "Teacher evaluation does not match frozen annotations"
        )

    result_root = resolved / "results" / "phase4c5d"
    _write_new_json(
        result_root / "candidate_predictions.json",
        {
            "schema_version": "1.0",
            "baseline_version": "phase_4c_5d",
            "samples": [
                {
                    "sample_id": sample.sample_id,
                    "evidence": predictions[sample.sample_id].model_dump(mode="json"),
                }
                for sample in manifest.samples
            ],
        },
    )
    annotation_by_id = {item.sample_id: item for item in frozen.annotations}
    overlay_root = resolved / "evaluation-overlays" / "phase4c5d"
    for sample in manifest.samples:
        render_evaluation_overlay(
            sample,
            annotation_by_id[sample.sample_id],
            predictions[sample.sample_id],
            samples_root / f"{sample.sample_id}.png",
            overlay_root / f"{sample.sample_id}.png",
        )

    safe = _safe_summary(report, provenance.model_dump(mode="json"))
    safe["comparisons"] = {
        "phase_4c_4r": _historical_comparison(PHASE_4C_4R_RESULT),
        "evidence_v2": _historical_comparison(EVIDENCE_V2_RESULT),
    }
    _write_new_json(
        result_root / "baseline_evaluation_teacher_v1.json",
        {
            "schema_version": "1.0",
            "baseline_version": "phase_4c_5d",
            "annotation_provenance": provenance.model_dump(mode="json"),
            "evaluation": report.model_dump(mode="json"),
            "safe_summary": safe,
        },
    )
    final_integrity = _integrity_gate(EVIDENCE_V2_ROOT)
    if final_integrity != integrity:
        raise EvidenceSeparationError(
            "Historical integrity changed during teacher evaluation"
        )
    if annotation_fingerprint(repository.load()) != live_fingerprint:
        raise EvidenceSeparationError(
            "Live teacher annotations changed after evaluation"
        )
    safe.update(
        {
            "snapshot_verified": True,
            "recovered_annotation_count": len(frozen.annotations),
            "recovered_answer_box_count": sum(
                len(item.answer_regions) for item in frozen.annotations
            ),
            "recovered_verified_empty_count": sum(
                not item.answer_regions for item in frozen.annotations
            ),
            "private_results_written": True,
            "private_overlays_written": len(manifest.samples),
            "integrity": final_integrity,
        }
    )
    return safe


def resume_teacher_baseline(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Complete reporting after a verified post-snapshot persistence failure."""

    integrity = _integrity_gate(EVIDENCE_V2_ROOT)
    resolved = root.resolve(strict=True)
    manifest = load_teacher_evidence_manifest(resolved / "benchmark_manifest.json")
    repository = EvidenceAnnotationRepository(
        manifest, resolved / "annotations.json", private_root=resolved
    )
    live = repository.load()
    validate_annotation_dataset(manifest, live, resolved / "samples")
    fingerprint = annotation_fingerprint(live)
    frozen, provenance = verify_frozen_teacher_annotations(
        manifest,
        resolved / "frozen" / f"annotations_{fingerprint}.json",
        resolved / "frozen" / f"provenance_{fingerprint}.json",
        samples_root=resolved / "samples",
        private_root=resolved,
    )
    predictions = _load_predictions(
        resolved / "results" / "phase4c5d" / "candidate_predictions.json"
    )
    report = evaluate_evidence_benchmark(manifest, frozen, predictions)
    if report.annotation_fingerprint != fingerprint:
        raise EvidenceSeparationError(
            "Resumed teacher evaluation does not match frozen annotations"
        )
    safe = _safe_summary(report, provenance.model_dump(mode="json"))
    safe["comparisons"] = {
        "phase_4c_4r": _historical_comparison(PHASE_4C_4R_RESULT),
        "evidence_v2": _historical_comparison(EVIDENCE_V2_RESULT),
    }
    _write_new_json(
        resolved / "results" / "phase4c5d" / "baseline_evaluation_teacher_v1.json",
        {
            "schema_version": "1.0",
            "baseline_version": "phase_4c_5d",
            "annotation_provenance": provenance.model_dump(mode="json"),
            "evaluation": report.model_dump(mode="json"),
            "safe_summary": safe,
        },
    )
    final_integrity = _integrity_gate(EVIDENCE_V2_ROOT)
    if (
        final_integrity != integrity
        or annotation_fingerprint(repository.load()) != fingerprint
    ):
        raise EvidenceSeparationError("Integrity changed during resumed evaluation")
    safe.update(
        {
            "snapshot_verified": True,
            "recovered_annotation_count": len(frozen.annotations),
            "recovered_answer_box_count": sum(
                len(item.answer_regions) for item in frozen.annotations
            ),
            "recovered_verified_empty_count": sum(
                not item.answer_regions for item in frozen.annotations
            ),
            "private_results_written": True,
            "private_overlays_written": len(manifest.samples),
            "integrity": final_integrity,
        }
    )
    return safe


def _safe_summary(
    report: EvidenceBenchmarkEvaluation, provenance: dict[str, Any]
) -> dict[str, Any]:
    classes = {row.evidence_type: row for row in report.classification.per_class}
    student = classes[EvidenceType.STUDENT_CANDIDATE]
    teacher = classes[EvidenceType.TEACHER_CANDIDATE]
    matrix = report.classification.confusion_matrix
    teacher_row = matrix[EvidenceType.TEACHER_CANDIDATE.value]
    teacher_support = teacher.support
    localization = report.answer_localization
    empty_f1 = _binary_f1(
        localization.correctly_predicted_empty,
        localization.predicted_empty_samples - localization.correctly_predicted_empty,
        localization.verified_empty_samples - localization.correctly_predicted_empty,
    )
    return {
        "sample_count": report.classification.sample_count,
        "annotation_fingerprint": report.annotation_fingerprint,
        "class_distribution": provenance["class_distribution"],
        "samples_with_answer_regions": provenance["samples_with_answer_regions"],
        "verified_empty_count": provenance["verified_empty_count"],
        "human_answer_box_count": provenance["human_answer_box_count"],
        "classification": report.classification.model_dump(mode="json"),
        "student_false_positives": student.false_positives,
        "student_false_negatives": student.false_negatives,
        "teacher_safety": {
            "teacher_support": teacher_support,
            "teacher_predicted_support": teacher.predicted,
            "teacher_precision": teacher.precision,
            "teacher_recall": teacher.recall,
            "teacher_f1": teacher.f1,
            "teacher_as_student_count": teacher_row[
                EvidenceType.STUDENT_CANDIDATE.value
            ],
            "teacher_as_student_rate": (
                teacher_row[EvidenceType.STUDENT_CANDIDATE.value] / teacher_support
                if teacher_support
                else None
            ),
            "teacher_to_printed": teacher_row[EvidenceType.PRINTED.value],
            "teacher_to_unknown": teacher_row[EvidenceType.UNKNOWN.value],
            "teacher_to_teacher": teacher_row[EvidenceType.TEACHER_CANDIDATE.value],
            "student_to_teacher": matrix[EvidenceType.STUDENT_CANDIDATE.value][
                EvidenceType.TEACHER_CANDIDATE.value
            ],
            "printed_to_teacher": matrix[EvidenceType.PRINTED.value][
                EvidenceType.TEACHER_CANDIDATE.value
            ],
            "unknown_to_teacher": matrix[EvidenceType.UNKNOWN.value][
                EvidenceType.TEACHER_CANDIDATE.value
            ],
            **_teacher_localization_contamination(report.samples),
        },
        "answer_localization": localization.model_dump(mode="json"),
        "empty_f1": empty_f1,
        "stratified": _stratified_teacher_analysis(report.samples),
    }


def _teacher_localization_contamination(
    samples: tuple[SampleEvidenceEvaluation, ...],
) -> dict[str, int | float | None]:
    teacher_samples = tuple(
        item for item in samples if item.human_class is EvidenceType.TEACHER_CANDIDATE
    )
    boxes = sum(item.predicted_answer_boxes for item in teacher_samples)
    return {
        "teacher_samples_predicted_answer_empty": sum(
            item.predicted_answer_boxes == 0 for item in teacher_samples
        ),
        "teacher_samples_with_predicted_answer_regions": sum(
            item.predicted_answer_boxes > 0 for item in teacher_samples
        ),
        "predicted_student_answer_boxes_on_teacher_samples": boxes,
        "mean_predicted_student_boxes_per_teacher_sample": (
            boxes / len(teacher_samples) if teacher_samples else None
        ),
    }


def _stratified_teacher_analysis(
    samples: tuple[SampleEvidenceEvaluation, ...],
) -> dict[str, dict[str, int | float | None]]:
    categories = sorted(
        {category for sample in samples for category in sample.categories}
    )
    result: dict[str, dict[str, int | float | None]] = {}
    for category in categories:
        group = tuple(sample for sample in samples if category in sample.categories)
        teacher = tuple(
            sample
            for sample in group
            if sample.human_class is EvidenceType.TEACHER_CANDIDATE
        )
        teacher_as_student = sum(
            sample.predicted_class is EvidenceType.STUDENT_CANDIDATE
            for sample in teacher
        )
        result[category] = {
            "sample_count": len(group),
            "classification_correct": sum(sample.class_correct for sample in group),
            "human_teacher_support": len(teacher),
            "teacher_as_student_count": teacher_as_student,
            "teacher_as_student_rate": (
                teacher_as_student / len(teacher) if teacher else None
            ),
            "teacher_samples_with_predicted_answer_regions": sum(
                sample.predicted_answer_boxes > 0 for sample in teacher
            ),
            "predicted_answer_boxes_on_teacher_samples": sum(
                sample.predicted_answer_boxes for sample in teacher
            ),
        }
    return result


def _historical_comparison(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        evaluation = payload["evaluation"]
        classification = ClassificationMetrics.model_validate(
            evaluation["classification"]
        )
        localization = evaluation["answer_localization"]
        provenance = payload["annotation_provenance"]
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise EvidenceSeparationError(
            "Historical evidence evaluation is invalid"
        ) from error
    classes = {row.evidence_type: row for row in classification.per_class}
    teacher = classes[EvidenceType.TEACHER_CANDIDATE]
    teacher_as_student = classification.confusion_matrix[
        EvidenceType.TEACHER_CANDIDATE.value
    ][EvidenceType.STUDENT_CANDIDATE.value]
    return {
        "sample_count": classification.sample_count,
        "class_distribution": provenance["class_distribution"],
        "overall_accuracy": classification.overall_accuracy,
        "macro_f1": classification.macro_f1,
        "student_f1": classes[EvidenceType.STUDENT_CANDIDATE].f1,
        "teacher_f1": teacher.f1,
        "teacher_as_student_rate": (
            teacher_as_student / teacher.support if teacher.support else None
        ),
        "iou_50_f1": localization["at_iou_50"]["f1"],
        "iou_25_f1": localization["at_iou_25"]["f1"],
        "verified_empty": localization["verified_empty_samples"],
        "correctly_predicted_empty": localization["correctly_predicted_empty"],
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
            "Private teacher predictions are invalid"
        ) from error
    if len(predictions) != len(records):
        raise EvidenceSeparationError("Private teacher predictions contain duplicates")
    return predictions


def _binary_f1(
    true_positive: int, false_positive: int, false_negative: int
) -> float | None:
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else None


def _write_new_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise EvidenceSeparationError("Phase 4C.5D result already exists")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.link(temporary, path)
    except FileExistsError as error:
        raise EvidenceSeparationError("Phase 4C.5D result already exists") from error
    except OSError as error:
        raise EvidenceSeparationError(
            "Private Phase 4C.5D result could not be written"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError("Phase 4C.5D finalization accepts no arguments")
    print(json.dumps(finalize_teacher_baseline(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Freeze and evaluate the private Phase 4C.5B evidence-v2 baseline."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
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
from app.evaluation.evidence_benchmark.reverification import (
    verify_frozen_annotations,
)
from app.evaluation.evidence_expansion.baseline import run_current_evidence_baseline
from app.evaluation.evidence_expansion.freezing import (
    freeze_expansion_annotations,
    verify_frozen_expansion_annotations,
)
from app.evaluation.evidence_expansion.manifest import (
    load_evidence_expansion_manifest,
)
from app.evaluation.ocr_benchmark.ground_truth import ground_truth_fingerprint
from app.evaluation.ocr_benchmark.manifest import load_manifest as load_ocr_manifest
from app.evidence.models import EvidenceType, TestEvidence

DEFAULT_ROOT = Path("data/evaluation/evidence_v2")
PHASE_4C_4R_FINGERPRINT = (
    "41d2364cbc0ac56269c30ef41473ccb67e9c08d7109e748f119f274f0671ab35"
)
OCR_FINGERPRINT = "33a5dc8e46a1cf0631d46da41a8490c4ec10a18194591144425422c61ff73f9a"
CORPUS_SHA256 = "431eafedebfa41e198952c3d357dd44605f041e7dd6f1bf6c29a3f322188d8d7"
CANONICAL_SHA256 = "93b6fedd873bbfe81ad5c388eece5b73c2c8b2feb8fd9b4db9bd7622a51f4697"


def finalize_expansion_baseline(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Validate, freeze, recover, and measure unchanged detector behavior."""

    integrity = verify_historical_integrity()
    resolved = root.resolve(strict=True)
    manifest = load_evidence_expansion_manifest(resolved / "benchmark_manifest.json")
    repository = EvidenceAnnotationRepository(
        manifest,
        resolved / "annotations.json",
        private_root=resolved,
    )
    live = repository.load()
    samples_root = resolved / "samples"
    validate_annotation_dataset(manifest, live, samples_root)
    live_fingerprint = annotation_fingerprint(live)
    snapshot_path, provenance_path, provenance = freeze_expansion_annotations(
        manifest,
        live,
        samples_root=samples_root,
        frozen_root=resolved / "frozen",
        private_root=resolved,
        phase_4c_4r_fingerprint=integrity["phase_4c_4r_fingerprint"],
        ocr_benchmark_fingerprint=integrity["ocr_benchmark_fingerprint"],
        corpus_sha256=integrity["corpus_sha256"],
        canonical_snapshot_sha256=integrity["canonical_snapshot_sha256"],
    )
    frozen, recovered_provenance = verify_frozen_expansion_annotations(
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
            "Live and frozen evidence-v2 annotation fingerprints do not match"
        )

    predictions = run_current_evidence_baseline(manifest)
    if annotation_fingerprint(repository.load()) != live_fingerprint:
        raise EvidenceSeparationError(
            "Human evidence-v2 annotations changed during evaluation"
        )
    report = evaluate_evidence_benchmark(manifest, frozen, predictions)
    if report.annotation_fingerprint != live_fingerprint:
        raise EvidenceSeparationError(
            "Evidence-v2 evaluation does not match the frozen annotations"
        )
    result_root = resolved / "results" / "phase4c5b"
    _write_new_json(
        result_root / "candidate_predictions.json",
        {
            "schema_version": "2.0",
            "baseline_version": "phase_4c_5b",
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
    overlay_root = resolved / "evaluation-overlays" / "phase4c5b"
    for sample in manifest.samples:
        render_evaluation_overlay(
            sample,
            annotation_by_id[sample.sample_id],
            predictions[sample.sample_id],
            samples_root / f"{sample.sample_id}.png",
            overlay_root / f"{sample.sample_id}.png",
        )
    safe = _safe_summary(report, provenance.model_dump(mode="json"))
    _write_new_json(
        result_root / "baseline_evaluation_v2.json",
        {
            "schema_version": "2.0",
            "baseline_version": "phase_4c_5b",
            "annotation_provenance": provenance.model_dump(mode="json"),
            "evaluation": report.model_dump(mode="json"),
            "safe_summary": safe,
        },
    )
    final_integrity = verify_historical_integrity()
    if final_integrity != integrity:
        raise EvidenceSeparationError(
            "Historical integrity changed during evidence-v2 evaluation"
        )
    safe.update(
        {
            "snapshot_verified": True,
            "recovered_annotation_count": len(frozen.annotations),
            "recovered_answer_box_count": sum(
                len(item.answer_regions) for item in frozen.annotations
            ),
            "private_results_written": True,
            "private_overlays_written": len(manifest.samples),
            "integrity": final_integrity,
        }
    )
    return safe


def verify_historical_integrity() -> dict[str, Any]:
    """Read-only verification of every pre-existing frozen asset."""

    evidence_root = Path("data/evaluation/evidence")
    from app.evaluation.evidence_benchmark.manifest import load_evidence_manifest

    old_manifest = load_evidence_manifest(evidence_root / "benchmark_manifest.json")
    old_fingerprint = verify_frozen_annotations(
        old_manifest,
        evidence_root / "frozen" / f"annotations_{PHASE_4C_4R_FINGERPRINT}.json",
        evidence_root / "frozen" / f"provenance_{PHASE_4C_4R_FINGERPRINT}.json",
        samples_root=evidence_root / "samples",
        private_root=evidence_root,
    )
    ocr_fingerprint = ground_truth_fingerprint(
        load_ocr_manifest(Path("data/evaluation/ocr/benchmark_manifest.json"))
    )
    raw = sorted(Path("data/raw/marked_papers").glob("*.pdf"), key=lambda p: p.name)
    corpus = hashlib.sha256()
    for path in raw:
        corpus.update(path.name.encode())
        corpus.update(path.read_bytes())
    runtime = Path("data/runtime")
    canonical_paths = sorted(
        runtime.glob("*/pages/page_*.png"), key=lambda path: path.as_posix()
    )
    canonical = hashlib.sha256()
    for path in canonical_paths:
        canonical.update(path.relative_to(runtime).as_posix().encode())
        canonical.update(path.read_bytes())
    observed = {
        "phase_4c_4r_fingerprint": old_fingerprint,
        "ocr_benchmark_fingerprint": ocr_fingerprint,
        "raw_pdf_count": len(raw),
        "corpus_sha256": corpus.hexdigest(),
        "canonical_snapshot_sha256": canonical.hexdigest(),
    }
    expected = {
        "phase_4c_4r_fingerprint": PHASE_4C_4R_FINGERPRINT,
        "ocr_benchmark_fingerprint": OCR_FINGERPRINT,
        "raw_pdf_count": 40,
        "corpus_sha256": CORPUS_SHA256,
        "canonical_snapshot_sha256": CANONICAL_SHA256,
    }
    if observed != expected:
        raise EvidenceSeparationError("Historical frozen-asset integrity failed")
    return observed


def persist_existing_expansion_evaluation(
    root: Path = DEFAULT_ROOT,
) -> dict[str, Any]:
    """Re-evaluate saved predictions against the verified snapshot without OCR."""

    verify_historical_integrity()
    resolved = root.resolve(strict=True)
    manifest = load_evidence_expansion_manifest(resolved / "benchmark_manifest.json")
    repository = EvidenceAnnotationRepository(
        manifest,
        resolved / "annotations.json",
        private_root=resolved,
    )
    fingerprint = annotation_fingerprint(repository.load())
    snapshot_path = resolved / "frozen" / f"annotations_{fingerprint}.json"
    provenance_path = resolved / "frozen" / f"provenance_{fingerprint}.json"
    frozen, provenance = verify_frozen_expansion_annotations(
        manifest,
        snapshot_path,
        provenance_path,
        samples_root=resolved / "samples",
        private_root=resolved,
    )
    predictions = _load_predictions(
        resolved / "results" / "phase4c5b" / "candidate_predictions.json"
    )
    report = evaluate_evidence_benchmark(manifest, frozen, predictions)
    safe = _safe_summary(report, provenance.model_dump(mode="json"))
    _write_new_json(
        resolved / "results" / "phase4c5b" / "baseline_evaluation_v2.json",
        {
            "schema_version": "2.0",
            "baseline_version": "phase_4c_5b",
            "annotation_provenance": provenance.model_dump(mode="json"),
            "evaluation": report.model_dump(mode="json"),
            "safe_summary": safe,
        },
    )
    return safe


def _safe_summary(
    report: EvidenceBenchmarkEvaluation, provenance: dict[str, Any]
) -> dict[str, Any]:
    classes = {row.evidence_type: row for row in report.classification.per_class}
    matrix = report.classification.confusion_matrix
    student = classes[EvidenceType.STUDENT_CANDIDATE]
    teacher = classes[EvidenceType.TEACHER_CANDIDATE]
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
        "teacher_to_student_errors": matrix[EvidenceType.TEACHER_CANDIDATE.value][
            EvidenceType.STUDENT_CANDIDATE.value
        ],
        "student_to_teacher_errors": matrix[EvidenceType.STUDENT_CANDIDATE.value][
            EvidenceType.TEACHER_CANDIDATE.value
        ],
        "teacher_metrics_defined": teacher.f1 is not None,
        "answer_localization": report.answer_localization.model_dump(mode="json"),
        "stratified": _stratified(report.samples),
    }


def _stratified(
    samples: tuple[SampleEvidenceEvaluation, ...],
) -> dict[str, dict[str, int]]:
    totals: Counter[str] = Counter()
    class_errors: Counter[str] = Counter()
    answer_misses: Counter[str] = Counter()
    answer_extras: Counter[str] = Counter()
    for sample in samples:
        totals.update(sample.categories)
        if not sample.class_correct:
            class_errors.update(sample.categories)
        if sample.matched_answer_boxes_at_50 < sample.human_answer_boxes:
            answer_misses.update(sample.categories)
        if sample.matched_answer_boxes_at_50 < sample.predicted_answer_boxes:
            answer_extras.update(sample.categories)
    return {
        key: {
            "sample_count": totals[key],
            "classification_errors": class_errors[key],
            "answer_miss_samples_at_iou_50": answer_misses[key],
            "answer_extra_samples_at_iou_50": answer_extras[key],
        }
        for key in sorted(totals)
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
            "Private evidence-v2 predictions are invalid"
        ) from error
    if len(predictions) != len(records):
        raise EvidenceSeparationError(
            "Private evidence-v2 predictions contain duplicates"
        )
    return predictions


def _write_new_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise EvidenceSeparationError("Phase 4C.5B result already exists")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.link(temporary, path)
    except FileExistsError as error:
        raise EvidenceSeparationError("Phase 4C.5B result already exists") from error
    except OSError as error:
        raise EvidenceSeparationError(
            "Private Phase 4C.5B result could not be written"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise ValueError("Phase 4C.5B finalization accepts no arguments")
    print(json.dumps(finalize_expansion_baseline(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

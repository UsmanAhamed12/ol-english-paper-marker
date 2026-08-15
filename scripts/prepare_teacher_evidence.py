"""Prepare the private Phase 4C.5C teacher-risk evidence candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

import cv2

from app.core.config import Settings
from app.core.exceptions import EvidenceSeparationError
from app.domain.models.paper import PaperPage
from app.evaluation.evidence_benchmark.annotations import annotation_fingerprint
from app.evaluation.evidence_expansion.freezing import (
    verify_frozen_expansion_annotations,
)
from app.evaluation.evidence_expansion.manifest import (
    load_evidence_expansion_manifest,
)
from app.evaluation.teacher_evidence.discovery import (
    TeacherEvidenceProposal,
    discover_teacher_evidence_candidates,
    select_teacher_evidence_candidates,
    suppress_teacher_candidate_duplicates,
)
from app.evaluation.teacher_evidence.manifest import load_teacher_evidence_manifest
from app.evaluation.teacher_evidence.models import (
    TeacherEvidenceManifest,
    TeacherEvidenceSample,
)
from app.evaluation.teacher_evidence.preparation import (
    prepare_teacher_artifacts,
    sha256_file,
    write_json_atomic,
)
from app.ocr.normalizer import OCRNormalizer
from app.ocr.providers.tesseract import TesseractOCRProvider
from app.ocr.service import OCRService
from scripts.finalize_evidence_expansion import verify_historical_integrity

DEFAULT_ROOT = Path("data/evaluation/evidence_teacher_v1")
EVIDENCE_V2_ROOT = Path("data/evaluation/evidence_v2")
EVIDENCE_V2_FINGERPRINT = (
    "b28eb7ce4daa69bdaa89687cc905366e92d4ed351205c706a77bb16ffea2614b"
)
TARGET_COUNT = 48
MINIMUM_COUNT = 30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "validate"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    return parser


def prepare_teacher_benchmark(
    *,
    output_root: Path,
    evidence_v2_root: Path = EVIDENCE_V2_ROOT,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Discover teacher-risk candidates without creating human ground truth."""

    integrity = _integrity_gate(evidence_v2_root)
    root = output_root.resolve()
    manifest_path = root / "benchmark_manifest.json"
    if (root / "annotations.json").exists():
        raise EvidenceSeparationError(
            "Teacher candidate preparation refuses to replace human annotations"
        )
    if manifest_path.exists():
        manifest = load_teacher_evidence_manifest(manifest_path)
        written, duplicates = prepare_teacher_artifacts(manifest, root)
        return _safe_summary(manifest, written, duplicates, integrity=integrity)

    settings = settings or Settings()
    provider = TesseractOCRProvider.from_system(
        language=settings.tesseract_language,
        psm=settings.tesseract_psm,
        timeout_seconds=settings.tesseract_timeout_seconds,
    )
    service = OCRService(provider, OCRNormalizer())
    pages, private_sources = _existing_v2_pages(evidence_v2_root)
    proposals: list[TeacherEvidenceProposal] = []
    page_registry: dict[tuple[str, int], PaperPage] = {}
    for alias, page in pages:
        image = cv2.imread(str(page.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise EvidenceSeparationError("Teacher discovery source page is invalid")
        before = sha256_file(page.image_path)
        result = service.process_page(page)
        proposals.extend(
            discover_teacher_evidence_candidates(page, result, image, paper_alias=alias)
        )
        if sha256_file(page.image_path) != before:
            raise EvidenceSeparationError("Teacher discovery changed a canonical page")
        page_registry[(alias, page.page_number)] = page

    suppressed = suppress_teacher_candidate_duplicates(tuple(proposals))
    selected = select_teacher_evidence_candidates(
        suppressed,
        target_count=TARGET_COUNT,
        maximum_per_paper=4,
        maximum_per_page=2,
    )
    if len(selected) < MINIMUM_COUNT:
        raise EvidenceSeparationError(
            "Teacher discovery produced fewer than 30 diverse candidates"
        )
    samples = tuple(
        _sample_from_proposal(index, proposal, page_registry)
        for index, proposal in enumerate(selected, start=1)
    )
    manifest = TeacherEvidenceManifest(samples=samples)
    root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(manifest_path, manifest.model_dump(mode="json"))
    written, duplicate_crops = prepare_teacher_artifacts(manifest, root)
    write_json_atomic(
        root / "candidate_discovery_provenance.json",
        {
            "schema_version": "1.0",
            "dataset_id": manifest.dataset_id,
            "candidate_categories_are_ground_truth": False,
            "human_ground_truth_created": False,
            "discovery_version": manifest.candidate_generation_version,
            "source_papers": private_sources,
            "scanned_page_count": len(pages),
            "raw_candidate_count": len(proposals),
            "after_overlap_suppression": len(suppressed),
            "selected_candidate_count": len(samples),
            "historical_integrity": integrity,
        },
    )
    write_json_atomic(
        root / "labeling_metadata.json",
        {
            "schema_version": "1.0",
            "dataset_id": manifest.dataset_id,
            "status": "pending_human_labeling",
            "expected_sample_ids": [sample.sample_id for sample in samples],
            "human_verified_count": 0,
            "pending_count": len(samples),
            "candidate_categories_are_ground_truth": False,
            "answer_box_policy_version": "student-answer-box-v2",
        },
    )
    return _safe_summary(
        manifest,
        written,
        duplicate_crops,
        integrity=integrity,
        discovered=len(proposals),
        suppressed=len(suppressed),
    )


def validate_teacher_benchmark(
    root: Path = DEFAULT_ROOT, *, evidence_v2_root: Path = EVIDENCE_V2_ROOT
) -> dict[str, Any]:
    """Validate pending private artifacts without evaluating the separator."""

    integrity = _integrity_gate(evidence_v2_root)
    resolved = root.resolve(strict=True)
    manifest = load_teacher_evidence_manifest(resolved / "benchmark_manifest.json")
    for sample in manifest.samples:
        if not (resolved / "samples" / f"{sample.sample_id}.png").is_file():
            raise EvidenceSeparationError("Teacher candidate sample image is missing")
        if sha256_file(sample.source_image_path) != sample.source_image_sha256:
            raise EvidenceSeparationError("Teacher candidate canonical hash changed")
    annotations = resolved / "annotations.json"
    human_verified = 0
    if annotations.exists():
        from app.evaluation.evidence_benchmark.annotations import (
            EvidenceAnnotationRepository,
        )

        human_verified = len(
            EvidenceAnnotationRepository(manifest, annotations, private_root=resolved)
            .load()
            .annotations
        )
    summary = _safe_summary(manifest, len(manifest.samples), 0, integrity=integrity)
    summary.update(
        human_verified_count=human_verified,
        pending_count=len(manifest.samples) - human_verified,
        benchmark_ready=human_verified == len(manifest.samples),
    )
    return summary


def _integrity_gate(root: Path) -> dict[str, Any]:
    integrity = verify_historical_integrity()
    resolved = root.resolve(strict=True)
    manifest = load_evidence_expansion_manifest(resolved / "benchmark_manifest.json")
    snapshot = resolved / "frozen" / f"annotations_{EVIDENCE_V2_FINGERPRINT}.json"
    provenance = resolved / "frozen" / f"provenance_{EVIDENCE_V2_FINGERPRINT}.json"
    recovered, record = verify_frozen_expansion_annotations(
        manifest,
        snapshot,
        provenance,
        samples_root=resolved / "samples",
        private_root=resolved,
    )
    if (
        annotation_fingerprint(recovered) != EVIDENCE_V2_FINGERPRINT
        or record.annotation_fingerprint != EVIDENCE_V2_FINGERPRINT
    ):
        raise EvidenceSeparationError("Evidence-v2 frozen integrity failed")
    return {**integrity, "evidence_v2_fingerprint": EVIDENCE_V2_FINGERPRINT}


def _existing_v2_pages(
    root: Path,
) -> tuple[tuple[tuple[str, PaperPage], ...], list[dict[str, object]]]:
    payload = json.loads(
        (root / "discovery_provenance.json").read_text(encoding="utf-8")
    )
    pages: list[tuple[str, PaperPage]] = []
    private_sources: list[dict[str, object]] = []
    for record in payload["source_papers"]:
        alias = str(record["paper_alias"])
        paper_id = UUID(str(record["rendered_paper_id"]))
        selected_pages = tuple(int(value) for value in record["selected_pages"])
        private_sources.append(dict(record))
        page_root = Path("data/runtime/evidence-v2/canonical") / paper_id.hex / "pages"
        for number in selected_pages:
            path = (page_root / f"page_{number:04d}.png").resolve(strict=True)
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise EvidenceSeparationError("Existing evidence-v2 page is invalid")
            height, width = image.shape[:2]
            pages.append(
                (
                    alias,
                    PaperPage(
                        paper_id=paper_id,
                        page_number=number,
                        image_path=path,
                        width=width,
                        height=height,
                    ),
                )
            )
    return tuple(pages), private_sources


def _sample_from_proposal(
    index: int,
    proposal: TeacherEvidenceProposal,
    pages: dict[tuple[str, int], PaperPage],
) -> TeacherEvidenceSample:
    page = pages[(proposal.paper_alias, proposal.page_number)]
    return TeacherEvidenceSample(
        sample_id=f"evidence_teacher_v1_{index:03d}",
        paper_alias=proposal.paper_alias,
        page_number=proposal.page_number,
        test_number=proposal.test_number,
        source_image_path=page.image_path,
        source_image_sha256=sha256_file(page.image_path),
        page_width=page.width,
        page_height=page.height,
        region=proposal.region,
        candidate_component=proposal.candidate_component,
        discovery_category=proposal.category,
        discovery_signals=proposal.signals,
        features=proposal.features,
        selection_rank=index,
        discovery_reason=proposal.reason,
    )


def _safe_summary(
    manifest: TeacherEvidenceManifest,
    written: int,
    duplicates: int,
    *,
    integrity: dict[str, Any],
    discovered: int | None = None,
    suppressed: int | None = None,
) -> dict[str, Any]:
    categories = Counter(sample.discovery_category.value for sample in manifest.samples)
    signals = Counter(
        signal.value
        for sample in manifest.samples
        for signal in sample.discovery_signals
    )
    result: dict[str, Any] = {
        "dataset_id": manifest.dataset_id,
        "sample_count": len(manifest.samples),
        "source_paper_count": len({sample.paper_alias for sample in manifest.samples}),
        "source_page_count": len(
            {(sample.paper_alias, sample.page_number) for sample in manifest.samples}
        ),
        "paper_alias_counts": dict(
            sorted(Counter(sample.paper_alias for sample in manifest.samples).items())
        ),
        "candidate_discovery_counts": dict(sorted(categories.items())),
        "candidate_signal_counts": dict(sorted(signals.items())),
        "test_context_count": sum(
            sample.test_number is not None for sample in manifest.samples
        ),
        "sample_images": written,
        "duplicate_crops_rejected": duplicates,
        "human_verified_count": 0,
        "pending_count": len(manifest.samples),
        "benchmark_ready": False,
        "candidate_categories_are_ground_truth": False,
        "integrity": integrity,
    }
    if discovered is not None:
        result["raw_candidate_count"] = discovered
    if suppressed is not None:
        result["after_overlap_suppression"] = suppressed
        result["overlap_candidates_rejected"] = (
            discovered - suppressed if discovered else 0
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    payload = (
        prepare_teacher_benchmark(output_root=arguments.output_root)
        if arguments.command == "prepare"
        else validate_teacher_benchmark(arguments.output_root)
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

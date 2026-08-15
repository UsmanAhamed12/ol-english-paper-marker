"""Discover and prepare the private Phase 4C.5A evidence-v2 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import string
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import cv2

from app.core.config import Settings
from app.core.exceptions import EvidenceSeparationError
from app.domain.models.paper import PaperDocument, PaperPage
from app.evaluation.evidence_expansion.discovery import (
    DiscoveredEvidenceCandidate,
    discover_page_candidates,
    select_balanced_candidates,
)
from app.evaluation.evidence_expansion.manifest import (
    load_evidence_expansion_manifest,
)
from app.evaluation.evidence_expansion.models import (
    EvidenceCandidateCategory,
    EvidenceExpansionManifest,
    EvidenceExpansionSample,
)
from app.evaluation.evidence_expansion.preparation import (
    prepare_expansion_artifacts,
    write_json_atomic,
)
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.pdf_renderer import PDFRenderer
from app.ingestion.validators import PDFValidator
from app.ocr.normalizer import OCRNormalizer
from app.ocr.providers.tesseract import TesseractOCRProvider
from app.ocr.service import OCRService

DEFAULT_ROOT = Path("data/evaluation/evidence_v2")
DEFAULT_MANIFEST = DEFAULT_ROOT / "benchmark_manifest.json"
DEFAULT_RAW_ROOT = Path("data/raw/marked_papers")
DEFAULT_RUNTIME_ROOT = Path("data/runtime/evidence-v2/canonical")
DEFAULT_STRUCTURE_MANIFEST = Path("data/evaluation/structure/manifest.json")
TARGET_QUOTAS = {
    EvidenceCandidateCategory.PRINTED: 10,
    EvidenceCandidateCategory.STUDENT: 14,
    EvidenceCandidateCategory.TEACHER: 12,
    EvidenceCandidateCategory.MIXED: 6,
    EvidenceCandidateCategory.BLANK: 6,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "validate"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--paper-count", type=int, default=12)
    return parser


def prepare_expanded_benchmark(
    *,
    output_root: Path,
    raw_root: Path = DEFAULT_RAW_ROOT,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    paper_count: int = 12,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Discover a balanced private candidate pool without assigning labels."""

    if paper_count < 4 or paper_count > 26:
        raise ValueError("Evidence-v2 paper count must be between 4 and 26")
    root = output_root.resolve()
    manifest_path = root / "benchmark_manifest.json"
    annotations_path = root / "annotations.json"
    if annotations_path.exists():
        raise EvidenceSeparationError(
            "Evidence-v2 preparation refuses to replace human annotations"
        )
    if manifest_path.exists():
        manifest = load_evidence_expansion_manifest(manifest_path)
        written, duplicate_count = prepare_expansion_artifacts(manifest, root)
        _normalize_prelabel_provenance(root / "discovery_provenance.json")
        return _safe_summary(manifest, written, duplicate_count)

    settings = settings or Settings()
    sources = _select_sources(
        raw_root.resolve(strict=True),
        paper_count,
        structure_manifest=DEFAULT_STRUCTURE_MANIFEST,
    )
    validator = PDFValidator(
        max_file_size_bytes=settings.max_pdf_size_mb * 1024 * 1024,
        max_pages=settings.max_pdf_pages,
    )
    renderer = PDFRenderer(
        runtime_data_dir=runtime_root,
        render_dpi=settings.pdf_render_dpi,
    )
    provider = TesseractOCRProvider.from_system(
        language=settings.tesseract_language,
        psm=settings.tesseract_psm,
        timeout_seconds=settings.tesseract_timeout_seconds,
    )
    ocr_service = OCRService(provider, OCRNormalizer())
    discovered: list[DiscoveredEvidenceCandidate] = []
    source_mapping: list[dict[str, object]] = []
    page_registry: dict[str, dict[int, PaperPage]] = {}
    page_count = 0
    for alias, source, source_hash in sources:
        document = _load_or_render(
            source, source_hash, validator, renderer, runtime_root
        )
        selected_pages = _selected_pages(document.pages)
        page_registry[alias] = {page.page_number: page for page in document.pages}
        source_mapping.append(
            {
                "paper_alias": alias,
                "source_path": str(source),
                "source_pdf_sha256": source_hash,
                "rendered_paper_id": document.paper_id.hex,
                "selected_pages": [page.page_number for page in selected_pages],
            }
        )
        for page in selected_pages:
            image = cv2.imread(str(page.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise EvidenceSeparationError("Canonical evidence-v2 page is invalid")
            result = ocr_service.process_page(page)
            discovered.extend(
                discover_page_candidates(page, result, image, paper_alias=alias)
            )
            page_count += 1

    selected = select_balanced_candidates(
        tuple(discovered), quotas=TARGET_QUOTAS, maximum_per_paper=5
    )
    required = sum(TARGET_QUOTAS.values())
    distribution = Counter(item.category for item in selected)
    if len(selected) != required or any(
        distribution[category] != target for category, target in TARGET_QUOTAS.items()
    ):
        raise EvidenceSeparationError(
            "Candidate discovery did not satisfy the predetermined balanced pool"
        )
    samples = tuple(
        _sample_from_candidate(index, item, page_registry)
        for index, item in enumerate(selected, start=1)
    )
    manifest = EvidenceExpansionManifest(samples=samples)
    root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(manifest_path, manifest.model_dump(mode="json"))
    written, duplicate_count = prepare_expansion_artifacts(manifest, root)
    provenance = {
        "schema_version": "2.0",
        "human_ground_truth_created": False,
        "candidate_categories_are_ground_truth": False,
        "discovery_version": "evidence-candidate-discovery-v1",
        "tesseract_language": settings.tesseract_language,
        "tesseract_psm": settings.tesseract_psm,
        "render_dpi": settings.pdf_render_dpi,
        "source_papers": source_mapping,
        "scanned_page_count": page_count,
        "discovered_candidate_count": len(discovered),
        "selected_candidate_count": len(selected),
        "target_quotas": {key.value: value for key, value in TARGET_QUOTAS.items()},
        "unselected_after_balancing": len(discovered) - len(selected),
    }
    write_json_atomic(root / "discovery_provenance.json", provenance)
    write_json_atomic(
        root / "labeling_metadata.json",
        {
            "schema_version": "2.0",
            "status": "pending_human_labeling",
            "expected_sample_ids": [sample.sample_id for sample in samples],
            "answer_box_policy_version": "student-answer-box-v2",
            "annotations_path": "annotations.json",
        },
    )
    return _safe_summary(manifest, written, duplicate_count, discovered=len(discovered))


def validate_expanded_benchmark(root: Path) -> dict[str, Any]:
    """Validate private candidates without evaluating or printing source paths."""

    resolved = root.resolve(strict=True)
    manifest = load_evidence_expansion_manifest(resolved / "benchmark_manifest.json")
    missing = [
        sample.sample_id
        for sample in manifest.samples
        if not (resolved / "samples" / f"{sample.sample_id}.png").is_file()
    ]
    if missing:
        raise EvidenceSeparationError(
            f"Evidence-v2 sample image is missing: {missing[0]}"
        )
    annotations = resolved / "annotations.json"
    human_verified = 0
    if annotations.exists():
        from app.evaluation.evidence_benchmark.annotations import (
            EvidenceAnnotationRepository,
        )

        repository = EvidenceAnnotationRepository(
            manifest,
            annotations,
            private_root=resolved,
        )
        human_verified = len(repository.load().annotations)
    summary = _safe_summary(manifest, len(manifest.samples), 0)
    summary.update(
        {
            "human_verified_count": human_verified,
            "pending_count": len(manifest.samples) - human_verified,
            "benchmark_ready": human_verified == len(manifest.samples),
        }
    )
    return summary


def _select_sources(
    raw_root: Path,
    count: int,
    *,
    structure_manifest: Path,
) -> tuple[tuple[str, Path, str], ...]:
    paths = sorted(raw_root.glob("*.pdf"))
    if len(paths) < count:
        raise EvidenceSeparationError("Insufficient private PDFs for evidence-v2")
    selected: list[Path] = []
    if structure_manifest.is_file():
        try:
            private_structure = json.loads(
                structure_manifest.read_text(encoding="utf-8")
            )
            selected = [
                Path(record["source_path"]).resolve(strict=True)
                for record in private_structure["papers"]
                if record.get("paper_alias") in {"paper-a", "paper-b", "paper-c"}
            ]
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise EvidenceSeparationError(
                "Private structure source mapping is invalid"
            ) from error
    remaining = sorted(
        (path for path in paths if path not in selected),
        key=lambda path: (_sha256(path), path.stat().st_size),
    )
    selected.extend(remaining[: count - len(selected)])
    aliases = tuple(f"paper-{string.ascii_lowercase[index]}" for index in range(count))
    return tuple(
        (alias, path.resolve(strict=True), _sha256(path))
        for alias, path in zip(aliases, selected, strict=True)
    )


def _load_or_render(
    source: Path,
    source_hash: str,
    validator: PDFValidator,
    renderer: PDFRenderer,
    runtime_root: Path,
) -> PaperDocument:
    paper_id = uuid5(NAMESPACE_URL, f"ol-english-evidence-v2/{source_hash}")
    loader = PDFLoader(validator, paper_id_factory=lambda: paper_id)
    document = loader.load(source)
    pages_root = runtime_root.resolve() / paper_id.hex / "pages"
    if not pages_root.is_dir():
        return renderer.render(document)
    pages = tuple(
        _page_from_path(paper_id, number, pages_root / f"page_{number:04d}.png")
        for number in range(1, document.page_count + 1)
    )
    return document.model_copy(update={"pages": pages})


def _page_from_path(paper_id: UUID, page_number: int, path: Path) -> PaperPage:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise EvidenceSeparationError("Existing evidence-v2 render is invalid")
    height, width = image.shape[:2]
    return PaperPage(
        paper_id=paper_id,
        page_number=page_number,
        image_path=path.resolve(strict=True),
        width=width,
        height=height,
    )


def _selected_pages(pages: tuple[PaperPage, ...]) -> tuple[PaperPage, ...]:
    count = len(pages)
    indices = sorted(
        {
            max(0, min(count - 1, round((count - 1) * fraction)))
            for fraction in (0.15, 0.40, 0.65, 0.88)
        }
    )
    return tuple(pages[index] for index in indices)


def _sample_from_candidate(
    index: int,
    candidate: DiscoveredEvidenceCandidate,
    page_registry: dict[str, dict[int, PaperPage]],
) -> EvidenceExpansionSample:
    page = page_registry[candidate.paper_alias][candidate.page_number]
    return EvidenceExpansionSample(
        sample_id=f"evidence_v2_{index:03d}",
        paper_alias=candidate.paper_alias,
        page_number=candidate.page_number,
        test_number=candidate.test_number,
        source_image_path=page.image_path,
        source_image_sha256=_sha256(page.image_path),
        page_width=page.width,
        page_height=page.height,
        region=candidate.bbox,
        discovery_category=candidate.category,
        context_tags=candidate.context_tags,
        discovery_reason=candidate.reason,
    )


def _safe_summary(
    manifest: EvidenceExpansionManifest,
    written: int,
    duplicate_count: int,
    *,
    discovered: int | None = None,
) -> dict[str, Any]:
    category_counts = Counter(
        sample.discovery_category.value for sample in manifest.samples
    )
    tags = Counter(
        tag.value for sample in manifest.samples for tag in sample.context_tags
    )
    result: dict[str, Any] = {
        "schema_version": manifest.schema_version,
        "sample_count": len(manifest.samples),
        "source_paper_count": len({sample.paper_alias for sample in manifest.samples}),
        "source_page_count": len(
            {(sample.paper_alias, sample.page_number) for sample in manifest.samples}
        ),
        "paper_alias_counts": dict(
            sorted(Counter(sample.paper_alias for sample in manifest.samples).items())
        ),
        "test_numbered_samples": sum(
            sample.test_number is not None for sample in manifest.samples
        ),
        "detected_test_numbers": sorted(
            {
                sample.test_number
                for sample in manifest.samples
                if sample.test_number is not None
            }
        ),
        "candidate_discovery_counts": dict(sorted(category_counts.items())),
        "context_tag_counts": dict(sorted(tags.items())),
        "sample_images": written,
        "duplicate_crops_rejected": duplicate_count,
        "human_verified_count": 0,
        "pending_count": len(manifest.samples),
        "benchmark_ready": False,
        "candidate_categories_are_ground_truth": False,
    }
    if discovered is not None:
        result["discovered_candidate_count"] = discovered
        result["unselected_after_balancing"] = discovered - len(manifest.samples)
    return result


def _normalize_prelabel_provenance(path: Path) -> None:
    """Correct a pre-labeling statistic name without touching candidates."""

    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise EvidenceSeparationError(
            "Private evidence-v2 discovery provenance is invalid"
        ) from error
    if not isinstance(payload, dict):
        raise EvidenceSeparationError(
            "Private evidence-v2 discovery provenance is invalid"
        )
    old_key = "duplicate_or_overlap_candidates_rejected"
    if old_key in payload and "unselected_after_balancing" not in payload:
        payload["unselected_after_balancing"] = payload.pop(old_key)
        write_json_atomic(path, payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        payload = prepare_expanded_benchmark(
            output_root=args.output_root,
            paper_count=args.paper_count,
        )
    else:
        payload = validate_expanded_benchmark(args.output_root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

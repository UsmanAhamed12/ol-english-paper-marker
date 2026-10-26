"""Run a privacy-safe three-paper structured-answer extraction smoke test."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.answers.extraction import StudentRegionCropper
from app.answers.models import (
    AnswerExtractionStatus,
    AnswerSourceContext,
    StructuredStudentPaper,
)
from app.answers.serialization import write_structured_paper
from app.answers.service import StudentExtractionService
from app.core.config import Settings
from app.evaluation.structure_benchmark.manifest import load_structure_manifest
from app.evidence.answer_regions import AnswerRegionDetector
from app.evidence.safety_filter import StudentCandidateSafetyFilter
from app.evidence.separator import EvidenceSeparator
from app.evidence.service import EvidenceSeparationService
from app.ocr.normalizer import OCRNormalizer
from app.ocr.providers.tesseract import TesseractOCRProvider
from app.ocr.service import OCRService
from app.structure.service import ExamStructureDetector
from scripts.evaluate_evidence_v2 import _verify_all_integrity
from scripts.evaluate_structure import _load_or_render_document

DEFAULT_MANIFEST = Path("data/evaluation/structure/manifest.json")
DEFAULT_OUTPUT = Path("data/runtime/student-answer-extraction")


def run_smoke(
    manifest_path: Path = DEFAULT_MANIFEST,
    output_root: Path = DEFAULT_OUTPUT,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Extract three safe-alias papers and return only operational metadata."""

    before = _verify_all_integrity()
    settings = settings or Settings()
    manifest = load_structure_manifest(manifest_path.resolve(strict=True))
    if len(manifest.papers) != 3:
        raise ValueError("Phase 4D smoke requires exactly three structure papers")
    provider = TesseractOCRProvider.from_system(
        language=settings.tesseract_language,
        psm=settings.tesseract_psm,
        timeout_seconds=settings.tesseract_timeout_seconds,
    )
    ocr_service = OCRService(provider, OCRNormalizer())
    evidence_service = EvidenceSeparationService(
        EvidenceSeparator(), AnswerRegionDetector()
    )
    safety_filter = StudentCandidateSafetyFilter()
    private_root = output_root.resolve()
    extraction_service = StudentExtractionService(
        ocr_service,
        StudentRegionCropper(private_root / "crops"),
    )
    papers: list[dict[str, Any]] = []
    for item in manifest.papers:
        document = _load_or_render_document(
            item.source_path,
            item.paper_alias,
            manifest_path.resolve().parent,
            settings,
        )
        page_ocr = ocr_service.process_document(document)
        structure = ExamStructureDetector().detect(document.pages, page_ocr)
        page_by_number = {page.page_number: page for page in document.pages}
        ocr_by_page = {result.page_number: result for result in page_ocr}
        safe_extractions = tuple(
            safety_filter.apply(
                evidence_service.analyze_region(
                    page_by_number[page_region.page_number],
                    ocr_by_page[page_region.page_number],
                    test_number=test.test_number,
                    region_bbox=page_region.bbox,
                )
            )
            for test in structure.tests
            for page_region in test.page_regions
        )
        extracted = extraction_service.extract_paper(
            structure,
            document.pages,
            safe_extractions,
            source_context=AnswerSourceContext.HISTORICAL_MARKED_EVALUATION,
        )
        write_structured_paper(
            extracted,
            private_root / "results" / f"{item.paper_alias}.json",
            private_root=private_root,
        )
        papers.append(_summary(item.paper_alias, extracted, len(structure.tests)))
    after = _verify_all_integrity()
    if before != after:
        raise RuntimeError("Frozen integrity changed during structured extraction")
    return {
        "paper_count": len(papers),
        "provider": provider.name,
        "model_version": provider.model_version,
        "papers": papers,
        "integrity": after,
        "private_output_root": str(private_root.relative_to(Path.cwd().resolve())),
    }


def _summary(
    alias: str, paper: StructuredStudentPaper, detected: int
) -> dict[str, Any]:
    statuses = {status: 0 for status in AnswerExtractionStatus}
    for answer in paper.answers:
        statuses[answer.status] += 1
    return {
        "paper_alias": alias,
        "tests_detected": detected,
        "tests_missing": len(paper.missing_test_numbers),
        "extracted": statuses[AnswerExtractionStatus.EXTRACTED],
        "empty": statuses[AnswerExtractionStatus.EMPTY],
        "review_required": statuses[AnswerExtractionStatus.REVIEW_REQUIRED],
        "ocr_failed": statuses[AnswerExtractionStatus.OCR_FAILED],
        "structure_incomplete": statuses[AnswerExtractionStatus.STRUCTURE_INCOMPLETE],
        "ocr_fragments": sum(len(answer.ocr_fragments) for answer in paper.answers),
        "duration_ms": paper.total_processing_duration_ms,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed local extraction smoke without displaying student text."""

    if argv:
        raise SystemExit("This command accepts no arguments")
    print(json.dumps(run_smoke(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

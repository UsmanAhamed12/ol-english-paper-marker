"""Application service producing one structured answer per detected Test."""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from time import perf_counter
from uuid import UUID

from app.answers.extraction import StudentRegionCropper
from app.answers.models import (
    AnswerExtractionProvenance,
    AnswerExtractionStatus,
    AnswerExtractionWarning,
    AnswerFragment,
    AnswerFragmentStatus,
    AnswerSourceContext,
    AnswerSourceRegion,
    PaperExtractionProvenance,
    ReviewEvidenceRegion,
    StructuredStudentAnswer,
    StructuredStudentPaper,
)
from app.answers.ordering import combine_fragment_text, order_answer_regions
from app.core.exceptions import AnswerExtractionError, OCRProcessingError
from app.domain.models.paper import PaperPage
from app.evidence.safety_models import CandidateDisposition, SafeTestExtraction
from app.ocr.models import BoundingBox
from app.ocr.service import OCRService
from app.structure.models import ExamStructure, TestRegion

EXTRACTION_VERSION = "structured-student-answer-v1"


class StudentExtractionService:
    """Convert accepted safe evidence into deterministic Test-owned answers."""

    def __init__(self, ocr_service: OCRService, cropper: StudentRegionCropper) -> None:
        self._ocr_service = ocr_service
        self._cropper = cropper

    def extract_paper(
        self,
        structure: ExamStructure,
        pages: tuple[PaperPage, ...],
        safe_extractions: tuple[SafeTestExtraction, ...],
        *,
        source_context: AnswerSourceContext = AnswerSourceContext.RUNTIME_UNMARKED,
    ) -> StructuredStudentPaper:
        """Extract detected Tests independently and preserve missing Tests."""

        started = perf_counter()
        page_by_number = self._validate_inputs(structure, pages, safe_extractions)
        by_test: dict[int, list[SafeTestExtraction]] = defaultdict(list)
        for item in safe_extractions:
            by_test[item.test_number].append(item)
        answers = tuple(
            self._extract_test(
                test,
                page_by_number,
                tuple(by_test[test.test_number]),
                source_context,
            )
            for test in structure.tests
        )
        return StructuredStudentPaper(
            paper_id=structure.paper_id,
            answers=answers,
            detected_test_numbers=tuple(item.test_number for item in answers),
            missing_test_numbers=structure.missing_test_numbers,
            review_required_test_numbers=_status_numbers(
                answers, AnswerExtractionStatus.REVIEW_REQUIRED
            ),
            empty_test_numbers=_status_numbers(answers, AnswerExtractionStatus.EMPTY),
            ocr_failed_test_numbers=_status_numbers(
                answers, AnswerExtractionStatus.OCR_FAILED
            ),
            structure_incomplete_test_numbers=_status_numbers(
                answers, AnswerExtractionStatus.STRUCTURE_INCOMPLETE
            ),
            total_processing_duration_ms=(perf_counter() - started) * 1000,
            provenance=PaperExtractionProvenance(
                extraction_version=EXTRACTION_VERSION,
                structure_test_count=len(structure.tests),
                source_page_count=len(pages),
                ocr_provider=self._ocr_service.provider_name,
                ocr_model_version=self._ocr_service.model_version,
                source_context=source_context,
            ),
        )

    def _extract_test(
        self,
        test: TestRegion,
        page_by_number: dict[int, PaperPage],
        safety: tuple[SafeTestExtraction, ...],
        source_context: AnswerSourceContext,
    ) -> StructuredStudentAnswer:
        started = perf_counter()
        expected_pages = tuple(item.page_number for item in test.page_regions)
        available_pages = {item.page_number for item in safety}
        incomplete = set(expected_pages) != available_pages
        ordered_safety = tuple(sorted(safety, key=lambda item: item.page_number))
        review_regions = tuple(
            ReviewEvidenceRegion(
                paper_id=decision.evidence.paper_id,
                test_number=decision.evidence.test_number,
                page_number=decision.evidence.page_number,
                bbox=decision.evidence.bbox,
                reasons=decision.reasons,
                source_image_path=decision.evidence.source_image_path,
            )
            for item in ordered_safety
            for decision in item.decisions
            if decision.disposition is CandidateDisposition.REVIEW_REQUIRED
        )
        answer_regions = order_answer_regions(
            tuple(region for item in ordered_safety for region in item.answer_regions)
        )
        source_regions: list[AnswerSourceRegion] = []
        fragments: list[AnswerFragment] = []
        for index, region in enumerate(answer_regions):
            try:
                source, crop_page = self._cropper.materialize(
                    page_by_number[region.page_number], region, region_index=index
                )
            except AnswerExtractionError:
                fragments.append(
                    _failed_fragment(
                        index,
                        region.paper_id,
                        region.test_number,
                        region.page_number,
                        region.bbox,
                        self._ocr_service,
                        error_code="crop_materialization_failed",
                    )
                )
                continue
            source_regions.append(source)
            try:
                result = self._ocr_service.process_page(crop_page)
            except OCRProcessingError:
                fragments.append(
                    _failed_fragment(
                        index,
                        region.paper_id,
                        region.test_number,
                        region.page_number,
                        region.bbox,
                        self._ocr_service,
                        error_code="ocr_processing_failed",
                    )
                )
                continue
            fragments.append(
                AnswerFragment(
                    fragment_index=index,
                    source_region_index=index,
                    paper_id=region.paper_id,
                    test_number=region.test_number,
                    page_number=region.page_number,
                    bbox=region.bbox,
                    status=AnswerFragmentStatus.EXTRACTED,
                    raw_text=result.raw_text,
                    normalized_text=result.normalized_text,
                    confidence=result.confidence,
                    provider=result.provider,
                    model_version=result.model_version,
                    processing_duration_ms=result.processing_duration_ms,
                    warnings=result.warnings,
                )
            )
        text = combine_fragment_text(
            tuple(
                item.normalized_text
                for item in fragments
                if item.status is AnswerFragmentStatus.EXTRACTED
            )
        )
        warnings = _warnings(
            incomplete, review_regions, fragments, text, source_context
        )
        status = _answer_status(
            incomplete=incomplete,
            review_regions=review_regions,
            fragments=tuple(fragments),
            text=text,
            source_context=source_context,
        )
        return StructuredStudentAnswer(
            paper_id=page_by_number[test.marker.page_number].paper_id,
            test_number=test.test_number,
            status=status,
            text=text,
            pages=expected_pages,
            source_regions=tuple(source_regions),
            ocr_fragments=tuple(fragments),
            review_regions=review_regions,
            evidence_confidence=_mean(
                tuple(item.evidence_confidence for item in source_regions)
            ),
            ocr_confidence=_mean(
                tuple(
                    item.confidence
                    for item in fragments
                    if item.status is AnswerFragmentStatus.EXTRACTED
                    and item.confidence is not None
                )
            ),
            warnings=warnings,
            provenance=AnswerExtractionProvenance(
                extraction_version=EXTRACTION_VERSION,
                test_label=test.label,
                marker_page_number=test.marker.page_number,
                marker_bbox=test.marker.bbox,
                marker_strategy=test.marker.strategy,
                marker_confidence=test.marker.confidence,
                marker_source_word_indices=test.marker.source_word_indices,
                safety_policy_versions=tuple(
                    dict.fromkeys(item.policy_version for item in ordered_safety)
                )
                or ("missing-safety-evidence",),
                ocr_provider=self._ocr_service.provider_name,
                ocr_model_version=self._ocr_service.model_version,
                source_context=source_context,
            ),
            processing_duration_ms=(perf_counter() - started) * 1000,
        )

    @staticmethod
    def _validate_inputs(
        structure: ExamStructure,
        pages: tuple[PaperPage, ...],
        safety: tuple[SafeTestExtraction, ...],
    ) -> dict[int, PaperPage]:
        if not pages or any(page.paper_id != structure.paper_id for page in pages):
            raise AnswerExtractionError("Extraction pages must belong to the structure")
        if [page.page_number for page in pages] != list(range(1, len(pages) + 1)):
            raise AnswerExtractionError("Extraction pages must be ordered from page 1")
        if len(pages) != structure.page_count:
            raise AnswerExtractionError("Extraction page count must match structure")
        detected = {test.test_number for test in structure.tests}
        expected_regions = {
            (test.test_number, region.page_number): region.bbox
            for test in structure.tests
            for region in test.page_regions
        }
        keys = [(item.test_number, item.page_number) for item in safety]
        if len(keys) != len(set(keys)):
            raise AnswerExtractionError("Safety evidence must be unique per Test page")
        if any(
            item.paper_id != structure.paper_id
            or item.test_number not in detected
            or item.page_number > len(pages)
            or expected_regions.get((item.test_number, item.page_number))
            != item.region_bbox
            for item in safety
        ):
            raise AnswerExtractionError("Safety evidence does not match exam structure")
        return {page.page_number: page for page in pages}


def _answer_status(
    *,
    incomplete: bool,
    review_regions: tuple[ReviewEvidenceRegion, ...],
    fragments: tuple[AnswerFragment, ...],
    text: str,
    source_context: AnswerSourceContext,
) -> AnswerExtractionStatus:
    if incomplete:
        return AnswerExtractionStatus.STRUCTURE_INCOMPLETE
    failed = any(item.status is AnswerFragmentStatus.OCR_FAILED for item in fragments)
    succeeded = any(item.status is AnswerFragmentStatus.EXTRACTED for item in fragments)
    if failed and not succeeded:
        return AnswerExtractionStatus.OCR_FAILED
    if source_context is AnswerSourceContext.HISTORICAL_MARKED_EVALUATION and succeeded:
        return AnswerExtractionStatus.REVIEW_REQUIRED
    if review_regions or failed or (succeeded and not text):
        return AnswerExtractionStatus.REVIEW_REQUIRED
    if succeeded:
        return AnswerExtractionStatus.EXTRACTED
    return AnswerExtractionStatus.EMPTY


def _warnings(
    incomplete: bool,
    review_regions: tuple[ReviewEvidenceRegion, ...],
    fragments: list[AnswerFragment],
    text: str,
    source_context: AnswerSourceContext,
) -> tuple[AnswerExtractionWarning, ...]:
    output: list[AnswerExtractionWarning] = []
    if incomplete:
        output.append(AnswerExtractionWarning.STRUCTURE_EVIDENCE_MISSING)
    if review_regions:
        output.append(AnswerExtractionWarning.REVIEW_EVIDENCE_PRESENT)
    if any(item.status is AnswerFragmentStatus.OCR_FAILED for item in fragments):
        output.append(AnswerExtractionWarning.OCR_REGION_FAILED)
    if (
        fragments
        and not text
        and all(item.status is AnswerFragmentStatus.EXTRACTED for item in fragments)
    ):
        output.append(AnswerExtractionWarning.OCR_RETURNED_EMPTY)
    if source_context is AnswerSourceContext.HISTORICAL_MARKED_EVALUATION and fragments:
        output.append(AnswerExtractionWarning.HISTORICAL_TEACHER_ANNOTATION_RISK)
    return tuple(output)


def _mean(values: tuple[float, ...]) -> float | None:
    return fmean(values) if values else None


def _status_numbers(
    answers: tuple[StructuredStudentAnswer, ...], status: AnswerExtractionStatus
) -> tuple[int, ...]:
    return tuple(item.test_number for item in answers if item.status is status)


def _failed_fragment(
    index: int,
    paper_id: UUID,
    test_number: int,
    page_number: int,
    bbox: BoundingBox,
    ocr_service: OCRService,
    *,
    error_code: str,
) -> AnswerFragment:
    return AnswerFragment(
        fragment_index=index,
        source_region_index=index,
        paper_id=paper_id,
        test_number=test_number,
        page_number=page_number,
        bbox=bbox,
        status=AnswerFragmentStatus.OCR_FAILED,
        provider=ocr_service.provider_name,
        model_version=ocr_service.model_version,
        processing_duration_ms=0,
        error_code=error_code,
    )

"""Production adapters composing existing document-processing services."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from uuid import UUID

from app.answers.extraction import StudentRegionCropper
from app.answers.models import AnswerExtractionStatus, AnswerSourceContext
from app.answers.service import StudentExtractionService
from app.evidence.safety_filter import StudentCandidateSafetyFilter
from app.evidence.service import EvidenceSeparationService
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.pdf_renderer import PDFRenderer
from app.ingestion.validators import PDFValidator
from app.marking_scheme.ingestion import MarkingSchemeIngestionService
from app.ocr.service import OCRService
from app.orchestration.interfaces import SchemePipelineOutput, StudentPipelineOutput
from app.orchestration.models import (
    PipelineStage,
    StageStatus,
    StageSummary,
)
from app.structure.service import ExamStructureDetector


class PDFInputValidator:
    """Adapt the parser-backed PDF validator to the orchestration Protocol."""

    def __init__(self, validator: PDFValidator) -> None:
        self._validator = validator

    def validate(self, path: Path) -> None:
        self._validator.validate(path)


class LocalStudentPaperPipeline:
    """Compose existing OCR, structure, safety, and extraction boundaries."""

    def __init__(
        self,
        *,
        loader: PDFLoader,
        renderer: PDFRenderer,
        page_ocr: OCRService,
        structure_detector: ExamStructureDetector,
        evidence_service: EvidenceSeparationService,
        safety_filter: StudentCandidateSafetyFilter,
        crop_root: Path,
    ) -> None:
        self._loader = loader
        self._renderer = renderer
        self._page_ocr = page_ocr
        self._structure_detector = structure_detector
        self._evidence_service = evidence_service
        self._safety_filter = safety_filter
        self._crop_root = crop_root

    def process(self, path: Path, *, run_id: UUID) -> StudentPipelineOutput:
        pdf_started = perf_counter()
        document = self._renderer.render(self._loader.load(path))
        page_ocr = self._page_ocr.process_document(document)
        pdf_warnings = tuple(
            sorted({warning.value for page in page_ocr for warning in page.warnings})
        )
        pdf_stage = StageSummary(
            stage=PipelineStage.STUDENT_PDF_PROCESSING,
            status=(
                StageStatus.COMPLETED_WITH_WARNINGS
                if pdf_warnings
                else StageStatus.COMPLETED
            ),
            duration_ms=(perf_counter() - pdf_started) * 1000,
            warnings=pdf_warnings,
        )

        structure_started = perf_counter()
        structure = self._structure_detector.detect(document.pages, page_ocr)
        structure_warnings = (
            (f"missing_tests:{len(structure.missing_test_numbers)}",)
            if structure.missing_test_numbers
            else ()
        )
        structure_stage = StageSummary(
            stage=PipelineStage.STRUCTURE_DETECTION,
            status=(
                StageStatus.COMPLETED_WITH_WARNINGS
                if structure_warnings
                else StageStatus.COMPLETED
            ),
            duration_ms=(perf_counter() - structure_started) * 1000,
            warnings=structure_warnings,
        )

        extraction_started = perf_counter()
        page_by_number = {page.page_number: page for page in document.pages}
        ocr_by_number = {page.page_number: page for page in page_ocr}
        safe_extractions = tuple(
            self._safety_filter.apply(
                self._evidence_service.analyze_region(
                    page_by_number[page_region.page_number],
                    ocr_by_number[page_region.page_number],
                    test_number=test.test_number,
                    region_bbox=page_region.bbox,
                )
            )
            for test in structure.tests
            for page_region in test.page_regions
        )
        extraction_service = StudentExtractionService(
            self._page_ocr,
            StudentRegionCropper(self._crop_root / run_id.hex),
        )
        paper = extraction_service.extract_paper(
            structure,
            document.pages,
            safe_extractions,
            source_context=AnswerSourceContext.RUNTIME_UNMARKED,
        )
        nonfinal = sum(
            answer.status is not AnswerExtractionStatus.EXTRACTED
            for answer in paper.answers
        )
        extraction_warnings = (f"non_extracted_tests:{nonfinal}",) if nonfinal else ()
        extraction_stage = StageSummary(
            stage=PipelineStage.STUDENT_ANSWER_EXTRACTION,
            status=(
                StageStatus.COMPLETED_WITH_WARNINGS
                if extraction_warnings
                else StageStatus.COMPLETED
            ),
            duration_ms=(perf_counter() - extraction_started) * 1000,
            warnings=extraction_warnings,
        )
        return StudentPipelineOutput(
            paper=paper,
            stages=(pdf_stage, structure_stage, extraction_stage),
        )


class LocalMarkingSchemePipeline:
    """Compose existing PDF rendering and scheme-ingestion boundaries."""

    def __init__(
        self,
        *,
        loader: PDFLoader,
        renderer: PDFRenderer,
        ingestion_service: MarkingSchemeIngestionService,
    ) -> None:
        self._loader = loader
        self._renderer = renderer
        self._ingestion = ingestion_service

    def process(self, path: Path, *, run_id: UUID) -> SchemePipelineOutput:
        del run_id  # The rendered document UUID already provides a safe directory.
        started = perf_counter()
        document = self._renderer.render(self._loader.load(path))
        scheme = self._ingestion.ingest(document)
        warning_count = len(scheme.warnings) + sum(
            len(test.warnings) for test in scheme.tests
        )
        warnings = (f"scheme_warnings:{warning_count}",) if warning_count else ()
        return SchemePipelineOutput(
            scheme=scheme,
            stages=(
                StageSummary(
                    stage=PipelineStage.MARKING_SCHEME_INGESTION,
                    status=(
                        StageStatus.COMPLETED_WITH_WARNINGS
                        if warnings
                        else StageStatus.COMPLETED
                    ),
                    duration_ms=(perf_counter() - started) * 1000,
                    warnings=warnings,
                ),
            ),
        )

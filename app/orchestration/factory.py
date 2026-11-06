"""Production-local construction for the end-to-end marking service."""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.evidence.answer_regions import AnswerRegionDetector
from app.evidence.safety_filter import StudentCandidateSafetyFilter
from app.evidence.separator import EvidenceSeparator
from app.evidence.service import EvidenceSeparationService
from app.feedback.service import PaperFeedbackService
from app.ingestion.pdf_loader import PDFLoader
from app.ingestion.pdf_renderer import PDFRenderer
from app.ingestion.validators import PDFValidator
from app.llm.factory import create_grading_chat_model
from app.marking.subjective.combined import CombinedMarkingService
from app.marking.subjective.ollama import OllamaSubjectiveMarkingProvider
from app.marking.subjective.service import SubjectiveMarkingService
from app.marking_scheme.alignment import TestAlignmentService
from app.marking_scheme.ingestion import MarkingSchemeIngestionService
from app.marking_scheme.parser import MarkingSchemeParser
from app.ocr.normalizer import OCRNormalizer
from app.ocr.providers.tesseract import TesseractOCRProvider
from app.ocr.service import OCRService
from app.orchestration.pipeline import (
    LocalMarkingSchemePipeline,
    LocalStudentPaperPipeline,
    PDFInputValidator,
)
from app.orchestration.service import PaperMarkingService
from app.structure.service import ExamStructureDetector
from app.vectorstore.chroma_store import ChromaKnowledgeBaseStore


def create_local_paper_marking_service(
    settings: Settings | None = None,
) -> PaperMarkingService:
    """Construct the sequential local FYP pipeline from existing components."""

    settings = settings or get_settings()
    validator = PDFValidator(
        max_file_size_bytes=settings.max_pdf_size_mb * 1024 * 1024,
        max_pages=settings.max_pdf_pages,
    )
    tesseract = TesseractOCRProvider.from_system(
        language=settings.tesseract_language,
        psm=settings.tesseract_psm,
        timeout_seconds=settings.tesseract_timeout_seconds,
        executable=settings.tesseract_executable,
    )
    page_ocr = OCRService(tesseract, OCRNormalizer())
    runtime_root = settings.runtime_data_dir / "paper-marking"
    student_pipeline = LocalStudentPaperPipeline(
        loader=PDFLoader(validator),
        renderer=PDFRenderer(
            runtime_data_dir=runtime_root / "student-documents",
            render_dpi=settings.pdf_render_dpi,
        ),
        page_ocr=page_ocr,
        structure_detector=ExamStructureDetector(),
        evidence_service=EvidenceSeparationService(
            EvidenceSeparator(), AnswerRegionDetector()
        ),
        safety_filter=StudentCandidateSafetyFilter(),
        crop_root=runtime_root / "student-crops",
    )
    scheme_pipeline = LocalMarkingSchemePipeline(
        loader=PDFLoader(validator),
        renderer=PDFRenderer(
            runtime_data_dir=runtime_root / "scheme-documents",
            render_dpi=settings.pdf_render_dpi,
        ),
        ingestion_service=MarkingSchemeIngestionService(
            page_ocr,
            MarkingSchemeParser(),
        ),
    )
    ollama_provider = OllamaSubjectiveMarkingProvider(
        chat_model=create_grading_chat_model(settings),
        model=settings.ollama_grading_model,
    )
    subjective = SubjectiveMarkingService(
        ollama_provider,
        grading_confidence_threshold=settings.grading_confidence_threshold,
        ocr_confidence_threshold=settings.ocr_confidence_threshold,
        syllabus_store=ChromaKnowledgeBaseStore.from_settings(settings),
        syllabus_retrieval_top_k=settings.retrieval_top_k,
    )
    return PaperMarkingService(
        input_validator=PDFInputValidator(validator),
        student_pipeline=student_pipeline,
        scheme_pipeline=scheme_pipeline,
        alignment_service=TestAlignmentService(),
        combined_marking_service=CombinedMarkingService(subjective),
        private_output_root=settings.runtime_data_dir,
        formative_feedback_service=PaperFeedbackService(
            ollama_provider,
            ocr_confidence_threshold=settings.ocr_confidence_threshold,
        ),
    )

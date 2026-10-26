"""Guarded application service for subjective marking and failure isolation."""

from __future__ import annotations

from decimal import Decimal
from time import perf_counter

from pydantic import ValidationError

from app.answers.models import AnswerExtractionStatus, StructuredStudentAnswer
from app.core.exceptions import (
    KnowledgeBaseRetrievalError,
    SubjectiveMarkingProviderError,
    SubjectiveMarkingTimeoutError,
)
from app.marking.models import StudentEvidenceReference
from app.marking.subjective.models import (
    ImprovementAdviceStatus,
    RubricEvidenceReference,
    SyllabusEvidence,
    SyllabusEvidenceReference,
    SubjectiveCriterionMarkResult,
    SubjectiveMarkingProvenance,
    SubjectiveMarkingStatus,
    SubjectiveMarkingWarning,
    SubjectivePaperMarkingResult,
    SubjectiveReasonCode,
)
from app.marking.subjective.prompting import (
    SUBJECTIVE_PROMPT_VERSION,
    build_subjective_prompt,
)
from app.marking.subjective.provider import SubjectiveMarkingProvider
from app.marking.subjective.validation import (
    SUBJECTIVE_VALIDATION_VERSION,
    SubjectiveValidationFailure,
    validate_subjective_response,
)
from app.marking_scheme.models import (
    AlignedPaper,
    AlignedTest,
    AlignmentStatus,
    CriterionType,
    MarkingCriterion,
)
from app.vectorstore.base import KnowledgeBaseStore
from app.syllabus.ingestion import SYLLABUS_SOURCE_ID

SUBJECTIVE_SERVICE_VERSION = "subjective-marking-service-v2"
SUBJECTIVE_TYPES = frozenset(
    {
        CriterionType.CONTENT_POINT,
        CriterionType.LANGUAGE_QUALITY,
        CriterionType.FORMAT_REQUIREMENT,
        CriterionType.HOLISTIC,
    }
)


class SubjectiveMarkingService:
    """Assess only subjective criteria while preserving every unsafe state."""

    def __init__(
        self,
        provider: SubjectiveMarkingProvider,
        *,
        grading_confidence_threshold: float = 0.70,
        ocr_confidence_threshold: float = 0.80,
        syllabus_store: KnowledgeBaseStore | None = None,
        syllabus_retrieval_top_k: int = 5,
    ) -> None:
        if not 0 <= grading_confidence_threshold <= 1:
            raise ValueError("Grading confidence threshold must be within [0, 1]")
        if not 0 <= ocr_confidence_threshold <= 1:
            raise ValueError("OCR confidence threshold must be within [0, 1]")
        if syllabus_retrieval_top_k < 1:
            raise ValueError("Syllabus retrieval top-k must be positive")
        self._provider = provider
        self._grading_threshold = grading_confidence_threshold
        self._ocr_threshold = ocr_confidence_threshold
        self._syllabus_store = syllabus_store
        self._syllabus_retrieval_top_k = syllabus_retrieval_top_k

    def mark_paper(self, aligned: AlignedPaper) -> SubjectivePaperMarkingResult:
        started = perf_counter()
        results = tuple(
            result for test in aligned.tests for result in self.mark_test(test)
        )
        confirmed = sum(
            (item.awarded_marks or Decimal("0") for item in results), Decimal("0")
        )
        review = sum(
            (
                item.maximum_marks or Decimal("0")
                for item in results
                if item.status
                not in {
                    SubjectiveMarkingStatus.MARKED,
                    SubjectiveMarkingStatus.UNSUPPORTED_CRITERION,
                }
            ),
            Decimal("0"),
        )
        unsupported = sum(
            (
                item.maximum_marks or Decimal("0")
                for item in results
                if item.status is SubjectiveMarkingStatus.UNSUPPORTED_CRITERION
            ),
            Decimal("0"),
        )
        return SubjectivePaperMarkingResult(
            scheme_id=aligned.scheme_id,
            student_paper_id=aligned.paper_id,
            criterion_results=results,
            confirmed_marks=confirmed,
            review_required_marks=review,
            unsupported_marks=unsupported,
            processing_duration_ms=(perf_counter() - started) * 1000,
        )

    def mark_test(
        self, aligned: AlignedTest
    ) -> tuple[SubjectiveCriterionMarkResult, ...]:
        scheme = aligned.marking_scheme_test
        if scheme is None:
            return ()
        return tuple(
            self._mark_criterion(aligned, criterion)
            for criterion in scheme.criteria
            if criterion.criterion_type in SUBJECTIVE_TYPES
            or criterion.criterion_type is CriterionType.UNKNOWN
        )

    def _mark_criterion(
        self,
        aligned: AlignedTest,
        criterion: MarkingCriterion,
    ) -> SubjectiveCriterionMarkResult:
        answer = aligned.student_answer
        if criterion.criterion_type is CriterionType.UNKNOWN:
            return self._unresolved(
                criterion,
                answer,
                SubjectiveMarkingStatus.UNSUPPORTED_CRITERION,
                SubjectiveReasonCode.UNKNOWN_CRITERION,
            )
        if criterion.maximum_marks is None:
            return self._unresolved(
                criterion,
                answer,
                SubjectiveMarkingStatus.SCHEME_INCOMPLETE,
                SubjectiveReasonCode.MISSING_MAXIMUM_MARKS,
                SubjectiveMarkingWarning.SCHEME_EVIDENCE_INCOMPLETE,
            )
        if answer is None or aligned.status is AlignmentStatus.STUDENT_TEST_MISSING:
            return self._unresolved(
                criterion,
                None,
                SubjectiveMarkingStatus.INSUFFICIENT_EVIDENCE,
                SubjectiveReasonCode.MISSING_STUDENT_TEST,
                SubjectiveMarkingWarning.MANUAL_REVIEW_REQUIRED,
            )
        if answer.status is AnswerExtractionStatus.OCR_FAILED:
            return self._unresolved(
                criterion,
                answer,
                SubjectiveMarkingStatus.OCR_FAILED,
                SubjectiveReasonCode.STUDENT_OCR_FAILED,
                SubjectiveMarkingWarning.OCR_EVIDENCE_UNCERTAIN,
            )
        if answer.status in {
            AnswerExtractionStatus.REVIEW_REQUIRED,
            AnswerExtractionStatus.STRUCTURE_INCOMPLETE,
        }:
            return self._unresolved(
                criterion,
                answer,
                SubjectiveMarkingStatus.REVIEW_REQUIRED,
                SubjectiveReasonCode.STUDENT_REVIEW_REQUIRED,
                SubjectiveMarkingWarning.MANUAL_REVIEW_REQUIRED,
            )
        if answer.status is AnswerExtractionStatus.EMPTY:
            return self._marked_empty(criterion, answer)
        if (
            answer.ocr_confidence is not None
            and answer.ocr_confidence < self._ocr_threshold
        ):
            return self._unresolved(
                criterion,
                answer,
                SubjectiveMarkingStatus.REVIEW_REQUIRED,
                SubjectiveReasonCode.LOW_OCR_CONFIDENCE,
                SubjectiveMarkingWarning.OCR_EVIDENCE_UNCERTAIN,
            )
        syllabus_evidence, advice_status = self._retrieve_syllabus(criterion, answer)
        prompt = build_subjective_prompt(
            criterion,
            answer,
            syllabus_evidence=syllabus_evidence,
        )
        try:
            response = self._provider.assess(prompt)
        except SubjectiveMarkingTimeoutError:
            return self._unresolved(
                criterion,
                answer,
                SubjectiveMarkingStatus.PROVIDER_FAILED,
                SubjectiveReasonCode.PROVIDER_TIMEOUT,
                SubjectiveMarkingWarning.PROVIDER_FAILURE,
            )
        except SubjectiveMarkingProviderError:
            return self._unresolved(
                criterion,
                answer,
                SubjectiveMarkingStatus.PROVIDER_FAILED,
                SubjectiveReasonCode.PROVIDER_UNAVAILABLE,
                SubjectiveMarkingWarning.PROVIDER_FAILURE,
            )
        validated_result = validate_subjective_response(response.raw_text, criterion)
        if isinstance(validated_result, SubjectiveValidationFailure):
            return self._unresolved(
                criterion,
                answer,
                SubjectiveMarkingStatus.INVALID_MODEL_RESPONSE,
                validated_result.reason,
                SubjectiveMarkingWarning.MODEL_RESPONSE_INVALID,
                raw_response=response.raw_text,
                duration_ms=response.processing_duration_ms,
            )
        validated = validated_result.assessment
        improvement_tips = validated_result.improvement_tips
        warnings: list[SubjectiveMarkingWarning] = []
        if advice_status is ImprovementAdviceStatus.SYLLABUS_NOT_INDEXED:
            warnings.append(SubjectiveMarkingWarning.SYLLABUS_EVIDENCE_UNAVAILABLE)
        elif advice_status is ImprovementAdviceStatus.SYLLABUS_RETRIEVAL_FAILED:
            warnings.append(SubjectiveMarkingWarning.SYLLABUS_RETRIEVAL_FAILED)
        elif validated_result.advice_status is ImprovementAdviceStatus.INVALID_MODEL_TIPS:
            warnings.append(SubjectiveMarkingWarning.IMPROVEMENT_TIPS_INVALID)
        if not syllabus_evidence:
            improvement_tips = ()
        elif validated_result.advice_status is ImprovementAdviceStatus.INVALID_MODEL_TIPS:
            advice_status = ImprovementAdviceStatus.INVALID_MODEL_TIPS
        else:
            advice_status = validated_result.advice_status
        if (
            validated.status is SubjectiveMarkingStatus.REVIEW_REQUIRED
            or validated.confidence < self._grading_threshold
        ):
            reason = (
                SubjectiveReasonCode.LOW_MODEL_CONFIDENCE
                if validated.confidence < self._grading_threshold
                else SubjectiveReasonCode.STUDENT_REVIEW_REQUIRED
            )
            return SubjectiveCriterionMarkResult(
                criterion_id=criterion.criterion_id,
                test_number=criterion.test_number,
                criterion_type=criterion.criterion_type,
                status=SubjectiveMarkingStatus.REVIEW_REQUIRED,
                student_evidence=_student_reference(answer),
                rubric_evidence=_rubric_reference(criterion),
                advisory_marks=validated.awarded_marks,
                maximum_marks=criterion.maximum_marks,
                matched_requirements=validated.matched_requirements,
                missing_requirements=validated.missing_requirements,
                reason_codes=(reason, *validated.reason_codes),
                feedback=validated.feedback,
                syllabus_references=tuple(
                    item.reference for item in syllabus_evidence
                ),
                improvement_advice_status=(
                    ImprovementAdviceStatus.REVIEW_REQUIRED
                    if syllabus_evidence
                    else advice_status
                ),
                confidence=validated.confidence,
                warnings=(
                    SubjectiveMarkingWarning.MANUAL_REVIEW_REQUIRED,
                    SubjectiveMarkingWarning.MODEL_CONFIDENCE_LOW,
                    *warnings,
                ),
                raw_model_response=response.raw_text,
                processing_duration_ms=response.processing_duration_ms,
                provenance=self._provenance(),
            )
        return SubjectiveCriterionMarkResult(
            criterion_id=criterion.criterion_id,
            test_number=criterion.test_number,
            criterion_type=criterion.criterion_type,
            status=SubjectiveMarkingStatus.MARKED,
            student_evidence=_student_reference(answer),
            rubric_evidence=_rubric_reference(criterion),
            awarded_marks=validated.awarded_marks,
            maximum_marks=criterion.maximum_marks,
            matched_requirements=validated.matched_requirements,
            missing_requirements=validated.missing_requirements,
            reason_codes=(
                SubjectiveReasonCode.RUBRIC_ASSESSMENT_ACCEPTED,
                *validated.reason_codes,
            ),
            feedback=validated.feedback,
            improvement_tips=improvement_tips,
            syllabus_references=tuple(item.reference for item in syllabus_evidence),
            improvement_advice_status=advice_status,
            confidence=validated.confidence,
            warnings=tuple(warnings),
            raw_model_response=response.raw_text,
            processing_duration_ms=response.processing_duration_ms,
            provenance=self._provenance(),
        )

    def _retrieve_syllabus(
        self,
        criterion: MarkingCriterion,
        answer: StructuredStudentAnswer,
    ) -> tuple[tuple[SyllabusEvidence, ...], ImprovementAdviceStatus]:
        if self._syllabus_store is None:
            return (), ImprovementAdviceStatus.SYLLABUS_NOT_INDEXED
        query = (
            f"Grade 10 English syllabus guidance for Test {criterion.test_number}. "
            f"Assessment criterion: {criterion.description}. "
            f"Student response: {answer.text[:1500]}"
        )
        try:
            results = self._syllabus_store.search(
                query,
                k=self._syllabus_retrieval_top_k,
                metadata_filter={
                    "source_type": "syllabus",
                    "source_id": SYLLABUS_SOURCE_ID,
                },
            )
        except KnowledgeBaseRetrievalError:
            return (), ImprovementAdviceStatus.SYLLABUS_RETRIEVAL_FAILED
        evidence: list[SyllabusEvidence] = []
        for result in results:
            metadata = result.metadata
            if (
                metadata.get("source_type") != "syllabus"
                or metadata.get("source_id") != SYLLABUS_SOURCE_ID
            ):
                continue
            try:
                reference = SyllabusEvidenceReference(
                    source_id=SYLLABUS_SOURCE_ID,
                    source_name=metadata["source_name"],
                    source_sha256=metadata["source_sha256"],
                    page_number=metadata["page_number"],
                )
                evidence.append(
                    SyllabusEvidence(text=result.text, reference=reference)
                )
            except (KeyError, ValidationError) as error:
                raise KnowledgeBaseRetrievalError(
                    "Retrieved syllabus evidence has invalid provenance"
                ) from error
        if not evidence:
            return (), ImprovementAdviceStatus.SYLLABUS_NOT_INDEXED
        return tuple(evidence), ImprovementAdviceStatus.AVAILABLE

    def _marked_empty(
        self, criterion: MarkingCriterion, answer: StructuredStudentAnswer
    ) -> SubjectiveCriterionMarkResult:
        return SubjectiveCriterionMarkResult(
            criterion_id=criterion.criterion_id,
            test_number=criterion.test_number,
            criterion_type=criterion.criterion_type,
            status=SubjectiveMarkingStatus.MARKED,
            student_evidence=_student_reference(answer),
            rubric_evidence=_rubric_reference(criterion),
            awarded_marks=Decimal("0"),
            maximum_marks=criterion.maximum_marks,
            reason_codes=(SubjectiveReasonCode.EMPTY_STUDENT_ANSWER,),
            feedback="No student answer evidence was extracted for this criterion.",
            processing_duration_ms=0,
            provenance=self._provenance(),
        )

    def _unresolved(
        self,
        criterion: MarkingCriterion,
        answer: StructuredStudentAnswer | None,
        status: SubjectiveMarkingStatus,
        reason: SubjectiveReasonCode,
        warning: SubjectiveMarkingWarning | None = None,
        *,
        raw_response: str | None = None,
        duration_ms: float = 0,
    ) -> SubjectiveCriterionMarkResult:
        return SubjectiveCriterionMarkResult(
            criterion_id=criterion.criterion_id,
            test_number=criterion.test_number,
            criterion_type=criterion.criterion_type,
            status=status,
            student_evidence=_student_reference(answer) if answer else None,
            rubric_evidence=_rubric_reference(criterion),
            maximum_marks=criterion.maximum_marks,
            reason_codes=(reason,),
            warnings=(warning,) if warning else (),
            raw_model_response=raw_response,
            processing_duration_ms=duration_ms,
            provenance=self._provenance(),
        )

    def _provenance(self) -> SubjectiveMarkingProvenance:
        return SubjectiveMarkingProvenance(
            service_version=SUBJECTIVE_SERVICE_VERSION,
            validation_version=SUBJECTIVE_VALIDATION_VERSION,
            prompt_version=SUBJECTIVE_PROMPT_VERSION,
            provider=self._provider.name,
            model_version=self._provider.model_version,
        )


def _student_reference(answer: StructuredStudentAnswer) -> StudentEvidenceReference:
    return StudentEvidenceReference(
        paper_id=answer.paper_id,
        test_number=answer.test_number,
        extraction_status=answer.status,
        original_text=answer.text,
        pages=answer.pages,
    )


def _rubric_reference(criterion: MarkingCriterion) -> RubricEvidenceReference:
    return RubricEvidenceReference(
        criterion_id=criterion.criterion_id,
        test_number=criterion.test_number,
        criterion_type=criterion.criterion_type,
        description=criterion.description,
        maximum_marks=criterion.maximum_marks,
        source_page=criterion.provenance.source.page_number,
    )

"""Immutable marking-scheme and student-to-scheme alignment models."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.answers.models import StructuredStudentAnswer
from app.ocr.models import BoundingBox, OCRWarningCode, OCRWord

PositiveTestNumber = Annotated[int, Field(ge=1, le=99)]
NonNegativeDuration = Annotated[float, Field(ge=0.0)]
PositiveMarks = Annotated[Decimal, Field(gt=0)]


class CriterionType(StrEnum):
    """Structural marking categories; they do not implement scoring."""

    EXACT_ANSWER = "exact_answer"
    ACCEPTED_VARIANTS = "accepted_variants"
    KEYWORD = "keyword"
    MULTI_PART = "multi_part"
    CONTENT_POINT = "content_point"
    LANGUAGE_QUALITY = "language_quality"
    FORMAT_REQUIREMENT = "format_requirement"
    HOLISTIC = "holistic"
    UNKNOWN = "unknown"


class CriterionParserRule(StrEnum):
    """Deterministic rule that produced a criterion classification."""

    NUMBERED_ANSWER = "numbered_answer"
    ACCEPTED_ANSWER_VARIANTS = "accepted_answer_variants"
    KEYWORD_LABEL = "keyword_label"
    CONTENT_LABEL = "content_label"
    LANGUAGE_LABEL = "language_label"
    FORMAT_LABEL = "format_label"
    HOLISTIC_LABEL = "holistic_label"
    PRESERVED_UNKNOWN = "preserved_unknown"


class MarkingSchemeWarning(StrEnum):
    """Safe, typed ingestion warnings requiring no fabricated data."""

    OCR_PAGE_FAILED = "ocr_page_failed"
    TEST_MAXIMUM_UNKNOWN = "test_maximum_unknown"
    TOTAL_MAXIMUM_INCOMPLETE = "total_maximum_incomplete"
    UNKNOWN_CRITERION = "unknown_criterion"


class MarkingSchemeOCRStatus(StrEnum):
    """Outcome of OCR for one marking-scheme page."""

    EXTRACTED = "extracted"
    OCR_FAILED = "ocr_failed"


class SchemeSourceRegion(BaseModel):
    """Page-local source evidence retained for auditability."""

    model_config = ConfigDict(frozen=True)

    page_number: Annotated[int, Field(gt=0)]
    bbox: BoundingBox | None = None
    source_word_indices: tuple[Annotated[int, Field(ge=0)], ...] = ()
    raw_source_text: str


class CriterionProvenance(BaseModel):
    """Parser and OCR evidence used for one criterion."""

    model_config = ConfigDict(frozen=True)

    parser_rule: CriterionParserRule
    source: SchemeSourceRegion


class CriterionComponent(BaseModel):
    """One explicitly allocated, independently markable criterion component."""

    model_config = ConfigDict(frozen=True)

    component_id: Annotated[str, Field(pattern=r"^component-[0-9]{2}$")]
    accepted_answers: tuple[Annotated[str, Field(min_length=1)], ...] = ()
    keywords: tuple[Annotated[str, Field(min_length=1)], ...] = ()
    maximum_marks: PositiveMarks

    @model_validator(mode="after")
    def component_has_explicit_expected_evidence(self) -> Self:
        if bool(self.accepted_answers) == bool(self.keywords):
            raise ValueError(
                "A criterion component requires accepted answers or keywords"
            )
        return self


class MarkingCriterion(BaseModel):
    """One structural rubric item without scoring behavior."""

    model_config = ConfigDict(frozen=True)

    criterion_id: Annotated[str, Field(pattern=r"^test-[0-9]{2}-criterion-[0-9]{3}$")]
    test_number: PositiveTestNumber
    criterion_type: CriterionType
    description: Annotated[str, Field(min_length=1)]
    accepted_answers: tuple[Annotated[str, Field(min_length=1)], ...] = ()
    keywords: tuple[Annotated[str, Field(min_length=1)], ...] = ()
    components: tuple[CriterionComponent, ...] = ()
    maximum_marks: PositiveMarks | None = None
    partial_credit_allowed: bool | None = None
    provenance: CriterionProvenance

    @field_validator("description")
    @classmethod
    def description_is_not_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Criterion description must not be blank")
        return value

    @model_validator(mode="after")
    def explicit_components_match_allocation(self) -> Self:
        expected_ids = tuple(
            f"component-{index:02d}" for index in range(1, len(self.components) + 1)
        )
        if tuple(item.component_id for item in self.components) != expected_ids:
            raise ValueError("Criterion components must be deterministic and ordered")
        if self.components and self.maximum_marks is not None:
            allocated = sum(
                (item.maximum_marks for item in self.components), Decimal("0")
            )
            if allocated != self.maximum_marks:
                raise ValueError("Component allocations must equal criterion maximum")
        return self


class MarkingSchemeTest(BaseModel):
    """Ordered marking material owned by one explicit Test number."""

    model_config = ConfigDict(frozen=True)

    test_number: PositiveTestNumber
    title: Annotated[str, Field(min_length=1)] | None = None
    maximum_marks: PositiveMarks | None = None
    criteria: tuple[MarkingCriterion, ...]
    accepted_answers: tuple[Annotated[str, Field(min_length=1)], ...] = ()
    source_pages: Annotated[tuple[int, ...], Field(min_length=1)]
    source_regions: Annotated[tuple[SchemeSourceRegion, ...], Field(min_length=1)]
    warnings: tuple[MarkingSchemeWarning, ...] = ()

    @model_validator(mode="after")
    def test_content_is_consistent(self) -> Self:
        if tuple(sorted(set(self.source_pages))) != self.source_pages:
            raise ValueError("Scheme Test pages must be unique and ordered")
        if any(item.test_number != self.test_number for item in self.criteria):
            raise ValueError("Criteria must belong to their marking-scheme Test")
        expected_ids = tuple(
            f"test-{self.test_number:02d}-criterion-{index:03d}"
            for index in range(1, len(self.criteria) + 1)
        )
        if tuple(item.criterion_id for item in self.criteria) != expected_ids:
            raise ValueError("Criterion identifiers must be deterministic and ordered")
        return self


class MarkingSchemePageOCR(BaseModel):
    """Raw and normalized OCR evidence for one source page."""

    model_config = ConfigDict(frozen=True)

    page_number: Annotated[int, Field(gt=0)]
    status: MarkingSchemeOCRStatus
    raw_text: str = ""
    normalized_text: str = ""
    words: tuple[OCRWord, ...] = ()
    provider: Annotated[str, Field(min_length=1)]
    model_version: Annotated[str, Field(min_length=1)] | None = None
    duration_ms: NonNegativeDuration
    warnings: tuple[OCRWarningCode, ...] = ()
    error_code: Annotated[str, Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def failure_has_no_fake_output(self) -> Self:
        if self.status is MarkingSchemeOCRStatus.OCR_FAILED:
            if (
                self.raw_text
                or self.normalized_text
                or self.words
                or not self.error_code
            ):
                raise ValueError("Failed scheme OCR requires an error and no output")
        elif self.error_code is not None:
            raise ValueError("Successful scheme OCR cannot contain an error code")
        return self


class MarkingSchemeProvenance(BaseModel):
    """Document, OCR, and parser versions used for ingestion."""

    model_config = ConfigDict(frozen=True)

    ingestion_version: Annotated[str, Field(min_length=1)]
    parser_version: Annotated[str, Field(min_length=1)]
    source_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    ocr_provider: Annotated[str, Field(min_length=1)]
    ocr_model_version: Annotated[str, Field(min_length=1)] | None = None
    successful_ocr_pages: tuple[Annotated[int, Field(gt=0)], ...] = ()
    failed_ocr_pages: tuple[Annotated[int, Field(gt=0)], ...] = ()


class StructuredMarkingScheme(BaseModel):
    """Stable marking-scheme contract for later deterministic marking."""

    model_config = ConfigDict(frozen=True)

    scheme_id: UUID
    source_document_id: UUID
    page_count: Annotated[int, Field(gt=0)]
    tests: Annotated[tuple[MarkingSchemeTest, ...], Field(min_length=1)]
    total_available_marks: PositiveMarks | None = None
    page_ocr: Annotated[tuple[MarkingSchemePageOCR, ...], Field(min_length=1)]
    warnings: tuple[MarkingSchemeWarning, ...] = ()
    provenance: MarkingSchemeProvenance
    processing_duration_ms: NonNegativeDuration

    @model_validator(mode="after")
    def scheme_is_ordered_and_complete(self) -> Self:
        numbers = tuple(item.test_number for item in self.tests)
        if numbers != tuple(sorted(set(numbers))):
            raise ValueError("Marking-scheme Tests must be unique and ordered")
        if tuple(item.page_number for item in self.page_ocr) != tuple(
            range(1, self.page_count + 1)
        ):
            raise ValueError("Scheme OCR pages must cover the source document")
        known = tuple(item.maximum_marks for item in self.tests)
        if self.total_available_marks is not None and (
            any(item is None for item in known)
            or self.total_available_marks != sum(item for item in known if item)
        ):
            raise ValueError("Total marks require all Test maxima and their exact sum")
        return self


class AlignmentStatus(StrEnum):
    """Structural readiness of one exact-number student/scheme relationship."""

    ALIGNED = "aligned"
    STUDENT_TEST_MISSING = "student_test_missing"
    SCHEME_TEST_MISSING = "scheme_test_missing"
    STUDENT_REVIEW_REQUIRED = "student_review_required"
    STUDENT_OCR_FAILED = "student_ocr_failed"
    STRUCTURE_INCOMPLETE = "structure_incomplete"


class AlignmentWarning(StrEnum):
    """Typed reason an aligned Test cannot proceed automatically."""

    STUDENT_TEST_ABSENT = "student_test_absent"
    SCHEME_TEST_ABSENT = "scheme_test_absent"
    STUDENT_REVIEW_REQUIRED = "student_review_required"
    STUDENT_OCR_FAILED = "student_ocr_failed"
    STUDENT_STRUCTURE_INCOMPLETE = "student_structure_incomplete"
    SCHEME_MAXIMUM_UNKNOWN = "scheme_maximum_unknown"


class AlignedTest(BaseModel):
    """Exact Test-number association preserving both source contracts."""

    model_config = ConfigDict(frozen=True)

    test_number: PositiveTestNumber
    status: AlignmentStatus
    student_answer: StructuredStudentAnswer | None = None
    marking_scheme_test: MarkingSchemeTest | None = None
    warnings: tuple[AlignmentWarning, ...] = ()

    @model_validator(mode="after")
    def alignment_matches_test_number(self) -> Self:
        if self.student_answer and self.student_answer.test_number != self.test_number:
            raise ValueError("Student answer does not match aligned Test")
        if (
            self.marking_scheme_test
            and self.marking_scheme_test.test_number != self.test_number
        ):
            raise ValueError("Marking scheme does not match aligned Test")
        if self.student_answer is None and self.marking_scheme_test is None:
            raise ValueError("An alignment requires student or scheme evidence")
        return self


class AlignedPaper(BaseModel):
    """Deterministically ordered student/scheme Test associations."""

    model_config = ConfigDict(frozen=True)

    paper_id: UUID
    scheme_id: UUID
    tests: tuple[AlignedTest, ...]
    aligned_test_numbers: tuple[PositiveTestNumber, ...] = ()
    student_missing_test_numbers: tuple[PositiveTestNumber, ...] = ()
    scheme_missing_test_numbers: tuple[PositiveTestNumber, ...] = ()
    review_required_test_numbers: tuple[PositiveTestNumber, ...] = ()
    ocr_failed_test_numbers: tuple[PositiveTestNumber, ...] = ()
    structure_incomplete_test_numbers: tuple[PositiveTestNumber, ...] = ()

    @model_validator(mode="after")
    def summaries_match_tests(self) -> Self:
        numbers = tuple(item.test_number for item in self.tests)
        if numbers != tuple(sorted(set(numbers))):
            raise ValueError("Aligned Tests must be unique and ordered")
        mapping = {
            AlignmentStatus.ALIGNED: self.aligned_test_numbers,
            AlignmentStatus.STUDENT_TEST_MISSING: self.student_missing_test_numbers,
            AlignmentStatus.SCHEME_TEST_MISSING: self.scheme_missing_test_numbers,
            AlignmentStatus.STUDENT_REVIEW_REQUIRED: (
                self.review_required_test_numbers
            ),
            AlignmentStatus.STUDENT_OCR_FAILED: self.ocr_failed_test_numbers,
            AlignmentStatus.STRUCTURE_INCOMPLETE: (
                self.structure_incomplete_test_numbers
            ),
        }
        for status, summary in mapping.items():
            actual = tuple(
                item.test_number for item in self.tests if item.status is status
            )
            if actual != summary:
                raise ValueError("Alignment summary does not match Test statuses")
        return self

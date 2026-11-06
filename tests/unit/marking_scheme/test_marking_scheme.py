"""Synthetic tests for marking-scheme ingestion and Test alignment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from app.answers.models import (
    AnswerExtractionProvenance,
    AnswerExtractionStatus,
    AnswerSourceContext,
    PaperExtractionProvenance,
    StructuredStudentAnswer,
    StructuredStudentPaper,
)
from app.core.exceptions import MarkingSchemeIngestionError
from app.domain.models.paper import PaperDocument, PaperPage
from app.marking_scheme.alignment import TestAlignmentService
from app.marking_scheme.ingestion import MarkingSchemeIngestionService
from app.marking_scheme.models import (
    AlignmentStatus,
    CriterionParserRule,
    CriterionType,
    MarkingSchemeTest,
    MarkingSchemeWarning,
    SchemeSourceRegion,
    StructuredMarkingScheme,
)
from app.marking_scheme.parser import MarkingSchemeParser, SourceLine
from app.marking_scheme.serialization import (
    aligned_paper_json,
    marking_scheme_json,
    write_private_json,
)
from app.ocr.models import BoundingBox, OCRExtraction
from app.ocr.normalizer import OCRNormalizer
from app.ocr.service import OCRService
from app.structure.models import MarkerDetectionStrategy


@dataclass
class FakeSchemeOCRProvider:
    """Return deterministic synthetic page text or errors."""

    outputs: dict[int, str | Exception]
    name: str = "fake-scheme-ocr"
    model_version: str | None = "fake-v1"

    def extract_page(self, page: PaperPage) -> OCRExtraction:
        output = self.outputs[page.page_number]
        if isinstance(output, Exception):
            raise output
        return OCRExtraction(
            raw_text=output,
            confidence=0.9,
            processing_duration_ms=1,
        )


def _document(tmp_path: Path, page_count: int) -> PaperDocument:
    paper_id = uuid4()
    source = tmp_path / "synthetic-scheme.pdf"
    source.write_bytes(b"synthetic scheme")
    pages: list[PaperPage] = []
    for page_number in range(1, page_count + 1):
        image_path = (tmp_path / f"page_{page_number:04d}.png").resolve()
        image = np.full((600, 800, 3), 255, dtype=np.uint8)
        assert cv2.imwrite(str(image_path), image)
        pages.append(
            PaperPage(
                paper_id=paper_id,
                page_number=page_number,
                image_path=image_path,
                width=800,
                height=600,
            )
        )
    return PaperDocument(
        paper_id=paper_id,
        source_path=source.resolve(),
        original_filename=source.name,
        page_count=page_count,
        file_size_bytes=source.stat().st_size,
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        pages=tuple(pages),
    )


def _ingest(
    tmp_path: Path, outputs: dict[int, str | Exception]
) -> StructuredMarkingScheme:
    provider = FakeSchemeOCRProvider(outputs)
    return MarkingSchemeIngestionService(
        OCRService(provider, OCRNormalizer()),
        MarkingSchemeParser(),
    ).ingest(_document(tmp_path, len(outputs)))


def _student_answer(
    paper_id: UUID,
    test_number: int,
    status: AnswerExtractionStatus,
) -> StructuredStudentAnswer:
    return StructuredStudentAnswer(
        paper_id=paper_id,
        test_number=test_number,
        status=status,
        text="",
        pages=(1,),
        provenance=AnswerExtractionProvenance(
            extraction_version="test-v1",
            test_label=f"Test {test_number:02d}",
            marker_page_number=1,
            marker_bbox=BoundingBox(x=0, y=0, width=10, height=10),
            marker_strategy=MarkerDetectionStrategy.EXACT_TOKENS,
            marker_confidence=1,
            marker_source_word_indices=(0,),
            safety_policy_versions=("safety-v1",),
            ocr_provider="fake",
            ocr_model_version="fake-v1",
            source_context=AnswerSourceContext.RUNTIME_UNMARKED,
        ),
        processing_duration_ms=1,
    )


def _student_paper(
    statuses: dict[int, AnswerExtractionStatus],
    *,
    paper_id: UUID | None = None,
) -> StructuredStudentPaper:
    identifier = paper_id or uuid4()
    answers = tuple(
        _student_answer(identifier, number, status)
        for number, status in sorted(statuses.items())
    )
    return StructuredStudentPaper(
        paper_id=identifier,
        answers=answers,
        detected_test_numbers=tuple(statuses),
        review_required_test_numbers=tuple(
            number
            for number, status in statuses.items()
            if status is AnswerExtractionStatus.REVIEW_REQUIRED
        ),
        empty_test_numbers=tuple(
            number
            for number, status in statuses.items()
            if status is AnswerExtractionStatus.EMPTY
        ),
        ocr_failed_test_numbers=tuple(
            number
            for number, status in statuses.items()
            if status is AnswerExtractionStatus.OCR_FAILED
        ),
        structure_incomplete_test_numbers=tuple(
            number
            for number, status in statuses.items()
            if status is AnswerExtractionStatus.STRUCTURE_INCOMPLETE
        ),
        total_processing_duration_ms=1,
        provenance=PaperExtractionProvenance(
            extraction_version="test-v1",
            structure_test_count=len(answers),
            source_page_count=1,
            ocr_provider="fake",
            ocr_model_version="fake-v1",
            source_context=AnswerSourceContext.RUNTIME_UNMARKED,
        ),
    )


def test_structured_scheme_accepts_ordered_tests_and_sums_marks(tmp_path: Path) -> None:
    scheme = _ingest(
        tmp_path,
        {
            1: "Test 01\n1. cat (1 mark)\nTotal: 1 mark\n"
            "Test 02\n1. blue (2 marks)\nTotal: 2 marks"
        },
    )

    assert tuple(item.test_number for item in scheme.tests) == (1, 2)
    assert scheme.total_available_marks == Decimal("3")
    assert scheme.source_document_id != scheme.scheme_id


def test_structured_scheme_rejects_reordered_tests(tmp_path: Path) -> None:
    scheme = _ingest(
        tmp_path,
        {1: "Test 01\n1. cat (1 mark)\nTest 02\n1. dog (1 mark)"},
    )
    payload = scheme.model_dump()
    payload["tests"] = tuple(reversed(payload["tests"]))

    with pytest.raises(ValidationError, match="unique and ordered"):
        StructuredMarkingScheme.model_validate(payload)


@pytest.mark.parametrize("marker", ["Test 1", "Test 01", "TEST 01"])
def test_test_marker_variants_are_parsed(marker: str) -> None:
    tests = MarkingSchemeParser().parse(
        (SourceLine(page_number=1, text=f"{marker}\n".strip()),)
    )

    assert tests[0].test_number == 1


def test_compact_test_marker_is_parsed() -> None:
    tests = MarkingSchemeParser().parse((SourceLine(page_number=1, text="Test01"),))

    assert tests[0].test_number == 1


def test_numbered_criterion_and_source_provenance_are_preserved() -> None:
    bbox = BoundingBox(x=10, y=20, width=100, height=15)
    tests = MarkingSchemeParser().parse(
        (
            SourceLine(page_number=1, text="Test 03"),
            SourceLine(
                page_number=1,
                text="1. orderd (1 mark)",
                bbox=bbox,
                source_word_indices=(2, 3, 4),
            ),
        )
    )
    criterion = tests[0].criteria[0]

    assert criterion.criterion_type is CriterionType.EXACT_ANSWER
    assert criterion.description == "orderd"
    assert criterion.accepted_answers == ("orderd",)
    assert criterion.maximum_marks == 1
    assert criterion.provenance.source.bbox == bbox
    assert criterion.provenance.source.source_word_indices == (2, 3, 4)


def test_accepted_answer_variants_are_split_conservatively() -> None:
    tests = MarkingSchemeParser().parse(
        (
            SourceLine(page_number=1, text="Test 04"),
            SourceLine(
                page_number=1,
                text="Accepted answers: colour / color [1 mark]",
            ),
        )
    )

    criterion = tests[0].criteria[0]
    assert criterion.criterion_type is CriterionType.ACCEPTED_VARIANTS
    assert criterion.accepted_answers == ("colour", "color")


def test_parenthesized_numbered_answer_key_is_split_into_scored_items() -> None:
    tests = MarkingSchemeParser().parse(
        (
            SourceLine(page_number=1, text="Test 01"),
            SourceLine(page_number=1, text="(01) c (02) e (03) f"),
            SourceLine(page_number=1, text="(04) a (05) d (06) b"),
        )
    )

    assert len(tests[0].criteria) == 6
    assert tuple(item.accepted_answers for item in tests[0].criteria) == (
        ("c",),
        ("e",),
        ("f",),
        ("a",),
        ("d",),
        ("b",),
    )
    assert all(item.maximum_marks == 1 for item in tests[0].criteria)
    assert tests[0].maximum_marks == 6
    assert all(
        item.provenance.parser_rule is CriterionParserRule.PARENTHESIZED_ANSWER_KEY
        for item in tests[0].criteria
    )


def test_parenthesized_lettered_answer_key_and_ocr_number_confusions() -> None:
    tests = MarkingSchemeParser().parse(
        (
            SourceLine(page_number=1, text="Test 05"),
            SourceLine(page_number=1, text="(a) krill/tiny organisms (b) blubber"),
            SourceLine(page_number=1, text="(Ol) colour (02) harbour"),
        )
    )

    assert tuple(item.criterion_type for item in tests[0].criteria) == (
        CriterionType.ACCEPTED_VARIANTS,
        CriterionType.EXACT_ANSWER,
        CriterionType.EXACT_ANSWER,
        CriterionType.EXACT_ANSWER,
    )
    assert tests[0].criteria[0].accepted_answers == ("krill", "tiny organisms")
    assert all(item.maximum_marks == 1 for item in tests[0].criteria)


def test_unsequenced_parenthesized_text_remains_unknown() -> None:
    tests = MarkingSchemeParser().parse(
        (
            SourceLine(page_number=1, text="Test 06"),
            SourceLine(page_number=1, text="Any suitable answer (1 mark)"),
        )
    )

    assert len(tests[0].criteria) == 1
    assert tests[0].criteria[0].criterion_type is CriterionType.UNKNOWN


def test_explicit_allocations_are_summed_when_every_criterion_has_marks() -> None:
    tests = MarkingSchemeParser().parse(
        (
            SourceLine(page_number=1, text="Test 05"),
            SourceLine(page_number=1, text="1. first (1 mark)"),
            SourceLine(page_number=1, text="2. second (2 marks)"),
        )
    )

    assert tests[0].maximum_marks == 3


def test_unknown_criterion_is_preserved_with_warning() -> None:
    tests = MarkingSchemeParser().parse(
        (
            SourceLine(page_number=1, text="Test 06"),
            SourceLine(page_number=1, text="Any suitable answer"),
        )
    )

    assert tests[0].criteria[0].criterion_type is CriterionType.UNKNOWN
    assert tests[0].criteria[0].description == "Any suitable answer"
    assert MarkingSchemeWarning.UNKNOWN_CRITERION in tests[0].warnings


def test_missing_maximum_marks_is_explicit(tmp_path: Path) -> None:
    scheme = _ingest(tmp_path, {1: "Test 07\nAny suitable answer"})

    assert scheme.tests[0].maximum_marks is None
    assert scheme.total_available_marks is None
    assert MarkingSchemeWarning.TEST_MAXIMUM_UNKNOWN in scheme.tests[0].warnings
    assert MarkingSchemeWarning.TOTAL_MAXIMUM_INCOMPLETE in scheme.warnings


def test_parser_rule_provenance_is_explicit() -> None:
    tests = MarkingSchemeParser().parse(
        (
            SourceLine(page_number=2, text="Test 08"),
            SourceLine(page_number=2, text="Keywords: river / water (2 marks)"),
        )
    )

    criterion = tests[0].criteria[0]
    assert criterion.provenance.parser_rule is CriterionParserRule.KEYWORD_LABEL
    assert criterion.provenance.source.page_number == 2


def test_exact_test_number_alignment() -> None:
    scheme = MarkingSchemeParser().parse((SourceLine(page_number=1, text="Test 01"),))
    structured = _ingest_for_alignment(scheme)
    student = _student_paper({1: AnswerExtractionStatus.EMPTY})

    result = TestAlignmentService().align(student, structured)

    assert result.tests[0].status is AlignmentStatus.ALIGNED
    assert result.tests[0].student_answer is student.answers[0]
    assert result.tests[0].marking_scheme_test is structured.tests[0]


def test_student_test_missing_is_explicit() -> None:
    structured = _ingest_for_alignment(
        MarkingSchemeParser().parse((SourceLine(page_number=1, text="Test 02"),))
    )

    result = TestAlignmentService().align(_student_paper({}), structured)

    assert result.student_missing_test_numbers == (2,)


def test_scheme_test_missing_is_explicit() -> None:
    structured = _ingest_for_alignment(
        MarkingSchemeParser().parse((SourceLine(page_number=1, text="Test 01"),))
    )

    result = TestAlignmentService().align(
        _student_paper(
            {1: AnswerExtractionStatus.EMPTY, 2: AnswerExtractionStatus.EMPTY}
        ),
        structured,
    )

    assert result.scheme_missing_test_numbers == (2,)


def test_review_required_answer_still_aligns_structurally() -> None:
    structured = _ingest_for_alignment(
        MarkingSchemeParser().parse((SourceLine(page_number=1, text="Test 01"),))
    )

    result = TestAlignmentService().align(
        _student_paper({1: AnswerExtractionStatus.REVIEW_REQUIRED}), structured
    )

    assert result.tests[0].status is AlignmentStatus.STUDENT_REVIEW_REQUIRED
    assert result.tests[0].marking_scheme_test is not None


def test_ocr_failed_answer_still_aligns_structurally() -> None:
    structured = _ingest_for_alignment(
        MarkingSchemeParser().parse((SourceLine(page_number=1, text="Test 01"),))
    )

    result = TestAlignmentService().align(
        _student_paper({1: AnswerExtractionStatus.OCR_FAILED}), structured
    )

    assert result.tests[0].status is AlignmentStatus.STUDENT_OCR_FAILED


def test_serialization_is_deterministic_and_private(tmp_path: Path) -> None:
    scheme = _ingest(tmp_path, {1: "Test 01\n1. answer (1 mark)"})
    aligned = TestAlignmentService().align(
        _student_paper({1: AnswerExtractionStatus.EMPTY}), scheme
    )
    private_root = tmp_path / "private"

    assert marking_scheme_json(scheme) == marking_scheme_json(scheme)
    assert aligned_paper_json(aligned) == aligned_paper_json(aligned)
    target = write_private_json(
        aligned,
        private_root / "alignment.json",
        private_root=private_root,
    )
    assert target.is_file()
    with pytest.raises(MarkingSchemeIngestionError, match="remain private"):
        write_private_json(
            scheme,
            tmp_path / "public.json",
            private_root=private_root,
        )


def test_malformed_scheme_without_test_marker_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(MarkingSchemeIngestionError, match="reliable Test markers"):
        _ingest(tmp_path, {1: "answers without a reliable heading"})


def test_one_page_ocr_failure_is_isolated(tmp_path: Path) -> None:
    scheme = _ingest(
        tmp_path,
        {1: RuntimeError("private failure"), 2: "Test 01\n1. answer (1 mark)"},
    )

    assert scheme.provenance.failed_ocr_pages == (1,)
    assert scheme.provenance.successful_ocr_pages == (2,)
    assert MarkingSchemeWarning.OCR_PAGE_FAILED in scheme.warnings
    assert "private failure" not in marking_scheme_json(scheme)


def test_source_region_accepts_optional_geometry() -> None:
    source = SchemeSourceRegion(page_number=1, raw_source_text="synthetic")

    assert source.bbox is None


def _ingest_for_alignment(
    tests: tuple[MarkingSchemeTest, ...],
) -> StructuredMarkingScheme:
    from app.marking_scheme.models import (
        MarkingSchemeOCRStatus,
        MarkingSchemePageOCR,
        MarkingSchemeProvenance,
    )

    known = tuple(item.maximum_marks for item in tests)
    total = (
        sum((item for item in known if item is not None), Decimal(0))
        if all(item is not None for item in known)
        else None
    )
    return StructuredMarkingScheme(
        scheme_id=uuid4(),
        source_document_id=uuid4(),
        page_count=1,
        tests=tests,
        total_available_marks=total,
        page_ocr=(
            MarkingSchemePageOCR(
                page_number=1,
                status=MarkingSchemeOCRStatus.EXTRACTED,
                provider="fake",
                duration_ms=1,
            ),
        ),
        provenance=MarkingSchemeProvenance(
            ingestion_version="test-v1",
            parser_version="test-parser-v1",
            source_sha256="a" * 64,
            ocr_provider="fake",
            successful_ocr_pages=(1,),
        ),
        processing_duration_ms=1,
    )

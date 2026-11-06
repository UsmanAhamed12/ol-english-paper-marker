"""Synthetic tests for the Phase 4D structured extraction boundary."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

import cv2
import numpy as np

from app.answers.extraction import StudentRegionCropper
from app.answers.models import (
    AnswerExtractionStatus,
    AnswerExtractionWarning,
    AnswerFragmentStatus,
    AnswerSourceContext,
)
from app.answers.serialization import structured_paper_json, write_structured_paper
from app.answers.service import StudentExtractionService
from app.domain.models.paper import PaperPage
from app.evidence.models import (
    AnswerRegionSignal,
    EvidenceRegion,
    EvidenceType,
    StudentAnswerRegion,
)
from app.evidence.safety_models import (
    CandidateDisposition,
    CandidateSafetyDecision,
    SafeTestExtraction,
    SafetyReason,
)
from app.evidence.safety_models import (
    TestExtractionStatus as SafetyExtractionStatus,
)
from app.ocr.models import BoundingBox, OCRExtraction
from app.ocr.normalizer import OCRNormalizer
from app.ocr.service import OCRService
from app.structure.models import (
    ExamPageStructure,
    ExamStructure,
    MarkerDetectionStrategy,
)
from app.structure.models import (
    TestMarker as StructureTestMarker,
)
from app.structure.models import (
    TestPageRegion as StructureTestPageRegion,
)
from app.structure.models import (
    TestRegion as StructureTestRegion,
)
from tests.unit.evidence.helpers import features


@dataclass
class QueueOCRProvider:
    """Offline provider returning or failing in deterministic call order."""

    outputs: list[str | Exception]
    name: str = "fake-region-ocr"
    model_version: str | None = "fake-v1"
    calls: list[Path] = field(default_factory=list)

    def extract_page(self, page: PaperPage) -> OCRExtraction:
        self.calls.append(page.image_path)
        output = self.outputs[len(self.calls) - 1]
        if isinstance(output, Exception):
            raise output
        return OCRExtraction(
            raw_text=output,
            confidence=0.7,
            processing_duration_ms=2,
        )


def _page(tmp_path: Path, paper_id: UUID, page_number: int) -> PaperPage:
    path = (tmp_path / f"page_{page_number:04d}.png").resolve()
    image = np.full((600, 800, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        f"synthetic-{page_number}",
        (80, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )
    assert cv2.imwrite(str(path), image)
    return PaperPage(
        paper_id=paper_id,
        page_number=page_number,
        image_path=path,
        width=800,
        height=600,
    )


def _marker(test_number: int, page_number: int) -> StructureTestMarker:
    return StructureTestMarker(
        test_number=test_number,
        raw_text=f"Test {test_number:02d}",
        page_number=page_number,
        bbox=BoundingBox(x=20, y=20, width=100, height=30),
        confidence=0.9,
        text_similarity=1.0,
        numeric_confidence=1.0,
        ocr_confidence=0.9,
        geometry_confidence=0.8,
        strategy=MarkerDetectionStrategy.EXACT_TOKENS,
        source_word_indices=(0, 1),
        sequence_confidence=0.9,
    )


def _test_region(
    test_number: int,
    page_numbers: tuple[int, ...],
) -> StructureTestRegion:
    marker = _marker(test_number, page_numbers[0])
    return StructureTestRegion(
        test_number=test_number,
        label=marker.label,
        marker=marker,
        page_regions=tuple(
            StructureTestPageRegion(
                page_number=page_number,
                bbox=BoundingBox(x=0, y=0, width=800, height=600),
            )
            for page_number in page_numbers
        ),
        start_page=page_numbers[0],
        end_page=page_numbers[-1],
    )


def _structure(
    paper_id: UUID,
    pages: tuple[PaperPage, ...],
    tests: tuple[StructureTestRegion, ...],
    *,
    missing: tuple[int, ...] = (),
) -> ExamStructure:
    return ExamStructure(
        paper_id=paper_id,
        page_count=len(pages),
        pages=tuple(
            ExamPageStructure(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                regions=tuple(
                    region
                    for test in tests
                    for region in test.page_regions
                    if region.page_number == page.page_number
                ),
            )
            for page in pages
        ),
        tests=tests,
        missing_test_numbers=missing,
    )


def _evidence(
    page: PaperPage,
    test_number: int,
    bbox: BoundingBox,
    *,
    evidence_type: EvidenceType = EvidenceType.STUDENT_CANDIDATE,
) -> EvidenceRegion:
    return EvidenceRegion(
        paper_id=page.paper_id,
        page_number=page.page_number,
        test_number=test_number,
        bbox=bbox,
        evidence_type=evidence_type,
        confidence=0.7,
        signals=(),
        features=features(confidence=0.5, irregularity=0.7),
        source_word_indices=(0,),
        source_image_path=page.image_path,
        classification_strategy="synthetic-v1",
    )


def _safety(
    page: PaperPage,
    test_number: int,
    *,
    safe_boxes: tuple[BoundingBox, ...] = (),
    review_box: BoundingBox | None = None,
    review_boxes: tuple[BoundingBox, ...] = (),
    rejected_box: BoundingBox | None = None,
) -> SafeTestExtraction:
    decisions: list[CandidateSafetyDecision] = []
    answers: list[StudentAnswerRegion] = []
    for index, bbox in enumerate(safe_boxes):
        evidence = _evidence(page, test_number, bbox)
        decisions.append(
            CandidateSafetyDecision(
                evidence_index=index,
                disposition=CandidateDisposition.SAFE_STUDENT,
                reasons=(SafetyReason.STUDENT_SIGNALS_PRESERVED,),
                evidence=evidence,
            )
        )
        answers.append(
            StudentAnswerRegion(
                paper_id=page.paper_id,
                page_number=page.page_number,
                test_number=test_number,
                bbox=bbox,
                confidence=0.75,
                signals=(AnswerRegionSignal.STUDENT_EVIDENCE_CLUSTER,),
                source_evidence_indices=(index,),
                source_image_path=page.image_path,
                detection_strategy="safe-student-region-v1",
            )
        )
    all_review_boxes = (*review_boxes, *((review_box,) if review_box else ()))
    for bbox in all_review_boxes:
        index = len(decisions)
        decisions.append(
            CandidateSafetyDecision(
                evidence_index=index,
                disposition=CandidateDisposition.REVIEW_REQUIRED,
                reasons=(SafetyReason.UNCERTAIN_CLASSIFICATION,),
                evidence=_evidence(
                    page,
                    test_number,
                    bbox,
                    evidence_type=EvidenceType.UNKNOWN,
                ),
            )
        )
    if rejected_box is not None:
        index = len(decisions)
        decisions.append(
            CandidateSafetyDecision(
                evidence_index=index,
                disposition=CandidateDisposition.REJECTED_NON_STUDENT,
                reasons=(SafetyReason.PRINTED_GUIDE_OR_RULE,),
                evidence=_evidence(page, test_number, rejected_box),
            )
        )
    status = (
        SafetyExtractionStatus.REVIEW_REQUIRED
        if all_review_boxes
        else SafetyExtractionStatus.SAFE
        if safe_boxes
        else SafetyExtractionStatus.EMPTY
    )
    return SafeTestExtraction(
        paper_id=page.paper_id,
        page_number=page.page_number,
        test_number=test_number,
        region_bbox=BoundingBox(x=0, y=0, width=800, height=600),
        status=status,
        decisions=tuple(decisions),
        answer_regions=tuple(answers),
        policy_version="student-candidate-safety-v1",
    )


def _service(
    tmp_path: Path, outputs: list[str | Exception]
) -> tuple[StudentExtractionService, QueueOCRProvider]:
    provider = QueueOCRProvider(outputs)
    service = StudentExtractionService(
        OCRService(provider, OCRNormalizer()),
        StudentRegionCropper(tmp_path / "runtime"),
    )
    return service, provider


def test_single_short_answer_is_extracted_without_correction(tmp_path: Path) -> None:
    paper_id = uuid4()
    page = _page(tmp_path, paper_id, 1)
    test = _test_region(1, (1,))
    service, _ = _service(tmp_path, ["goverment"])

    result = service.extract_paper(
        _structure(paper_id, (page,), (test,)),
        (page,),
        (
            _safety(
                page, 1, safe_boxes=(BoundingBox(x=100, y=150, width=90, height=40),)
            ),
        ),
    )

    assert result.answers[0].status is AnswerExtractionStatus.EXTRACTED
    assert result.answers[0].text == "goverment"
    assert result.answers[0].ocr_fragments[0].raw_text == "goverment"


def test_historical_marked_source_requires_review_without_discarding_text(
    tmp_path: Path,
) -> None:
    paper_id = uuid4()
    page = _page(tmp_path, paper_id, 1)
    test = _test_region(1, (1,))
    service, _ = _service(tmp_path, ["observed text"])

    answer = service.extract_paper(
        _structure(paper_id, (page,), (test,)),
        (page,),
        (
            _safety(
                page,
                1,
                safe_boxes=(BoundingBox(x=100, y=150, width=90, height=40),),
            ),
        ),
        source_context=AnswerSourceContext.HISTORICAL_MARKED_EVALUATION,
    ).answers[0]

    assert answer.status is AnswerExtractionStatus.REVIEW_REQUIRED
    assert answer.text == "observed text"
    assert AnswerExtractionWarning.HISTORICAL_TEACHER_ANNOTATION_RISK in answer.warnings
    assert (
        answer.provenance.source_context
        is AnswerSourceContext.HISTORICAL_MARKED_EVALUATION
    )


def test_multiple_fragments_use_top_to_bottom_then_left_to_right(
    tmp_path: Path,
) -> None:
    paper_id = uuid4()
    page = _page(tmp_path, paper_id, 1)
    test = _test_region(1, (1,))
    service, _ = _service(tmp_path, ["first", "second", "third"])
    boxes = (
        BoundingBox(x=300, y=250, width=80, height=30),
        BoundingBox(x=220, y=120, width=80, height=30),
        BoundingBox(x=100, y=120, width=80, height=30),
    )

    answer = service.extract_paper(
        _structure(paper_id, (page,), (test,)),
        (page,),
        (_safety(page, 1, safe_boxes=boxes),),
    ).answers[0]

    assert [(item.bbox.y, item.bbox.x) for item in answer.source_regions] == [
        (120, 100),
        (120, 220),
        (250, 300),
    ]
    assert answer.text == "first\nsecond\nthird"


def test_paragraph_fragments_preserve_line_structure(tmp_path: Path) -> None:
    paper_id = uuid4()
    page = _page(tmp_path, paper_id, 1)
    service, _ = _service(tmp_path, ["He go home", "yesterday."])
    answer = service.extract_paper(
        _structure(paper_id, (page,), (_test_region(12, (1,)),)),
        (page,),
        (
            _safety(
                page,
                12,
                safe_boxes=(
                    BoundingBox(x=80, y=180, width=500, height=50),
                    BoundingBox(x=80, y=250, width=500, height=50),
                ),
            ),
        ),
    ).answers[0]

    assert answer.text == "He go home\nyesterday."


def test_cross_page_ordering_is_deterministic(tmp_path: Path) -> None:
    paper_id = uuid4()
    first_page = _page(tmp_path, paper_id, 1)
    second_page = _page(tmp_path, paper_id, 2)
    test = _test_region(16, (1, 2))
    service, _ = _service(tmp_path, ["page one", "page two"])
    answer = service.extract_paper(
        _structure(paper_id, (first_page, second_page), (test,)),
        (first_page, second_page),
        (
            _safety(
                second_page,
                16,
                safe_boxes=(BoundingBox(x=80, y=80, width=400, height=80),),
            ),
            _safety(
                first_page,
                16,
                safe_boxes=(BoundingBox(x=80, y=400, width=400, height=80),),
            ),
        ),
    ).answers[0]

    assert answer.pages == (1, 2)
    assert [item.page_number for item in answer.ocr_fragments] == [1, 2]
    assert answer.text == "page one\npage two"


def test_empty_answer_skips_ocr(tmp_path: Path) -> None:
    paper_id = uuid4()
    page = _page(tmp_path, paper_id, 1)
    service, provider = _service(tmp_path, [])
    answer = service.extract_paper(
        _structure(paper_id, (page,), (_test_region(2, (1,)),)),
        (page,),
        (_safety(page, 2),),
    ).answers[0]

    assert answer.status is AnswerExtractionStatus.EMPTY
    assert answer.text == ""
    assert answer.source_regions == ()
    assert provider.calls == []


def test_review_evidence_is_preserved_but_not_ocrd(tmp_path: Path) -> None:
    paper_id = uuid4()
    page = _page(tmp_path, paper_id, 1)
    service, provider = _service(tmp_path, [])
    answer = service.extract_paper(
        _structure(paper_id, (page,), (_test_region(3, (1,)),)),
        (page,),
        (_safety(page, 3, review_box=BoundingBox(x=700, y=100, width=30, height=30)),),
    ).answers[0]

    assert answer.status is AnswerExtractionStatus.REVIEW_REQUIRED
    assert answer.text == ""
    assert len(answer.review_regions) == 1
    assert AnswerExtractionWarning.REVIEW_EVIDENCE_PRESENT in answer.warnings
    assert provider.calls == []


def test_many_uncertain_candidates_are_summarized_per_test_page(
    tmp_path: Path,
) -> None:
    page = _page(tmp_path, uuid4(), 1)
    service, _ = _service(tmp_path, [])

    answer = service.extract_paper(
        _structure(page.paper_id, (page,), (_test_region(3, (1,)),)),
        (page,),
        (
            _safety(
                page,
                3,
                review_boxes=(
                    BoundingBox(x=100, y=100, width=20, height=20),
                    BoundingBox(x=500, y=400, width=25, height=18),
                ),
            ),
        ),
    ).answers[0]

    assert answer.status is AnswerExtractionStatus.REVIEW_REQUIRED
    assert len(answer.review_regions) == 1
    assert answer.review_regions[0].bbox == BoundingBox(x=0, y=0, width=800, height=600)
    assert answer.review_regions[0].reasons == (SafetyReason.UNCERTAIN_CLASSIFICATION,)


def test_rejected_evidence_never_enters_answer(tmp_path: Path) -> None:
    paper_id = uuid4()
    page = _page(tmp_path, paper_id, 1)
    service, provider = _service(tmp_path, [])
    answer = service.extract_paper(
        _structure(paper_id, (page,), (_test_region(4, (1,)),)),
        (page,),
        (_safety(page, 4, rejected_box=BoundingBox(x=80, y=200, width=500, height=8)),),
    ).answers[0]

    assert answer.status is AnswerExtractionStatus.EMPTY
    assert answer.source_regions == ()
    assert answer.review_regions == ()
    assert provider.calls == []


def test_ocr_failure_isolated_from_other_test(tmp_path: Path) -> None:
    paper_id = uuid4()
    page = _page(tmp_path, paper_id, 1)
    first = _test_region(1, (1,))
    second = _test_region(2, (1,))
    service, _ = _service(tmp_path, [RuntimeError("private failure"), "usable"])
    paper = service.extract_paper(
        _structure(paper_id, (page,), (first, second)),
        (page,),
        (
            _safety(
                page, 1, safe_boxes=(BoundingBox(x=80, y=100, width=120, height=40),)
            ),
            _safety(
                page, 2, safe_boxes=(BoundingBox(x=80, y=300, width=120, height=40),)
            ),
        ),
    )

    assert paper.answers[0].status is AnswerExtractionStatus.OCR_FAILED
    assert paper.answers[0].ocr_fragments[0].status is AnswerFragmentStatus.OCR_FAILED
    assert paper.answers[0].text == ""
    assert paper.answers[1].status is AnswerExtractionStatus.EXTRACTED
    assert paper.answers[1].text == "usable"


def test_missing_test_is_explicit_without_fabricated_answer(tmp_path: Path) -> None:
    paper_id = uuid4()
    page = _page(tmp_path, paper_id, 1)
    test = _test_region(1, (1,))
    service, _ = _service(tmp_path, [])
    paper = service.extract_paper(
        _structure(paper_id, (page,), (test,), missing=(2,)),
        (page,),
        (_safety(page, 1),),
    )

    assert paper.detected_test_numbers == (1,)
    assert paper.missing_test_numbers == (2,)
    assert [item.test_number for item in paper.answers] == [1]


def test_serialization_is_deterministic_and_private(tmp_path: Path) -> None:
    paper_id = uuid4()
    page = _page(tmp_path, paper_id, 1)
    service, _ = _service(tmp_path, [])
    paper = service.extract_paper(
        _structure(paper_id, (page,), (_test_region(1, (1,)),)),
        (page,),
        (_safety(page, 1),),
    )
    private_root = (tmp_path / "private").resolve()
    target = private_root / "paper.json"

    first = structured_paper_json(paper)
    second = structured_paper_json(paper)
    write_structured_paper(paper, target, private_root=private_root)

    assert first == second
    assert target.read_text(encoding="utf-8").rstrip("\n") == first


def test_marker_and_region_provenance_are_preserved(tmp_path: Path) -> None:
    paper_id = uuid4()
    page = _page(tmp_path, paper_id, 1)
    test = _test_region(7, (1,))
    service, _ = _service(tmp_path, ["x"])
    answer = service.extract_paper(
        _structure(paper_id, (page,), (test,)),
        (page,),
        (
            _safety(
                page, 7, safe_boxes=(BoundingBox(x=100, y=200, width=30, height=30),)
            ),
        ),
    ).answers[0]

    assert answer.provenance.marker_bbox == test.marker.bbox
    assert answer.provenance.marker_strategy is MarkerDetectionStrategy.EXACT_TOKENS
    assert answer.provenance.marker_source_word_indices == (0, 1)
    assert answer.source_regions[0].source_evidence_indices == (0,)


def test_canonical_image_is_immutable_and_crop_is_derived(tmp_path: Path) -> None:
    paper_id = uuid4()
    page = _page(tmp_path, paper_id, 1)
    before = hashlib.sha256(page.image_path.read_bytes()).hexdigest()
    service, _ = _service(tmp_path, ["answer"])
    answer = service.extract_paper(
        _structure(paper_id, (page,), (_test_region(1, (1,)),)),
        (page,),
        (
            _safety(
                page, 1, safe_boxes=(BoundingBox(x=100, y=150, width=100, height=50),)
            ),
        ),
    ).answers[0]

    assert hashlib.sha256(page.image_path.read_bytes()).hexdigest() == before
    assert answer.source_regions[0].canonical_image_sha256 == before
    assert answer.source_regions[0].derived_image_path != page.image_path
    assert answer.source_regions[0].derived_image_path.is_file()

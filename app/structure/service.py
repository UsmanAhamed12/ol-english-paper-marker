"""Application service for document-level exam structure detection."""

from __future__ import annotations

from collections import Counter

from app.domain.models.paper import PaperPage
from app.ocr.models import OCRPageResult
from app.structure.marker_detector import detect_marker_candidates, select_markers
from app.structure.models import ExamPageStructure, ExamStructure
from app.structure.segmenter import segment_test_regions


class ExamStructureDetector:
    """Convert ordered structured OCR evidence into spatial Test regions."""

    def __init__(
        self,
        *,
        expected_test_numbers: tuple[int, ...] = tuple(range(1, 17)),
        minimum_marker_confidence: float = 0.60,
    ) -> None:
        if not expected_test_numbers:
            raise ValueError("Expected Test numbers must not be empty")
        if tuple(sorted(set(expected_test_numbers))) != expected_test_numbers:
            raise ValueError("Expected Test numbers must be unique and increasing")
        if not 0.0 <= minimum_marker_confidence <= 1.0:
            raise ValueError("Marker confidence threshold must be in [0, 1]")
        self._expected = expected_test_numbers
        self._minimum_confidence = minimum_marker_confidence

    def detect(
        self,
        pages: tuple[PaperPage, ...],
        ocr_results: tuple[OCRPageResult, ...],
    ) -> ExamStructure:
        """Detect candidates, select an ordered sequence, and segment regions."""

        if not pages or len(pages) != len(ocr_results):
            raise ValueError("Structure detection requires OCR for every page")
        if [page.page_number for page in pages] != list(range(1, len(pages) + 1)):
            raise ValueError("Structure pages must be ordered from page 1")
        paper_id = pages[0].paper_id
        if any(page.paper_id != paper_id for page in pages):
            raise ValueError("Structure pages must belong to one paper")

        page_candidates = tuple(
            detect_marker_candidates(page, result)
            for page, result in zip(pages, ocr_results, strict=True)
        )
        all_candidates = tuple(
            candidate for candidates in page_candidates for candidate in candidates
        )
        markers, rejected, duplicates = select_markers(
            all_candidates,
            expected_test_numbers=self._expected,
            minimum_confidence=self._minimum_confidence,
        )
        test_regions = segment_test_regions(pages, markers)
        detected = {marker.test_number for marker in markers}
        missing = tuple(number for number in self._expected if number not in detected)

        page_models = tuple(
            ExamPageStructure(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                candidates=page_candidates[index],
                markers=tuple(
                    marker
                    for marker in markers
                    if marker.page_number == page.page_number
                ),
                regions=tuple(
                    page_region
                    for test in test_regions
                    for page_region in test.page_regions
                    if page_region.page_number == page.page_number
                ),
            )
            for index, page in enumerate(pages)
        )
        candidate_counts = Counter(
            candidate.test_number for candidate in all_candidates
        )
        all_duplicates = tuple(
            sorted(
                set(duplicates)
                | {number for number, count in candidate_counts.items() if count > 1}
            )
        )
        return ExamStructure(
            paper_id=paper_id,
            page_count=len(pages),
            pages=page_models,
            tests=test_regions,
            missing_test_numbers=missing,
            duplicate_test_numbers=all_duplicates,
            rejected_candidates=rejected,
        )

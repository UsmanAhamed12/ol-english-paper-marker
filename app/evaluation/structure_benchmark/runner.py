"""Deterministic spatial metrics for detected exam Test markers."""

from __future__ import annotations

from collections import Counter
from statistics import fmean

from app.evaluation.structure_benchmark.models import (
    StructureBenchmarkPaper,
    StructureBenchmarkResult,
    StructureBenchmarkSummary,
    StructureGroundTruthMarker,
)
from app.structure.models import ExamStructure, TestMarker


def evaluate_structure(
    ground_truth: StructureBenchmarkPaper,
    detected: ExamStructure,
) -> StructureBenchmarkResult:
    """Measure exact marker identity and spatially aligned number accuracy."""

    expected = ground_truth.expected_markers
    markers = tuple(region.marker for region in detected.tests)
    exact_matches = _greedy_matches(expected, markers, require_number=True)
    location_matches = _greedy_matches(expected, markers, require_number=False)
    true_positives = len(exact_matches)
    false_positives = len(markers) - true_positives
    false_negatives = len(expected) - true_positives
    precision = _ratio(true_positives, len(markers), empty_value=1.0)
    recall = _ratio(true_positives, len(expected), empty_value=1.0)
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    number_correct = sum(
        expected[expected_index].test_number == markers[marker_index].test_number
        for expected_index, marker_index in location_matches
    )
    number_accuracy = _ratio(
        number_correct,
        len(location_matches),
        empty_value=1.0 if not expected and not markers else 0.0,
    )
    expected_sequence = tuple(marker.test_number for marker in expected)
    detected_sequence = tuple(marker.test_number for marker in markers)
    ordering_accuracy = _ordering_accuracy(expected_sequence, detected_sequence)
    expected_numbers = {marker.test_number for marker in expected}
    detected_numbers = {marker.test_number for marker in markers}
    candidate_counts = Counter(
        candidate.test_number
        for page in detected.pages
        for candidate in page.candidates
    )
    return StructureBenchmarkResult(
        paper_alias=ground_truth.paper_alias,
        page_count=detected.page_count,
        expected_markers=len(expected),
        detected_markers=len(markers),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        duplicate_markers=sum(max(0, count - 1) for count in candidate_counts.values()),
        precision=precision,
        recall=recall,
        f1=f1,
        test_number_accuracy=number_accuracy,
        ordering_accuracy=ordering_accuracy,
        missing_test_numbers=tuple(sorted(expected_numbers - detected_numbers)),
    )


def summarize_structure_results(
    results: tuple[StructureBenchmarkResult, ...],
) -> StructureBenchmarkSummary:
    """Micro-average marker metrics and mean paper-level accuracies."""

    if not results:
        raise ValueError("Structure benchmark requires at least one result")
    true_positives = sum(result.true_positives for result in results)
    detected = sum(result.detected_markers for result in results)
    expected = sum(result.expected_markers for result in results)
    precision = _ratio(true_positives, detected, empty_value=1.0)
    recall = _ratio(true_positives, expected, empty_value=1.0)
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return StructureBenchmarkSummary(
        paper_count=len(results),
        page_count=sum(result.page_count for result in results),
        expected_markers=expected,
        detected_markers=detected,
        true_positives=true_positives,
        false_positives=sum(result.false_positives for result in results),
        false_negatives=sum(result.false_negatives for result in results),
        duplicate_markers=sum(result.duplicate_markers for result in results),
        precision=precision,
        recall=recall,
        f1=f1,
        mean_test_number_accuracy=fmean(
            result.test_number_accuracy for result in results
        ),
        mean_ordering_accuracy=fmean(result.ordering_accuracy for result in results),
    )


def _greedy_matches(
    expected: tuple[StructureGroundTruthMarker, ...],
    detected: tuple[TestMarker, ...],
    *,
    require_number: bool,
) -> tuple[tuple[int, int], ...]:
    matches: list[tuple[int, int]] = []
    used_detected: set[int] = set()
    for expected_index, expected_marker in enumerate(expected):
        options = [
            (abs(_center_y(expected_marker) - _center_y(marker)), marker_index)
            for marker_index, marker in enumerate(detected)
            if marker_index not in used_detected
            and marker.page_number == expected_marker.page_number
            and (
                not require_number or marker.test_number == expected_marker.test_number
            )
            and _vertically_near(expected_marker, marker)
        ]
        if not options:
            continue
        _, marker_index = min(options)
        used_detected.add(marker_index)
        matches.append((expected_index, marker_index))
    return tuple(matches)


def _vertically_near(
    expected: StructureGroundTruthMarker, detected: TestMarker
) -> bool:
    tolerance = max(80, expected.bbox.height * 3)
    return abs(_center_y(expected) - _center_y(detected)) <= tolerance


def _center_y(marker: StructureGroundTruthMarker | TestMarker) -> float:
    return marker.bbox.y + marker.bbox.height / 2


def _ratio(numerator: int, denominator: int, *, empty_value: float) -> float:
    return empty_value if denominator == 0 else numerator / denominator


def _ordering_accuracy(expected: tuple[int, ...], detected: tuple[int, ...]) -> float:
    if not expected and not detected:
        return 1.0
    denominator = max(len(expected), len(detected))
    if denominator == 0:
        return 0.0
    return _longest_common_subsequence(expected, detected) / denominator


def _longest_common_subsequence(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    previous = [0] * (len(right) + 1)
    for left_value in left:
        current = [0]
        for right_index, right_value in enumerate(right, start=1):
            if left_value == right_value:
                current.append(previous[right_index - 1] + 1)
            else:
                current.append(max(current[-1], previous[right_index]))
        previous = current
    return previous[-1]

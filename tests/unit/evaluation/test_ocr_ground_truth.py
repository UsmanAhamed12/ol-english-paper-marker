"""Synthetic-only tests for freezing human OCR ground truth."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.exceptions import OCRBenchmarkPreparationError
from app.evaluation.ocr_benchmark.ground_truth import (
    freeze_ground_truth,
    ground_truth_fingerprint,
    parse_transcription_worksheet,
)
from app.evaluation.ocr_benchmark.models import (
    BenchmarkDifficulty,
    BenchmarkManifest,
    GroundTruthStatus,
    OCRBenchmarkSample,
)


def _sample(index: int) -> OCRBenchmarkSample:
    return OCRBenchmarkSample(
        sample_id=f"synthetic-sample-{index:02d}",
        paper_alias="synthetic-paper-a",
        page_number=index,
        image_path=Path(f"private/page_{index:04d}.png"),
        image_width=100,
        image_height=200,
        difficulty=BenchmarkDifficulty.MEDIUM,
        categories=("synthetic-category",),
        printed_content_present=True,
        teacher_annotations_present=False,
        ground_truth_status=GroundTruthStatus.PENDING,
        ground_truth_student_text=None,
    )


def _manifest(count: int = 2) -> BenchmarkManifest:
    return BenchmarkManifest(samples=tuple(_sample(i) for i in range(1, count + 1)))


def _worksheet(values: tuple[str, ...]) -> str:
    sections = ["# Synthetic private worksheet", ""]
    for index, value in enumerate(values, start=1):
        sections.extend(
            [
                f"## Sample {index:03d}",
                f"Sample ID: synthetic-sample-{index:02d}",
                "Ground truth student text:",
                value,
                "",
            ]
        )
    return "\n".join(sections)


def test_transfers_worksheet_text_to_matching_manifest_samples() -> None:
    frozen = freeze_ground_truth(
        _manifest(),
        _worksheet(("synthetic first", "synthetic second")),
        verified_empty_sample_ids=frozenset(),
    )

    assert [sample.ground_truth_student_text for sample in frozen.samples] == [
        "synthetic first",
        "synthetic second",
    ]
    assert frozen.is_ready


def test_preserves_spelling_capitalization_and_newlines_exactly() -> None:
    human_text = "He go to scool\nYestarday"

    frozen = freeze_ground_truth(
        _manifest(count=1),
        _worksheet((human_text,)),
        verified_empty_sample_ids=frozenset(),
    )

    assert frozen.samples[0].ground_truth_student_text == human_text


def test_pending_blank_is_rejected_without_explicit_empty_decision() -> None:
    with pytest.raises(OCRBenchmarkPreparationError, match="blank"):
        freeze_ground_truth(
            _manifest(count=1),
            _worksheet(("",)),
            verified_empty_sample_ids=frozenset(),
        )


def test_explicit_verified_empty_is_accepted() -> None:
    frozen = freeze_ground_truth(
        _manifest(count=1),
        _worksheet(("empty",)),
        verified_empty_sample_ids={"synthetic-sample-01"},
    )

    sample = frozen.samples[0]
    assert sample.ground_truth_status is GroundTruthStatus.VERIFIED_EMPTY
    assert sample.ground_truth_student_text == ""
    assert sample.is_ready


def test_verified_empty_rejects_unexpected_nonempty_worksheet_text() -> None:
    with pytest.raises(OCRBenchmarkPreparationError, match="unexpected"):
        freeze_ground_truth(
            _manifest(count=1),
            _worksheet(("synthetic answer",)),
            verified_empty_sample_ids={"synthetic-sample-01"},
        )


def test_fingerprint_is_deterministic_and_changes_with_ground_truth() -> None:
    first = freeze_ground_truth(
        _manifest(count=1),
        _worksheet(("synthetic answer",)),
        verified_empty_sample_ids=frozenset(),
    )
    same = freeze_ground_truth(
        _manifest(count=1),
        _worksheet(("synthetic answer",)),
        verified_empty_sample_ids=frozenset(),
    )
    changed = freeze_ground_truth(
        _manifest(count=1),
        _worksheet(("synthetic answar",)),
        verified_empty_sample_ids=frozenset(),
    )

    assert ground_truth_fingerprint(first) == ground_truth_fingerprint(same)
    assert ground_truth_fingerprint(first) != ground_truth_fingerprint(changed)


def test_fingerprint_rejects_pending_manifest() -> None:
    with pytest.raises(OCRBenchmarkPreparationError, match="ready"):
        ground_truth_fingerprint(_manifest(count=1))


def test_all_eight_samples_must_be_explicitly_verified() -> None:
    values = tuple(f"synthetic {index}" for index in range(1, 8)) + ("",)

    with pytest.raises(OCRBenchmarkPreparationError, match="blank"):
        freeze_ground_truth(
            _manifest(count=8),
            _worksheet(values),
            verified_empty_sample_ids=frozenset(),
        )

    ready = freeze_ground_truth(
        _manifest(count=8),
        _worksheet(values),
        verified_empty_sample_ids={"synthetic-sample-08"},
    )
    assert len(ready.samples) == 8
    assert ready.is_ready


def test_model_status_invariants_are_explicit() -> None:
    data = _sample(1).model_dump()

    with pytest.raises(ValidationError, match="non-empty"):
        OCRBenchmarkSample.model_validate(
            {
                **data,
                "ground_truth_status": GroundTruthStatus.VERIFIED,
                "ground_truth_student_text": "",
            }
        )
    with pytest.raises(ValidationError, match="empty string"):
        OCRBenchmarkSample.model_validate(
            {
                **data,
                "ground_truth_status": GroundTruthStatus.VERIFIED_EMPTY,
                "ground_truth_student_text": None,
            }
        )


def test_parser_rejects_mismatched_worksheet_ids() -> None:
    parsed = parse_transcription_worksheet(_worksheet(("synthetic",)))
    assert set(parsed) == {"synthetic-sample-01"}

    with pytest.raises(OCRBenchmarkPreparationError, match="exactly match"):
        freeze_ground_truth(
            _manifest(count=2),
            _worksheet(("synthetic",)),
            verified_empty_sample_ids=frozenset(),
        )

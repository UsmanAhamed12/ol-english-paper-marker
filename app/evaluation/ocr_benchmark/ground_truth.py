"""Freeze authoritative human OCR ground truth without rewriting it."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Set

from app.core.exceptions import OCRBenchmarkPreparationError
from app.evaluation.ocr_benchmark.models import (
    BenchmarkManifest,
    GroundTruthStatus,
    OCRBenchmarkSample,
)

_SECTION_PATTERN = re.compile(
    r"(?ms)^## Sample \d{3}\n(?P<body>.*?)(?=^## Sample \d{3}\n|\Z)"
)
_SAMPLE_ID_PATTERN = re.compile(r"(?m)^Sample ID: (?P<sample_id>[a-z0-9_-]+)$")
_GROUND_TRUTH_MARKER = "Ground truth student text:\n"
_FINGERPRINT_VERSION = "ocr-ground-truth-v1"


def parse_transcription_worksheet(worksheet: str) -> dict[str, str]:
    """Extract exact human text while discarding Markdown section separators."""

    transcriptions: dict[str, str] = {}
    sections = list(_SECTION_PATTERN.finditer(worksheet))
    if not sections:
        raise OCRBenchmarkPreparationError("Worksheet contains no benchmark samples")

    for section in sections:
        body = section.group("body")
        sample_id_match = _SAMPLE_ID_PATTERN.search(body)
        if sample_id_match is None or _GROUND_TRUTH_MARKER not in body:
            raise OCRBenchmarkPreparationError("Worksheet sample structure is invalid")
        sample_id = sample_id_match.group("sample_id")
        if sample_id in transcriptions:
            raise OCRBenchmarkPreparationError("Worksheet sample IDs must be unique")
        entered_text = body.split(_GROUND_TRUTH_MARKER, maxsplit=1)[1]
        transcriptions[sample_id] = entered_text.rstrip("\n")
    return transcriptions


def freeze_ground_truth(
    manifest: BenchmarkManifest,
    worksheet: str,
    *,
    verified_empty_sample_ids: Set[str],
    notes_by_sample_id: Mapping[str, str] | None = None,
) -> BenchmarkManifest:
    """Transfer human text exactly and require explicit status for every sample."""

    transcriptions = parse_transcription_worksheet(worksheet)
    manifest_ids = {sample.sample_id for sample in manifest.samples}
    if transcriptions.keys() != manifest_ids:
        raise OCRBenchmarkPreparationError(
            "Worksheet samples do not exactly match the benchmark manifest"
        )
    if not verified_empty_sample_ids <= manifest_ids:
        raise OCRBenchmarkPreparationError(
            "Verified-empty sample IDs must exist in the benchmark manifest"
        )

    notes = notes_by_sample_id or {}
    if not notes.keys() <= manifest_ids:
        raise OCRBenchmarkPreparationError(
            "Ground-truth notes must reference benchmark samples"
        )

    frozen_samples: list[OCRBenchmarkSample] = []
    for sample in manifest.samples:
        entered_text = transcriptions[sample.sample_id]
        if sample.sample_id in verified_empty_sample_ids:
            if entered_text.strip() and entered_text.strip().lower() not in {
                "empty",
                "none",
                "no student-answer text",
                "no student answer text",
            }:
                raise OCRBenchmarkPreparationError(
                    "Verified-empty worksheet field contains unexpected text"
                )
            status = GroundTruthStatus.VERIFIED_EMPTY
            ground_truth = ""
        else:
            if not entered_text.strip():
                raise OCRBenchmarkPreparationError(
                    "Pending blank transcription cannot be frozen"
                )
            status = GroundTruthStatus.VERIFIED
            ground_truth = entered_text

        sample_data = sample.model_dump()
        sample_data.update(
            ground_truth_status=status,
            ground_truth_student_text=ground_truth,
        )
        if sample.sample_id in notes:
            sample_data["notes"] = _append_note(sample.notes, notes[sample.sample_id])
        frozen_samples.append(OCRBenchmarkSample.model_validate(sample_data))

    frozen = BenchmarkManifest(
        schema_version=manifest.schema_version,
        samples=tuple(frozen_samples),
    )
    if not frozen.is_ready:
        raise OCRBenchmarkPreparationError(
            "Every benchmark sample must be explicitly human verified"
        )
    return frozen


def ground_truth_fingerprint(manifest: BenchmarkManifest) -> str:
    """Hash a canonical representation of verified ground truth."""

    if not manifest.is_ready:
        raise OCRBenchmarkPreparationError(
            "Ground-truth fingerprint requires a ready benchmark"
        )
    payload = {
        "fingerprint_version": _FINGERPRINT_VERSION,
        "samples": [
            {
                "sample_id": sample.sample_id,
                "status": sample.ground_truth_status.value,
                "student_text": sample.ground_truth_student_text,
            }
            for sample in sorted(manifest.samples, key=lambda item: item.sample_id)
        ],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _append_note(existing: str, note: str) -> str:
    if not note.strip():
        raise OCRBenchmarkPreparationError("Ground-truth note must not be blank")
    if note in existing.splitlines():
        return existing
    return f"{existing}\n{note}" if existing else note

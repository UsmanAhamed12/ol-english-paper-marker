"""Synthetic tests for private upload materialization and cleanup."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest
from pytest import MonkeyPatch

from app.core.exceptions import OrchestrationError
from app.ui import runtime
from app.ui.runtime import PrivateUploadStore, validate_pdf_upload

RUN_ID = UUID("72000000-0000-0000-0000-000000000007")


@dataclass(frozen=True)
class FakeUpload:
    name: str
    type: str
    content: bytes

    @property
    def size(self) -> int:
        return len(self.content)

    def getvalue(self) -> bytes:
        return self.content


def _upload(
    name: str = "private-name.pdf", content: bytes = b"%PDF-1.7\nX"
) -> FakeUpload:
    return FakeUpload(name=name, type="application/pdf", content=content)


def test_uploads_use_safe_names_and_are_removed_after_processing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data" / "runtime" / "streamlit-uploads"
    student = _upload("student-identity.pdf", b"%PDF-1.7\nstudent")
    scheme = _upload("official-scheme.pdf", b"%PDF-1.7\nscheme")
    student_hash = hashlib.sha256(student.content).hexdigest()
    store = PrivateUploadStore(root, maximum_bytes=1024)
    materialized = store.materialize(student, scheme, run_id=RUN_ID)
    assert materialized.run_dir.name == RUN_ID.hex
    assert materialized.student_pdf.name == "student-paper.pdf"
    assert materialized.marking_scheme_pdf.name == "marking-scheme.pdf"
    assert "identity" not in str(materialized.student_pdf)
    assert hashlib.sha256(student.content).hexdigest() == student_hash
    assert materialized.student_pdf.read_bytes() == student.content
    store.cleanup(materialized.run_dir)
    assert not materialized.run_dir.exists()


def test_materialization_refuses_to_overwrite_existing_run(tmp_path: Path) -> None:
    store = PrivateUploadStore(tmp_path / "runtime", maximum_bytes=1024)
    store.materialize(_upload(), _upload("scheme.pdf"), run_id=RUN_ID)
    with pytest.raises(FileExistsError):
        store.materialize(_upload(), _upload("scheme.pdf"), run_id=RUN_ID)


@pytest.mark.parametrize(
    ("upload", "message"),
    [
        (_upload("answer.txt"), "must be a PDF"),
        (
            FakeUpload("answer.pdf", "text/plain", b"%PDF-1.7\nX"),
            "PDF content type",
        ),
        (_upload(content=b"not a pdf"), "not a valid PDF"),
        (_upload(content=b"%PDF-1.7\n" + b"x" * 20), "invalid file size"),
    ],
)
def test_invalid_uploads_fail_with_safe_messages(
    upload: FakeUpload, message: str
) -> None:
    with pytest.raises(OrchestrationError, match=message):
        validate_pdf_upload(upload, role="Student paper", maximum_bytes=20)


def test_cleanup_never_touches_paths_outside_private_root(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    outside = tmp_path / "outside"
    outside.mkdir()
    evidence = outside / "student-paper.pdf"
    evidence.write_bytes(b"immutable")
    PrivateUploadStore(root, maximum_bytes=1024).cleanup(outside)
    assert evidence.read_bytes() == b"immutable"


def test_partial_materialization_is_cleaned_after_write_failure(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    root = tmp_path / "runtime"
    original = runtime._atomic_write
    call_count = 0

    def fail_second_write(path: Path, content: bytes) -> Path:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("synthetic write failure")
        return original(path, content)

    monkeypatch.setattr(runtime, "_atomic_write", fail_second_write)
    store = PrivateUploadStore(root, maximum_bytes=1024)

    with pytest.raises(OSError, match="synthetic write failure"):
        store.materialize(_upload(), _upload("scheme.pdf"), run_id=RUN_ID)

    assert not (root / RUN_ID.hex).exists()

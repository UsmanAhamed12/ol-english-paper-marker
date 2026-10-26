"""Private upload materialization and UI workflow-state helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.core.exceptions import OrchestrationError


class UploadedPDF(Protocol):
    """Small boundary implemented by Streamlit's UploadedFile."""

    @property
    def name(self) -> str: ...

    @property
    def type(self) -> str: ...

    @property
    def size(self) -> int: ...

    def getvalue(self) -> bytes: ...


class MaterializedUploads(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_dir: Path
    student_pdf: Path
    marking_scheme_pdf: Path
    input_signature: str


class PrivateUploadStore:
    """Write uploads under one ignored run directory using safe names."""

    def __init__(self, root: Path, *, maximum_bytes: int) -> None:
        self._root = root.expanduser().resolve()
        self._maximum_bytes = maximum_bytes

    def materialize(
        self,
        student: UploadedPDF,
        scheme: UploadedPDF,
        *,
        run_id: UUID,
    ) -> MaterializedUploads:
        student_bytes = validate_pdf_upload(
            student, role="Student paper", maximum_bytes=self._maximum_bytes
        )
        scheme_bytes = validate_pdf_upload(
            scheme, role="Marking scheme", maximum_bytes=self._maximum_bytes
        )
        run_dir = (self._root / run_id.hex).resolve()
        if not run_dir.is_relative_to(self._root):
            raise OrchestrationError("Upload directory must remain private")
        run_dir.mkdir(parents=True, exist_ok=False)
        try:
            student_path = _atomic_write(run_dir / "student-paper.pdf", student_bytes)
            scheme_path = _atomic_write(run_dir / "marking-scheme.pdf", scheme_bytes)
        except OSError:
            self.cleanup(run_dir)
            raise
        signature = hashlib.sha256(student_bytes + b"\0" + scheme_bytes).hexdigest()
        return MaterializedUploads(
            run_dir=run_dir,
            student_pdf=student_path,
            marking_scheme_pdf=scheme_path,
            input_signature=signature,
        )

    def cleanup(self, run_dir: Path | None) -> None:
        """Delete only known UI-upload files and their private run directory."""

        if run_dir is None:
            return
        resolved = run_dir.expanduser().resolve()
        if resolved.parent != self._root or not resolved.is_dir():
            return
        for name in ("student-paper.pdf", "marking-scheme.pdf"):
            candidate = resolved / name
            if candidate.is_file():
                candidate.unlink()
        try:
            resolved.rmdir()
        except OSError:
            return


def validate_pdf_upload(upload: UploadedPDF, *, role: str, maximum_bytes: int) -> bytes:
    """Apply cheap safe checks before the production PDF validator runs."""

    if Path(upload.name).suffix.lower() != ".pdf":
        raise OrchestrationError(f"{role} must be a PDF file")
    if upload.type not in {"application/pdf", "application/octet-stream"}:
        raise OrchestrationError(f"{role} must use a PDF content type")
    if upload.size <= 0 or upload.size > maximum_bytes:
        raise OrchestrationError(f"{role} has an invalid file size")
    content = upload.getvalue()
    if len(content) != upload.size or not content.startswith(b"%PDF-"):
        raise OrchestrationError(f"{role} is not a valid PDF upload")
    return content


def _atomic_write(path: Path, content: bytes) -> Path:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return path

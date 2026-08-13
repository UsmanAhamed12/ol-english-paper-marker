"""Offline tests for typed Tesseract OCR evidence."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from uuid import uuid4

import pytesseract  # type: ignore[import-untyped]
import pytest
from pydantic import ValidationError

from app.core.exceptions import OCRProviderError
from app.domain.models.paper import PaperPage
from app.ocr.base import OCRProvider
from app.ocr.models import BoundingBox, OCRWord
from app.ocr.normalizer import OCRNormalizer
from app.ocr.providers.tesseract import (
    PyTesseractClient,
    TesseractClient,
    TesseractData,
    TesseractOCRProvider,
    normalize_tesseract_confidence,
    parse_tesseract_words,
)
from app.ocr.service import OCRService


class FakeClient:
    def __init__(
        self,
        data: TesseractData | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.data = data or _data()
        self.failure = failure
        self.calls: list[dict[str, object]] = []

    def version(self) -> str:
        return "5.5.2"

    def image_to_data(
        self,
        image_path: str,
        *,
        language: str,
        config: str,
        timeout_seconds: float,
    ) -> TesseractData:
        self.calls.append(
            {
                "image_path": image_path,
                "language": language,
                "config": config,
                "timeout_seconds": timeout_seconds,
            }
        )
        if self.failure is not None:
            raise self.failure
        return self.data


def _data() -> TesseractData:
    return {
        "text": ["", "goverment", "works"],
        "conf": ["-1", "85.5", "70"],
        "left": [0, 10, 80],
        "top": [0, 20, 20],
        "width": [1, 60, 40],
        "height": [1, 15, 15],
        "block_num": [0, 1, 1],
        "par_num": [0, 1, 1],
        "line_num": [0, 1, 1],
        "word_num": [0, 1, 2],
    }


def _page(tmp_path: Path) -> PaperPage:
    path = (tmp_path / "page.png").resolve()
    path.write_bytes(b"synthetic image")
    return PaperPage(
        paper_id=uuid4(),
        page_number=1,
        image_path=path,
        width=100,
        height=100,
    )


def test_bounding_box_and_word_validation() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(x=-1, y=0, width=1, height=1)
    with pytest.raises(ValidationError):
        OCRWord(text=" ", bbox=BoundingBox(x=0, y=0, width=1, height=1))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("85.5", 0.855), (0, 0.0), (100, 1.0), (-1, None), (101, None)],
)
def test_tesseract_confidence_conversion(
    raw: str | int,
    expected: float | None,
) -> None:
    assert normalize_tesseract_confidence(raw) == expected


def test_word_parsing_ignores_empty_rows_and_preserves_hierarchy() -> None:
    words = parse_tesseract_words(_data())

    assert [word.text for word in words] == ["goverment", "works"]
    assert words[0].confidence == 0.855
    assert words[0].bbox == BoundingBox(x=10, y=20, width=60, height=15)
    assert words[0].block_number == 1
    assert words[0].word_number == 1


def test_malformed_column_lengths_are_rejected() -> None:
    data = dict(_data())
    data["conf"] = ["1"]

    with pytest.raises(ValueError, match="inconsistent"):
        parse_tesseract_words(data)


def test_provider_preserves_layout_evidence_and_service_compatibility(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    provider = TesseractOCRProvider(
        client=cast(TesseractClient, client),
        language="eng",
        psm=6,
        timeout_seconds=120,
        model_version="5.5.2",
    )

    result = OCRService(provider, OCRNormalizer()).process_page(_page(tmp_path))

    assert result.raw_text == "goverment works"
    assert result.normalized_text == "goverment works"
    assert result.evidence is not None
    assert len(result.evidence.words) == 2
    assert result.confidence == pytest.approx(0.7775)
    assert result.provider == "tesseract"
    assert result.model_version == "5.5.2"
    assert isinstance(provider, OCRProvider)
    assert client.calls[0]["config"] == "--psm 6"
    assert client.calls[0]["timeout_seconds"] == 120


def test_provider_wraps_malformed_data(tmp_path: Path) -> None:
    client = FakeClient(data={"text": ["x"]})
    provider = TesseractOCRProvider(
        client=cast(TesseractClient, client),
        language="eng",
        psm=6,
        timeout_seconds=120,
    )

    with pytest.raises(OCRProviderError, match="malformed"):
        provider.extract_page(_page(tmp_path))


def test_provider_preserves_safe_tesseract_failure(tmp_path: Path) -> None:
    client = FakeClient(
        failure=OCRProviderError("Local Tesseract OCR request timed out")
    )
    provider = TesseractOCRProvider(
        client=cast(TesseractClient, client),
        language="eng",
        psm=6,
        timeout_seconds=120,
    )

    with pytest.raises(OCRProviderError, match="timed out"):
        provider.extract_page(_page(tmp_path))


def test_invalid_image_path_is_rejected(tmp_path: Path) -> None:
    page = _page(tmp_path)
    page.image_path.unlink()
    provider = TesseractOCRProvider(
        client=cast(TesseractClient, FakeClient()),
        language="eng",
        psm=6,
        timeout_seconds=120,
    )

    with pytest.raises(OCRProviderError, match="unavailable"):
        provider.extract_page(page)


def test_missing_tesseract_binary_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> object:
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr("pytesseract.get_tesseract_version", missing)

    with pytest.raises(OCRProviderError, match="executable is unavailable"):
        PyTesseractClient().version()


def test_pytesseract_timeout_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("Tesseract process timeout")

    monkeypatch.setattr("pytesseract.image_to_data", timeout)

    with pytest.raises(OCRProviderError, match="timed out"):
        PyTesseractClient().image_to_data(
            "synthetic.png",
            language="eng",
            config="--psm 6",
            timeout_seconds=1,
        )

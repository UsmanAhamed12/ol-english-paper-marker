"""Plain Tesseract OCR provider preserving word-level layout evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import fmean
from time import perf_counter
from typing import Protocol

import pytesseract  # type: ignore[import-untyped]
from pytesseract import Output

from app.core.exceptions import ConfigurationError, OCRProviderError
from app.domain.models.paper import PaperPage
from app.ocr.layout import reconstruct_layout_text
from app.ocr.models import (
    BoundingBox,
    OCRExtraction,
    OCRStructuredEvidence,
    OCRWord,
)

TesseractValue = str | int | float
TesseractData = Mapping[str, Sequence[TesseractValue]]

_REQUIRED_COLUMNS = (
    "text",
    "conf",
    "left",
    "top",
    "width",
    "height",
    "block_num",
    "par_num",
    "line_num",
    "word_num",
)


class TesseractClient(Protocol):
    """Narrow pytesseract boundary used by the provider and offline tests."""

    def version(self) -> str:
        """Return the installed engine version."""

    def image_to_data(
        self,
        image_path: str,
        *,
        language: str,
        config: str,
        timeout_seconds: float,
    ) -> TesseractData:
        """Return one Tesseract TSV result as columns."""


class PyTesseractClient:
    """Production adapter around the small pytesseract API surface in use."""

    def version(self) -> str:
        """Resolve the local Tesseract executable version."""

        try:
            return str(pytesseract.get_tesseract_version())
        except pytesseract.TesseractNotFoundError as error:
            raise OCRProviderError(
                "Local Tesseract executable is unavailable"
            ) from error

    def image_to_data(
        self,
        image_path: str,
        *,
        language: str,
        config: str,
        timeout_seconds: float,
    ) -> TesseractData:
        """Run non-preprocessed OCR against one canonical PNG path."""

        try:
            result = pytesseract.image_to_data(
                image_path,
                lang=language,
                config=config,
                output_type=Output.DICT,
                timeout=timeout_seconds,
            )
        except pytesseract.TesseractNotFoundError as error:
            raise OCRProviderError(
                "Local Tesseract executable is unavailable"
            ) from error
        except RuntimeError as error:
            if "timeout" in str(error).casefold():
                raise OCRProviderError(
                    "Local Tesseract OCR request timed out"
                ) from error
            raise OCRProviderError("Local Tesseract OCR request failed") from error
        except pytesseract.TesseractError as error:
            raise OCRProviderError("Local Tesseract OCR request failed") from error
        if not isinstance(result, Mapping):
            raise OCRProviderError("Local Tesseract returned malformed OCR data")
        return result


class TesseractOCRProvider:
    """Extract layout-preserving OCR evidence from canonical page images."""

    def __init__(
        self,
        *,
        client: TesseractClient,
        language: str,
        psm: int,
        timeout_seconds: float,
        model_version: str | None = None,
    ) -> None:
        if not language.strip():
            raise ConfigurationError("Tesseract language must not be blank")
        if not 0 <= psm <= 13:
            raise ConfigurationError("Tesseract PSM must be between 0 and 13")
        if timeout_seconds <= 0:
            raise ConfigurationError("Tesseract timeout must be positive")
        self._client = client
        self._language = language
        self._psm = psm
        self._timeout_seconds = timeout_seconds
        self._model_version = model_version

    @classmethod
    def from_system(
        cls,
        *,
        language: str,
        psm: int,
        timeout_seconds: float,
    ) -> TesseractOCRProvider:
        """Construct a provider whose version comes from the installed binary."""

        client = PyTesseractClient()
        return cls(
            client=client,
            language=language,
            psm=psm,
            timeout_seconds=timeout_seconds,
            model_version=client.version(),
        )

    @property
    def name(self) -> str:
        """Return stable provider provenance."""

        return "tesseract"

    @property
    def model_version(self) -> str | None:
        """Return the installed engine version when it was resolved."""

        return self._model_version

    def extract_page(self, page: PaperPage) -> OCRExtraction:
        """Run plain OCR and retain every non-empty valid word record."""

        if not page.image_path.is_file():
            raise OCRProviderError("OCR source image is unavailable")
        started = perf_counter()
        try:
            data = self._client.image_to_data(
                str(page.image_path),
                language=self._language,
                config=f"--psm {self._psm}",
                timeout_seconds=self._timeout_seconds,
            )
            words = parse_tesseract_words(data)
        except OCRProviderError:
            raise
        except (TypeError, ValueError, IndexError, KeyError) as error:
            raise OCRProviderError(
                "Local Tesseract returned malformed OCR data"
            ) from error

        layout_text = reconstruct_layout_text(words)
        confidences = [word.confidence for word in words if word.confidence is not None]
        return OCRExtraction(
            raw_text=layout_text,
            confidence=fmean(confidences) if confidences else None,
            processing_duration_ms=(perf_counter() - started) * 1000,
            evidence=OCRStructuredEvidence(words=words, layout_text=layout_text),
        )


def parse_tesseract_words(data: TesseractData) -> tuple[OCRWord, ...]:
    """Validate aligned Tesseract columns and construct typed word evidence."""

    if any(column not in data for column in _REQUIRED_COLUMNS):
        raise ValueError("missing Tesseract data column")
    lengths = {len(data[column]) for column in _REQUIRED_COLUMNS}
    if len(lengths) != 1:
        raise ValueError("Tesseract data columns have inconsistent lengths")

    words: list[OCRWord] = []
    row_count = lengths.pop()
    for index in range(row_count):
        text = str(data["text"][index])
        if not text.strip():
            continue
        words.append(
            OCRWord(
                text=text,
                confidence=normalize_tesseract_confidence(data["conf"][index]),
                bbox=BoundingBox(
                    x=_integer(data["left"][index]),
                    y=_integer(data["top"][index]),
                    width=_integer(data["width"][index]),
                    height=_integer(data["height"][index]),
                ),
                block_number=_positive_or_none(data["block_num"][index]),
                paragraph_number=_positive_or_none(data["par_num"][index]),
                line_number=_positive_or_none(data["line_num"][index]),
                word_number=_positive_or_none(data["word_num"][index]),
            )
        )
    return tuple(words)


def normalize_tesseract_confidence(value: TesseractValue) -> float | None:
    """Convert valid 0-100 Tesseract confidence to the generic unit interval."""

    confidence = float(value)
    if confidence < 0 or confidence > 100:
        return None
    return confidence / 100


def _integer(value: TesseractValue) -> int:
    number = float(value)
    if not number.is_integer():
        raise ValueError("Tesseract coordinate must be an integer")
    return int(number)


def _positive_or_none(value: TesseractValue) -> int | None:
    number = _integer(value)
    return number if number > 0 else None

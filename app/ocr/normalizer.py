"""Conservative deterministic normalization of raw OCR text."""

from __future__ import annotations

import unicodedata


class OCRNormalizer:
    """Normalize representation without correcting student language."""

    def normalize(self, raw_text: str) -> str:
        """Return NFC text with stable newlines and trailing space removed."""

        normalized = unicodedata.normalize("NFC", raw_text)
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        lines = (line.rstrip(" \t") for line in normalized.split("\n"))
        return "\n".join(lines).strip("\n")

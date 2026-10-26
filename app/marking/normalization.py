"""Conservative, comparison-only normalization for deterministic marking."""

from __future__ import annotations

import re
import unicodedata

MARKING_NORMALIZATION_VERSION = "marking-normalization-v1"
_SURROUNDING_PUNCTUATION = " \t\r\n.,;:!?\"'()[]{}"
_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


def normalize_for_marking(text: str, *, case_sensitive: bool = False) -> str:
    """Normalize a comparison copy without changing stored student evidence."""

    normalized = unicodedata.normalize(
        "NFC", text.replace("\r\n", "\n").replace("\r", "\n")
    )
    normalized = " ".join(normalized.split()).strip(_SURROUNDING_PUNCTUATION)
    return normalized if case_sensitive else normalized.casefold()


def tokenize_for_marking(text: str) -> tuple[str, ...]:
    """Return normalized word tokens for boundary-aware keyword matching."""

    return tuple(_TOKEN_PATTERN.findall(normalize_for_marking(text)))


def contains_keyword_or_phrase(student_text: str, expected: str) -> bool:
    """Match a whole token or contiguous token phrase, never a substring."""

    student_tokens = tokenize_for_marking(student_text)
    expected_tokens = tokenize_for_marking(expected)
    if not expected_tokens or len(expected_tokens) > len(student_tokens):
        return False
    width = len(expected_tokens)
    return any(
        student_tokens[index : index + width] == expected_tokens
        for index in range(len(student_tokens) - width + 1)
    )

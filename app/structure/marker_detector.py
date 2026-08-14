"""Conservative multi-signal detection of spatial Test headings."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import fmean

from app.domain.models.paper import PaperPage
from app.ocr.models import BoundingBox, OCRPageResult, OCRWord
from app.structure.models import (
    MarkerDetectionStrategy,
    TestMarker,
    TestMarkerCandidate,
)

_KEYWORD = "test"
_KNOWN_KEYWORD_CONFUSIONS = {"lest", "tesl", "tost"}
_NUMBER_CONFUSIONS = str.maketrans({"o": "0", "O": "0", "l": "1", "I": "1"})


@dataclass(frozen=True, slots=True)
class _IndexedWord:
    index: int
    word: OCRWord


def detect_marker_candidates(
    page: PaperPage,
    result: OCRPageResult,
) -> tuple[TestMarkerCandidate, ...]:
    """Detect plausible line-leading Test labels while preserving evidence."""

    if result.page_number != page.page_number or result.paper_id != page.paper_id:
        raise ValueError("OCR result does not belong to the supplied page")
    if result.evidence is None:
        return ()

    candidates: list[TestMarkerCandidate] = []
    for line in _group_lines(result.evidence.words):
        candidate = _candidate_from_line(page, line)
        if candidate is not None:
            candidates.append(candidate)
    return tuple(sorted(candidates, key=_candidate_position))


def select_markers(
    candidates: tuple[TestMarkerCandidate, ...],
    *,
    expected_test_numbers: tuple[int, ...],
    minimum_confidence: float = 0.62,
) -> tuple[tuple[TestMarker, ...], tuple[TestMarkerCandidate, ...], tuple[int, ...]]:
    """Select the strongest increasing sequence without inventing missing Tests."""

    expected = set(expected_test_numbers)
    eligible = tuple(
        candidate
        for candidate in sorted(candidates, key=_candidate_position)
        if candidate.test_number in expected
        and candidate.confidence >= minimum_confidence
    )
    duplicates = tuple(
        sorted(
            number
            for number, count in Counter(
                candidate.test_number for candidate in eligible
            ).items()
            if count > 1
        )
    )
    if not eligible:
        return (), candidates, duplicates

    scores: list[float] = []
    chains: list[tuple[int, ...]] = []
    for current_index, current in enumerate(eligible):
        best_score = current.confidence + (
            0.1 if current.test_number == min(expected) else 0.0
        )
        best_chain: tuple[int, ...] = (current_index,)
        for previous_index in range(current_index):
            previous = eligible[previous_index]
            if previous.test_number >= current.test_number:
                continue
            support = _sequence_support(previous.test_number, current.test_number)
            score = scores[previous_index] + current.confidence + 0.25 * support
            candidate_chain = chains[previous_index] + (current_index,)
            if score > best_score or (
                score == best_score and candidate_chain < best_chain
            ):
                best_score = score
                best_chain = candidate_chain
        scores.append(best_score)
        chains.append(best_chain)

    selected_indices = set(chains[max(range(len(scores)), key=scores.__getitem__)])
    selected_candidates = tuple(
        candidate
        for index, candidate in enumerate(eligible)
        if index in selected_indices
    )
    markers = tuple(
        _accepted_marker(selected_candidates, index)
        for index in range(len(selected_candidates))
    )
    selected_ids = {id(candidate) for candidate in selected_candidates}
    rejected = tuple(
        candidate for candidate in candidates if id(candidate) not in selected_ids
    )
    return markers, rejected, duplicates


def _candidate_from_line(
    page: PaperPage,
    line: tuple[_IndexedWord, ...],
) -> TestMarkerCandidate | None:
    if not line:
        return None
    marker_start = next(
        (
            index
            for index, item in enumerate(line[:2])
            if any(character.isalnum() for character in item.word.text)
        ),
        None,
    )
    if marker_start is None:
        return None
    first = line[marker_start]
    compact = _split_compact(first.word.text)
    keyword: str
    number_text: str | None
    marker_words: tuple[_IndexedWord, ...]
    if compact is not None:
        keyword, number_text = compact
        marker_words = (first,)
        strategy = MarkerDetectionStrategy.COMPACT_TOKEN
    else:
        keyword = _letters(first.word.text)
        number_word = _find_number_word(line, marker_start)
        if number_word is None:
            return None
        number_text = _number_token(number_word.word.text)
        marker_words = (first, number_word)
        strategy = MarkerDetectionStrategy.EXACT_TOKENS

    similarity = _keyword_similarity(keyword)
    if similarity < 0.75 or number_text is None:
        return None
    parsed = _parse_number(number_text)
    if parsed is None:
        return None
    test_number, numeric_confidence = parsed
    if keyword != _KEYWORD:
        strategy = (
            MarkerDetectionStrategy.OCR_CONFUSION
            if keyword in _KNOWN_KEYWORD_CONFUSIONS
            else MarkerDetectionStrategy.FUZZY_KEYWORD
        )
    elif numeric_confidence < 1.0:
        strategy = MarkerDetectionStrategy.OCR_CONFUSION

    ocr_values = [
        item.word.confidence
        for item in marker_words
        if item.word.confidence is not None
    ]
    ocr_confidence = fmean(ocr_values) if ocr_values else None
    geometry_confidence = _geometry_confidence(page, line, marker_words)
    confidence = (
        0.35 * similarity
        + 0.2 * numeric_confidence
        + 0.2 * (ocr_confidence if ocr_confidence is not None else 0.5)
        + 0.25 * geometry_confidence
    )
    if len(line) > marker_start + 4:
        confidence *= 0.65
    if marker_words[0].word.bbox.y >= page.height * 0.92:
        confidence *= 0.55
    return TestMarkerCandidate(
        test_number=test_number,
        raw_text=" ".join(item.word.text for item in marker_words),
        page_number=page.page_number,
        bbox=_union_boxes(tuple(item.word.bbox for item in marker_words)),
        confidence=min(1.0, confidence),
        text_similarity=similarity,
        numeric_confidence=numeric_confidence,
        ocr_confidence=ocr_confidence,
        geometry_confidence=geometry_confidence,
        strategy=strategy,
        source_word_indices=tuple(item.index for item in marker_words),
    )


def _group_lines(words: tuple[OCRWord, ...]) -> tuple[tuple[_IndexedWord, ...], ...]:
    grouped: dict[tuple[int, int, int], list[_IndexedWord]] = defaultdict(list)
    for index, word in enumerate(words):
        key = (
            word.block_number if word.block_number is not None else word.bbox.y + 1,
            word.paragraph_number if word.paragraph_number is not None else 1,
            word.line_number if word.line_number is not None else word.bbox.y + 1,
        )
        grouped[key].append(_IndexedWord(index=index, word=word))
    lines = [
        tuple(
            sorted(
                line,
                key=lambda item: (
                    item.word.word_number
                    if item.word.word_number is not None
                    else 2**31,
                    item.word.bbox.x,
                    item.index,
                ),
            )
        )
        for line in grouped.values()
    ]
    return tuple(
        sorted(
            lines,
            key=lambda line: (
                min(item.word.bbox.y for item in line),
                min(item.word.bbox.x for item in line),
            ),
        )
    )


def _split_compact(value: str) -> tuple[str, str] | None:
    cleaned = "".join(character for character in value if character.isalnum())
    split_at = next(
        (index for index, character in enumerate(cleaned) if not character.isalpha()),
        None,
    )
    if split_at is None:
        for index, character in enumerate(cleaned):
            if character in "01" and index >= 3:
                split_at = index
                break
    if split_at is None:
        return None
    keyword = cleaned[:split_at].casefold()
    number = cleaned[split_at:]
    return (keyword, number) if keyword and number else None


def _letters(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalpha())


def _number_token(value: str) -> str | None:
    cleaned = "".join(character for character in value if character.isalnum())
    return cleaned if 1 <= len(cleaned) <= 2 else None


def _find_number_word(
    line: tuple[_IndexedWord, ...], marker_start: int
) -> _IndexedWord | None:
    """Find a nearby number while allowing isolated dash/punctuation tokens."""

    for item in line[marker_start + 1 : marker_start + 4]:
        if not any(character.isalnum() for character in item.word.text):
            continue
        return item
    return None


def _parse_number(value: str) -> tuple[int, float] | None:
    normalized = value.translate(_NUMBER_CONFUSIONS)
    if not normalized.isdigit() or len(normalized) > 2:
        return None
    number = int(normalized)
    if number < 1 or number > 99:
        return None
    return number, 1.0 if value.isdigit() else 0.8


def _keyword_similarity(value: str) -> float:
    if not value:
        return 0.0
    distance = _edit_distance(value, _KEYWORD)
    return max(0.0, 1.0 - distance / max(len(value), len(_KEYWORD)))


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _geometry_confidence(
    page: PaperPage,
    line: tuple[_IndexedWord, ...],
    marker_words: tuple[_IndexedWord, ...],
) -> float:
    box = _union_boxes(tuple(item.word.bbox for item in marker_words))
    compactness = 1.0 if len(line) <= 3 else 0.45
    horizontal = 1.0 if box.x <= page.width * 0.7 else 0.5
    vertical = 0.1 if box.y >= page.height * 0.92 else 1.0
    return fmean((compactness, horizontal, vertical))


def _union_boxes(boxes: tuple[BoundingBox, ...]) -> BoundingBox:
    left = min(box.x for box in boxes)
    top = min(box.y for box in boxes)
    right = max(box.x + box.width for box in boxes)
    bottom = max(box.y + box.height for box in boxes)
    return BoundingBox(x=left, y=top, width=right - left, height=bottom - top)


def _candidate_position(candidate: TestMarkerCandidate) -> tuple[int, int, int, int]:
    return (
        candidate.page_number,
        candidate.bbox.y,
        candidate.bbox.x,
        candidate.test_number,
    )


def _sequence_support(previous: int, current: int) -> float:
    gap = current - previous
    if gap == 1:
        return 1.0
    if gap == 2:
        return 0.7
    return 0.4


def _accepted_marker(
    candidates: tuple[TestMarkerCandidate, ...],
    index: int,
) -> TestMarker:
    previous = candidates[index - 1].test_number if index > 0 else None
    following = (
        candidates[index + 1].test_number if index + 1 < len(candidates) else None
    )
    supports = []
    if previous is not None:
        supports.append(_sequence_support(previous, candidates[index].test_number))
    if following is not None:
        supports.append(_sequence_support(candidates[index].test_number, following))
    sequence_confidence = fmean(supports) if supports else 0.5
    data = candidates[index].model_dump()
    data["confidence"] = min(
        1.0, 0.85 * candidates[index].confidence + 0.15 * sequence_confidence
    )
    return TestMarker(
        **data,
        sequence_confidence=sequence_confidence,
    )

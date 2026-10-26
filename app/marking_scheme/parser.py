"""Conservative deterministic parser for printed marking-scheme OCR."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal

from app.core.exceptions import MarkingSchemeParseError
from app.marking_scheme.models import (
    CriterionParserRule,
    CriterionProvenance,
    CriterionType,
    MarkingCriterion,
    MarkingSchemeTest,
    MarkingSchemeWarning,
    SchemeSourceRegion,
)
from app.ocr.models import BoundingBox, OCRWord

PARSER_VERSION = "marking-scheme-parser-v1"
_TEST_MARKER = re.compile(
    r"^\s*test\s*0*([1-9]\d?)(?=$|\s|[:.\-])(?:\s*[:.\-]\s*|\s+)?(.*)$",
    re.IGNORECASE,
)
_TOTAL_MARKS = re.compile(
    r"^\s*(?:total|maximum)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\s*(?:marks?)?\s*$",
    re.IGNORECASE,
)
_MARK_ALLOCATION = re.compile(
    r"(?:\(|\[)?\s*(\d+(?:\.\d+)?)\s*marks?\s*(?:\)|\])?",
    re.IGNORECASE,
)
_ITEM_PREFIX = re.compile(r"^\s*(?:(\d+|[A-Za-z])[.)]|[-*•])\s+")
_LABELED = re.compile(
    r"^\s*(accepted answers?|keywords?|content|language|format|holistic)\s*:\s*(.+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SourceLine:
    """One page-local OCR line with optional word geometry."""

    page_number: int
    text: str
    bbox: BoundingBox | None = None
    source_word_indices: tuple[int, ...] = ()


class MarkingSchemeParser:
    """Turn ordered OCR lines into explicit Test and criterion models."""

    def parse(self, lines: tuple[SourceLine, ...]) -> tuple[MarkingSchemeTest, ...]:
        segments: list[tuple[int, str | None, list[SourceLine]]] = []
        current: tuple[int, str | None, list[SourceLine]] | None = None
        seen: set[int] = set()
        for line in lines:
            marker = _TEST_MARKER.fullmatch(line.text.strip())
            if marker:
                number = int(marker.group(1))
                if number in seen:
                    raise MarkingSchemeParseError(
                        "Duplicate marking-scheme Test marker"
                    )
                if seen and number <= max(seen):
                    raise MarkingSchemeParseError(
                        "Scheme Test markers are out of order"
                    )
                if current is not None:
                    segments.append(current)
                title = marker.group(2).strip(" :-") or None
                current = (number, title, [line])
                seen.add(number)
            elif current is not None:
                current[2].append(line)
        if current is not None:
            segments.append(current)
        if not segments:
            raise MarkingSchemeParseError("No reliable Test markers were detected")
        return tuple(self._parse_test(*segment) for segment in segments)

    def _parse_test(
        self, test_number: int, title: str | None, lines: list[SourceLine]
    ) -> MarkingSchemeTest:
        criteria: list[MarkingCriterion] = []
        explicit_total: Decimal | None = None
        for line in lines[1:]:
            text = line.text.strip()
            if not text:
                continue
            total = _TOTAL_MARKS.fullmatch(text)
            if total:
                explicit_total = Decimal(total.group(1))
                continue
            criteria.append(_criterion(test_number, len(criteria) + 1, line))
        maximum = explicit_total or _sum_explicit_marks(tuple(criteria))
        warnings: list[MarkingSchemeWarning] = []
        if maximum is None:
            warnings.append(MarkingSchemeWarning.TEST_MAXIMUM_UNKNOWN)
        if any(item.criterion_type is CriterionType.UNKNOWN for item in criteria):
            warnings.append(MarkingSchemeWarning.UNKNOWN_CRITERION)
        source_regions = tuple(_source_region(line) for line in lines)
        accepted = tuple(
            answer for criterion in criteria for answer in criterion.accepted_answers
        )
        return MarkingSchemeTest(
            test_number=test_number,
            title=title,
            maximum_marks=maximum,
            criteria=tuple(criteria),
            accepted_answers=tuple(dict.fromkeys(accepted)),
            source_pages=tuple(dict.fromkeys(line.page_number for line in lines)),
            source_regions=source_regions,
            warnings=tuple(warnings),
        )


def lines_from_words(
    page_number: int, words: tuple[OCRWord, ...]
) -> tuple[SourceLine, ...]:
    """Build deterministic hierarchy-aware source lines from OCR words."""

    grouped: dict[tuple[int, int, int], list[tuple[int, OCRWord]]] = defaultdict(list)
    for index, word in enumerate(words):
        key = (
            word.block_number if word.block_number is not None else word.bbox.y + 1,
            word.paragraph_number if word.paragraph_number is not None else 1,
            word.line_number if word.line_number is not None else word.bbox.y + 1,
        )
        grouped[key].append((index, word))
    output: list[SourceLine] = []
    for items in grouped.values():
        ordered = sorted(
            items,
            key=lambda item: (
                item[1].word_number if item[1].word_number is not None else 2**31,
                item[1].bbox.x,
                item[0],
            ),
        )
        output.append(
            SourceLine(
                page_number=page_number,
                text=" ".join(word.text for _, word in ordered),
                bbox=_union(tuple(word.bbox for _, word in ordered)),
                source_word_indices=tuple(index for index, _ in ordered),
            )
        )
    return tuple(
        sorted(
            output,
            key=lambda line: (
                line.bbox.y if line.bbox else 0,
                line.bbox.x if line.bbox else 0,
            ),
        )
    )


def lines_from_text(page_number: int, text: str) -> tuple[SourceLine, ...]:
    """Build source lines when a provider supplies no word geometry."""

    return tuple(
        SourceLine(page_number=page_number, text=line) for line in text.splitlines()
    )


def _criterion(test_number: int, index: int, line: SourceLine) -> MarkingCriterion:
    original = line.text.strip()
    allocation = _MARK_ALLOCATION.search(original)
    maximum = Decimal(allocation.group(1)) if allocation else None
    without_mark = _MARK_ALLOCATION.sub("", original).strip(" -:;()[]")
    labeled = _LABELED.fullmatch(without_mark)
    accepted: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    partial: bool | None = None
    if labeled:
        label, body = labeled.groups()
        description = body.strip()
        normalized_label = label.casefold()
        if normalized_label.startswith("accepted answer"):
            accepted = _split_variants(description)
            kind = (
                CriterionType.ACCEPTED_VARIANTS
                if len(accepted) > 1
                else CriterionType.EXACT_ANSWER
            )
            rule = CriterionParserRule.ACCEPTED_ANSWER_VARIANTS
        elif normalized_label.startswith("keyword"):
            keywords = _split_variants(description)
            kind = CriterionType.KEYWORD
            rule = CriterionParserRule.KEYWORD_LABEL
            partial = True
        elif normalized_label == "content":
            kind = CriterionType.CONTENT_POINT
            rule = CriterionParserRule.CONTENT_LABEL
            partial = True
        elif normalized_label == "language":
            kind = CriterionType.LANGUAGE_QUALITY
            rule = CriterionParserRule.LANGUAGE_LABEL
            partial = True
        elif normalized_label == "format":
            kind = CriterionType.FORMAT_REQUIREMENT
            rule = CriterionParserRule.FORMAT_LABEL
        else:
            kind = CriterionType.HOLISTIC
            rule = CriterionParserRule.HOLISTIC_LABEL
            partial = True
    else:
        prefix = _ITEM_PREFIX.match(without_mark)
        description = _ITEM_PREFIX.sub("", without_mark, count=1).strip()
        variants = _split_variants(description)
        if prefix and len(variants) > 1:
            kind = CriterionType.ACCEPTED_VARIANTS
            rule = CriterionParserRule.ACCEPTED_ANSWER_VARIANTS
            accepted = variants
        elif prefix:
            kind = CriterionType.EXACT_ANSWER
            rule = CriterionParserRule.NUMBERED_ANSWER
            accepted = (description,) if description else ()
        else:
            kind = CriterionType.UNKNOWN
            rule = CriterionParserRule.PRESERVED_UNKNOWN
    if not description:
        description = original
        kind = CriterionType.UNKNOWN
        rule = CriterionParserRule.PRESERVED_UNKNOWN
        accepted = ()
        keywords = ()
    return MarkingCriterion(
        criterion_id=f"test-{test_number:02d}-criterion-{index:03d}",
        test_number=test_number,
        criterion_type=kind,
        description=description,
        accepted_answers=accepted,
        keywords=keywords,
        maximum_marks=maximum,
        partial_credit_allowed=partial,
        provenance=CriterionProvenance(
            parser_rule=rule,
            source=_source_region(line),
        ),
    )


def _split_variants(text: str) -> tuple[str, ...]:
    if "/" not in text:
        return (text.strip(),) if text.strip() else ()
    return tuple(part.strip() for part in text.split("/") if part.strip())


def _sum_explicit_marks(criteria: tuple[MarkingCriterion, ...]) -> Decimal | None:
    if not criteria or any(item.maximum_marks is None for item in criteria):
        return None
    return sum(
        (item.maximum_marks for item in criteria if item.maximum_marks), Decimal(0)
    )


def _source_region(line: SourceLine) -> SchemeSourceRegion:
    return SchemeSourceRegion(
        page_number=line.page_number,
        bbox=line.bbox,
        source_word_indices=line.source_word_indices,
        raw_source_text=line.text,
    )


def _union(boxes: tuple[BoundingBox, ...]) -> BoundingBox:
    left = min(item.x for item in boxes)
    top = min(item.y for item in boxes)
    right = max(item.x + item.width for item in boxes)
    bottom = max(item.y + item.height for item in boxes)
    return BoundingBox(x=left, y=top, width=right - left, height=bottom - top)

"""Conservative deterministic grouping for fragmented evidence candidates."""

from __future__ import annotations

from statistics import median

from app.evidence.models import EvidenceRegion
from app.evidence.v2_models import (
    EvidenceComponent,
    EvidenceGroup,
    EvidenceSeparatorV2Config,
)
from app.ocr.models import BoundingBox


def group_evidence_components(
    regions: tuple[EvidenceRegion, ...],
    config: EvidenceSeparatorV2Config,
) -> tuple[EvidenceGroup, ...]:
    """Group nearby aligned candidates without crossing large whitespace."""

    components = tuple(
        EvidenceComponent(
            component_index=index,
            bbox=region.bbox,
            source_word_indices=region.source_word_indices,
            base_region=region,
        )
        for index, region in enumerate(regions)
    )
    if not components:
        return ()
    if not config.enable_grouping:
        return tuple(
            EvidenceGroup(
                bbox=item.bbox,
                component_indices=(item.component_index,),
                components=(item,),
            )
            for item in components
        )

    parent = list(range(len(components)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    typical_height = max(1.0, float(median(item.bbox.height for item in components)))
    for left_index, left in enumerate(components):
        for right_index in range(left_index + 1, len(components)):
            right = components[right_index]
            if _should_group(left, right, typical_height, config):
                union(left_index, right_index)

    buckets: dict[int, list[EvidenceComponent]] = {}
    for index, component in enumerate(components):
        buckets.setdefault(find(index), []).append(component)
    groups = []
    for items in buckets.values():
        ordered = tuple(sorted(items, key=lambda item: (item.bbox.y, item.bbox.x)))
        groups.append(
            EvidenceGroup(
                bbox=_union(tuple(item.bbox for item in ordered)),
                component_indices=tuple(item.component_index for item in ordered),
                components=ordered,
            )
        )
    return tuple(sorted(groups, key=lambda item: (item.bbox.y, item.bbox.x)))


def _should_group(
    left: EvidenceComponent,
    right: EvidenceComponent,
    typical_height: float,
    config: EvidenceSeparatorV2Config,
) -> bool:
    left_box, right_box = left.bbox, right.bbox
    vertical_overlap = _axis_overlap(
        left_box.y,
        left_box.y + left_box.height,
        right_box.y,
        right_box.y + right_box.height,
    ) / min(left_box.height, right_box.height)
    horizontal_gap = _gap(
        left_box.x,
        left_box.x + left_box.width,
        right_box.x,
        right_box.x + right_box.width,
    )
    same_line = (
        vertical_overlap >= config.group_min_vertical_overlap
        and horizontal_gap <= config.group_horizontal_gap_ratio * typical_height
    )
    horizontal_overlap = _axis_overlap(
        left_box.x,
        left_box.x + left_box.width,
        right_box.x,
        right_box.x + right_box.width,
    ) / min(left_box.width, right_box.width)
    vertical_gap = _gap(
        left_box.y,
        left_box.y + left_box.height,
        right_box.y,
        right_box.y + right_box.height,
    )
    paragraph_neighbor = (
        config.enable_paragraph_merging
        and horizontal_overlap >= 0.20
        and vertical_gap <= config.group_vertical_gap_ratio * typical_height
    )
    same_ocr_line = bool(set(left.source_word_indices) & set(right.source_word_indices))
    color_difference = abs(
        left.base_region.features.ink.mean_saturation
        - right.base_region.features.ink.mean_saturation
    )
    return (
        same_ocr_line or same_line or paragraph_neighbor
    ) and color_difference <= 0.45


def _axis_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _gap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, max(a_start, b_start) - min(a_end, b_end))


def _union(boxes: tuple[BoundingBox, ...]) -> BoundingBox:
    left = min(box.x for box in boxes)
    top = min(box.y for box in boxes)
    right = max(box.x + box.width for box in boxes)
    bottom = max(box.y + box.height for box in boxes)
    return BoundingBox(x=left, y=top, width=right - left, height=bottom - top)

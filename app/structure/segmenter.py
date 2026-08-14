"""Deterministic spatial segmentation between accepted Test markers."""

from __future__ import annotations

from app.domain.models.paper import PaperPage
from app.ocr.models import BoundingBox
from app.structure.models import TestMarker, TestPageRegion, TestRegion


def segment_test_regions(
    pages: tuple[PaperPage, ...],
    markers: tuple[TestMarker, ...],
) -> tuple[TestRegion, ...]:
    """Create contiguous page spans without assuming one Test per page."""

    if not markers:
        return ()
    page_by_number = {page.page_number: page for page in pages}
    regions: list[TestRegion] = []
    for index, marker in enumerate(markers):
        following = markers[index + 1] if index + 1 < len(markers) else None
        end_page = following.page_number if following is not None else len(pages)
        page_regions: list[TestPageRegion] = []
        for page_number in range(marker.page_number, end_page + 1):
            page = page_by_number[page_number]
            top = marker.bbox.y if page_number == marker.page_number else 0
            bottom = (
                following.bbox.y
                if following is not None and page_number == following.page_number
                else page.height
            )
            if bottom <= top:
                raise ValueError("Test markers do not define positive regions")
            page_regions.append(
                TestPageRegion(
                    page_number=page_number,
                    bbox=BoundingBox(
                        x=0,
                        y=top,
                        width=page.width,
                        height=bottom - top,
                    ),
                )
            )
        regions.append(
            TestRegion(
                test_number=marker.test_number,
                label=marker.label,
                marker=marker,
                page_regions=tuple(page_regions),
                start_page=marker.page_number,
                end_page=end_page,
            )
        )
    return tuple(regions)

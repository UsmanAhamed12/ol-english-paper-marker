"""Explicit V2 evidence separator that leaves production V1 unchanged."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2

from app.core.exceptions import EvidenceSeparationError
from app.domain.models.paper import PaperPage
from app.evidence.answer_regions import AnswerRegionDetector
from app.evidence.answer_regions_v2 import (
    AnswerSpaceDetectorV2,
    StudentAnswerRegionDetectorV2,
)
from app.evidence.classifier_v2 import EvidenceClassifierV2
from app.evidence.features_v2 import EvidenceFeatureExtractorV2
from app.evidence.grouping_v2 import group_evidence_components
from app.evidence.models import (
    EvidenceRegion,
    EvidenceSignal,
    EvidenceType,
    TestEvidence,
)
from app.evidence.separator import EvidenceSeparator
from app.evidence.v2_models import (
    EvidenceDecisionV2,
    EvidenceGroup,
    EvidenceReasonV2,
    EvidenceSeparatorV2Config,
    TestEvidenceV2,
)
from app.ocr.models import BoundingBox, OCRPageResult


class EvidenceSeparatorV2:
    """Group V1 candidates and reclassify them with typed multi-signal evidence."""

    def __init__(
        self,
        config: EvidenceSeparatorV2Config,
        *,
        feature_extractor: EvidenceFeatureExtractorV2 | None = None,
    ) -> None:
        self.config = config
        self._features = feature_extractor or EvidenceFeatureExtractorV2()
        self._classifier = EvidenceClassifierV2(config)
        self._space_detector = AnswerSpaceDetectorV2()
        self._answer_detector = StudentAnswerRegionDetectorV2(config)

    def separate(
        self,
        page: PaperPage,
        ocr_result: OCRPageResult,
        *,
        test_number: int,
        region_bbox: BoundingBox,
    ) -> TestEvidenceV2:
        """Run the existing OCR boundary through an explicit parallel V2 path."""

        base_regions = EvidenceSeparator().separate(
            page,
            ocr_result,
            test_number=test_number,
            region_bbox=region_bbox,
        )
        base_answers = AnswerRegionDetector().detect(
            page,
            test_number=test_number,
            region_bbox=region_bbox,
            evidence_regions=base_regions,
        )
        return self.separate_existing(
            page,
            TestEvidence(
                paper_id=page.paper_id,
                page_number=page.page_number,
                test_number=test_number,
                region_bbox=region_bbox,
                evidence_regions=base_regions,
                answer_regions=base_answers,
            ),
        )

    def separate_existing(self, page: PaperPage, base: TestEvidence) -> TestEvidenceV2:
        """Apply V2 deterministically to frozen V1 evidence without rerunning OCR."""

        _validate(page, base)
        source_hash = _sha256(page.image_path)
        image = cv2.imread(str(page.image_path), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3:
            raise EvidenceSeparationError("V2 evidence source image is invalid")
        answer_spaces = self._space_detector.detect(base)
        groups = _bounded_groups(
            group_evidence_components(base.evidence_regions, self.config),
            base.region_bbox,
            self.config,
        )
        features = tuple(
            self._features.extract(
                image,
                group,
                test_region=base.region_bbox,
                answer_spaces=tuple(space.bbox for space in answer_spaces),
            )
            for group in groups
        )
        decisions = tuple(self._classifier.classify(item) for item in features)
        regions = tuple(
            _region(group, decision)
            for group, decision in zip(groups, decisions, strict=True)
        )
        answers = self._answer_detector.detect(base, regions, answer_spaces)
        if _sha256(page.image_path) != source_hash:
            raise EvidenceSeparationError("Canonical page changed during V2 separation")
        return TestEvidenceV2(
            paper_id=base.paper_id,
            page_number=base.page_number,
            test_number=base.test_number,
            region_bbox=base.region_bbox,
            groups=groups,
            group_features=features,
            decisions=decisions,
            evidence_regions=regions,
            answer_spaces=answer_spaces,
            student_answer_regions=answers,
            configuration_hash=self.config.fingerprint(),
        )


def as_test_evidence(result: TestEvidenceV2) -> TestEvidence:
    """Adapt explicit V2 results to the existing benchmark metric boundary."""

    return TestEvidence(
        paper_id=result.paper_id,
        page_number=result.page_number,
        test_number=result.test_number,
        region_bbox=result.region_bbox,
        evidence_regions=result.evidence_regions,
        answer_regions=result.student_answer_regions,
    )


def _region(group: EvidenceGroup, decision: EvidenceDecisionV2) -> EvidenceRegion:
    base = group.components[0].base_region
    score = {
        EvidenceType.PRINTED: decision.scores.printed,
        EvidenceType.STUDENT_CANDIDATE: decision.scores.student,
        EvidenceType.TEACHER_CANDIDATE: decision.scores.teacher,
        EvidenceType.UNKNOWN: decision.scores.ambiguity,
    }[decision.evidence_type]
    return EvidenceRegion(
        paper_id=base.paper_id,
        page_number=base.page_number,
        test_number=base.test_number,
        bbox=group.bbox,
        evidence_type=decision.evidence_type,
        confidence=score,
        signals=_legacy_signals(group, decision),
        features=base.features,
        source_word_indices=tuple(
            sorted(
                {
                    index
                    for component in group.components
                    for index in component.source_word_indices
                }
            )
        ),
        source_image_path=base.source_image_path,
        classification_strategy=decision.strategy_version,
    )


def _legacy_signals(
    group: EvidenceGroup, decision: EvidenceDecisionV2
) -> tuple[EvidenceSignal, ...]:
    signals = [
        signal for item in group.components for signal in item.base_region.signals
    ]
    mapping = {
        EvidenceReasonV2.REGULAR_OCR_LINE: EvidenceSignal.REGULAR_GEOMETRY,
        EvidenceReasonV2.HIGH_OCR_SUPPORT: EvidenceSignal.HIGH_OCR_CONFIDENCE,
        EvidenceReasonV2.DENSE_ALIGNED_TEXT: EvidenceSignal.DENSE_TEXT,
        EvidenceReasonV2.IRREGULAR_GEOMETRY: EvidenceSignal.IRREGULAR_BASELINE,
        EvidenceReasonV2.FRAGMENTED_COMPONENTS: EvidenceSignal.FRAGMENTED_STROKES,
        EvidenceReasonV2.CHROMATIC_DIFFERENCE: EvidenceSignal.CHROMATIC_INK,
        EvidenceReasonV2.ISOLATED_SMALL_MARK: EvidenceSignal.ISOLATED_MARK,
        EvidenceReasonV2.MARGIN_ANNOTATION: EvidenceSignal.MARGIN_POSITION,
    }
    signals.extend(mapping[reason] for reason in decision.reasons if reason in mapping)
    return tuple(dict.fromkeys(signals))


def _bounded_groups(
    groups: tuple[EvidenceGroup, ...],
    region: BoundingBox,
    config: EvidenceSeparatorV2Config,
) -> tuple[EvidenceGroup, ...]:
    maximum = region.width * region.height * config.maximum_group_area_ratio
    output: list[EvidenceGroup] = []
    for group in groups:
        if group.bbox.width * group.bbox.height <= maximum:
            output.append(group)
            continue
        output.extend(
            EvidenceGroup(
                bbox=component.bbox,
                component_indices=(component.component_index,),
                components=(component,),
            )
            for component in group.components
        )
    return tuple(sorted(output, key=lambda item: (item.bbox.y, item.bbox.x)))


def _validate(page: PaperPage, base: TestEvidence) -> None:
    if not page.image_path.is_file():
        raise EvidenceSeparationError("V2 evidence source image is unavailable")
    if base.paper_id != page.paper_id or base.page_number != page.page_number:
        raise EvidenceSeparationError("V2 evidence provenance does not match page")
    source_paths = {
        region.source_image_path.resolve() for region in base.evidence_regions
    }
    source_paths.update(
        region.source_image_path.resolve() for region in base.answer_regions
    )
    if source_paths and source_paths != {page.image_path.resolve()}:
        raise EvidenceSeparationError("V2 evidence source does not match page")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

"""Deterministic aggregation of Phase 5B and guarded Phase 5C outcomes."""

from __future__ import annotations

from decimal import Decimal
from time import perf_counter

from app.marking.engine import ENGINE_VERSION, DeterministicMarkingEngine
from app.marking.models import CriterionMarkResult, MarkingStatus, PaperMarkingResult
from app.marking.subjective.models import (
    CombinedPaperMarkingResult,
    CombinedTestMarkResult,
    SubjectiveCriterionMarkResult,
    SubjectiveMarkingStatus,
    SubjectivePaperMarkingResult,
)
from app.marking.subjective.service import (
    SUBJECTIVE_SERVICE_VERSION,
    SUBJECTIVE_TYPES,
    SubjectiveMarkingService,
)
from app.marking_scheme.models import AlignedPaper, CriterionType


class CombinedMarkingService:
    """Run both engines and expose confirmed marks without a false final score."""

    def __init__(
        self,
        subjective: SubjectiveMarkingService,
        *,
        deterministic: DeterministicMarkingEngine | None = None,
    ) -> None:
        self._subjective = subjective
        self._deterministic = deterministic or DeterministicMarkingEngine()

    def mark_paper(self, aligned: AlignedPaper) -> CombinedPaperMarkingResult:
        objective_paper = self.mark_objective(aligned)
        subjective_paper = self.mark_subjective(aligned)
        return self.combine(aligned, objective_paper, subjective_paper)

    def mark_objective(self, aligned: AlignedPaper) -> PaperMarkingResult:
        """Run the unchanged Phase 5B engine exactly once."""

        return self._deterministic.mark_paper(aligned)

    def mark_subjective(self, aligned: AlignedPaper) -> SubjectivePaperMarkingResult:
        """Run the guarded Phase 5C service exactly once."""

        return self._subjective.mark_paper(aligned)

    def combine(
        self,
        aligned: AlignedPaper,
        objective_paper: PaperMarkingResult,
        subjective_paper: SubjectivePaperMarkingResult,
    ) -> CombinedPaperMarkingResult:
        """Combine previously measured component results without re-marking."""

        started = perf_counter()
        if (
            objective_paper.scheme_id != aligned.scheme_id
            or subjective_paper.scheme_id != aligned.scheme_id
            or objective_paper.student_paper_id != aligned.paper_id
            or subjective_paper.student_paper_id != aligned.paper_id
        ):
            raise ValueError("Component marking results do not match alignment")
        objective_by_test = {
            item.test_number: item for item in objective_paper.test_results
        }
        subjective_by_test: dict[int, list[SubjectiveCriterionMarkResult]] = {}
        for item in subjective_paper.criterion_results:
            subjective_by_test.setdefault(item.test_number, []).append(item)

        test_results: list[CombinedTestMarkResult] = []
        for aligned_test in aligned.tests:
            objective_test = objective_by_test[aligned_test.test_number]
            objective_results = tuple(
                item
                for item in objective_test.criterion_results
                if item.criterion_type not in SUBJECTIVE_TYPES
                and item.criterion_type is not CriterionType.UNKNOWN
            )
            subjective_results = tuple(
                subjective_by_test.get(aligned_test.test_number, ())
            )
            maximum = (
                aligned_test.marking_scheme_test.maximum_marks
                if aligned_test.marking_scheme_test is not None
                and aligned_test.marking_scheme_test.maximum_marks is not None
                else _known_criterion_maximum(objective_results, subjective_results)
            )
            objective_confirmed = _objective_confirmed(objective_results)
            subjective_confirmed = _subjective_confirmed(subjective_results)
            review = _review_marks(objective_results, subjective_results)
            unsupported = _unsupported_marks(objective_results, subjective_results)
            confirmed = objective_confirmed + subjective_confirmed
            fully_resolved = (
                review == 0
                and unsupported == 0
                and _all_results_resolved(objective_results, subjective_results)
            )
            test_results.append(
                CombinedTestMarkResult(
                    test_number=aligned_test.test_number,
                    objective_results=objective_results,
                    subjective_results=subjective_results,
                    confirmed_objective_marks=objective_confirmed,
                    confirmed_subjective_marks=subjective_confirmed,
                    confirmed_marks=confirmed,
                    unresolved_marks=review,
                    unsupported_marks=unsupported,
                    maximum_known_marks=maximum,
                    final_score=confirmed if fully_resolved else None,
                )
            )

        objective_total = sum(
            (item.confirmed_objective_marks for item in test_results), Decimal("0")
        )
        subjective_total = sum(
            (item.confirmed_subjective_marks for item in test_results), Decimal("0")
        )
        confirmed_total = objective_total + subjective_total
        review_total = sum(
            (item.unresolved_marks for item in test_results), Decimal("0")
        )
        unsupported_total = sum(
            (item.unsupported_marks for item in test_results), Decimal("0")
        )
        maximum_total = sum(
            (item.maximum_known_marks for item in test_results), Decimal("0")
        )
        final_score = (
            confirmed_total
            if review_total == 0
            and unsupported_total == 0
            and all(item.final_score is not None for item in test_results)
            else None
        )
        return CombinedPaperMarkingResult(
            scheme_id=aligned.scheme_id,
            student_paper_id=aligned.paper_id,
            test_results=tuple(test_results),
            confirmed_objective_marks=objective_total,
            confirmed_subjective_marks=subjective_total,
            confirmed_marks=confirmed_total,
            review_required_marks=review_total,
            unsupported_marks=unsupported_total,
            maximum_known_marks=maximum_total,
            final_score=final_score,
            processing_duration_ms=(perf_counter() - started) * 1000,
            engine_versions=(ENGINE_VERSION, SUBJECTIVE_SERVICE_VERSION),
        )


def _known_criterion_maximum(
    objective: tuple[CriterionMarkResult, ...],
    subjective: tuple[SubjectiveCriterionMarkResult, ...],
) -> Decimal:
    return sum(
        (item.maximum_marks or Decimal("0") for item in objective),
        Decimal("0"),
    ) + sum((item.maximum_marks or Decimal("0") for item in subjective), Decimal("0"))


def _objective_confirmed(results: tuple[CriterionMarkResult, ...]) -> Decimal:
    return sum((item.awarded_marks or Decimal("0") for item in results), Decimal("0"))


def _subjective_confirmed(
    results: tuple[SubjectiveCriterionMarkResult, ...],
) -> Decimal:
    return sum((item.awarded_marks or Decimal("0") for item in results), Decimal("0"))


def _review_marks(
    objective: tuple[CriterionMarkResult, ...],
    subjective: tuple[SubjectiveCriterionMarkResult, ...],
) -> Decimal:
    objective_review = sum(
        (
            item.maximum_marks or Decimal("0")
            for item in objective
            if item.status
            not in {
                MarkingStatus.MARKED,
                MarkingStatus.NOT_ATTEMPTED,
                MarkingStatus.UNSUPPORTED_CRITERION,
            }
        ),
        Decimal("0"),
    )
    subjective_review = sum(
        (
            item.maximum_marks or Decimal("0")
            for item in subjective
            if item.status
            not in {
                SubjectiveMarkingStatus.MARKED,
                SubjectiveMarkingStatus.UNSUPPORTED_CRITERION,
            }
        ),
        Decimal("0"),
    )
    return objective_review + subjective_review


def _unsupported_marks(
    objective: tuple[CriterionMarkResult, ...],
    subjective: tuple[SubjectiveCriterionMarkResult, ...],
) -> Decimal:
    objective_unsupported = sum(
        (
            item.maximum_marks or Decimal("0")
            for item in objective
            if item.status is MarkingStatus.UNSUPPORTED_CRITERION
        ),
        Decimal("0"),
    )
    subjective_unsupported = sum(
        (
            item.maximum_marks or Decimal("0")
            for item in subjective
            if item.status is SubjectiveMarkingStatus.UNSUPPORTED_CRITERION
        ),
        Decimal("0"),
    )
    return objective_unsupported + subjective_unsupported


def _all_results_resolved(
    objective: tuple[CriterionMarkResult, ...],
    subjective: tuple[SubjectiveCriterionMarkResult, ...],
) -> bool:
    return all(
        item.status in {MarkingStatus.MARKED, MarkingStatus.NOT_ATTEMPTED}
        for item in objective
    ) and all(item.status is SubjectiveMarkingStatus.MARKED for item in subjective)

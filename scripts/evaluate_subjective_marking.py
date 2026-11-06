"""Run a privacy-safe synthetic smoke of the guarded subjective path."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from decimal import Decimal

from app.answers.models import AnswerExtractionStatus
from app.core.config import get_settings
from app.llm.factory import create_grading_chat_model
from app.marking.subjective.models import (
    SubjectiveMarkingStatus,
    SubjectiveReasonCode,
)
from app.marking.subjective.ollama import OllamaSubjectiveMarkingProvider
from app.marking.subjective.provider import (
    SubjectiveMarkingProvider,
    SubjectiveProviderResponse,
)
from app.marking.subjective.service import SubjectiveMarkingService
from app.marking_scheme.models import (
    AlignedPaper,
    AlignedTest,
    AlignmentStatus,
    CriterionType,
    MarkingSchemeTest,
    SchemeSourceRegion,
)
from app.ocr.providers.ollama import OllamaHTTPClient
from scripts.evaluate_marking_engine import PAPER_ID, SCHEME_ID, _answer, _criterion


class SyntheticSubjectiveProvider:
    """Deterministic fake used by the default development smoke."""

    name = "synthetic"
    model_version = "synthetic-v1"

    def assess(self, prompt: str) -> SubjectiveProviderResponse:
        criterion_line = prompt.split("AUTHORITATIVE_CRITERION_JSON:\n", 1)[1]
        criterion = json.loads(criterion_line.split("\n", 1)[0])
        payload = {
            "test_number": criterion["test_number"],
            "criterion_id": criterion["criterion_id"],
            "criterion_type": criterion["criterion_type"],
            "maximum_marks": criterion["maximum_marks"],
            "awarded_marks": "2",
            "status": "marked",
            "matched_requirements": ["synthetic requirement"],
            "missing_requirements": [],
            "reason_codes": ["synthetic_assessment"],
            "feedback": "Synthetic feedback.",
            "confidence": 0.95,
        }
        return SubjectiveProviderResponse(
            raw_text=json.dumps(payload),
            provider=self.name,
            model_version=self.model_version,
            processing_duration_ms=0,
        )


def build_subjective_alignment() -> AlignedPaper:
    """Build four non-private subjective examples."""

    kinds = (
        CriterionType.CONTENT_POINT,
        CriterionType.LANGUAGE_QUALITY,
        CriterionType.FORMAT_REQUIREMENT,
        CriterionType.HOLISTIC,
    )
    tests: list[AlignedTest] = []
    for number, kind in enumerate(kinds, start=1):
        criterion = _criterion(
            number,
            kind,
            answers=(),
            keywords=(),
            maximum=Decimal("4"),
        )
        tests.append(
            AlignedTest(
                test_number=number,
                status=AlignmentStatus.ALIGNED,
                student_answer=_answer(
                    number,
                    "A synthetic response used only for integration validation.",
                    AnswerExtractionStatus.EXTRACTED,
                ),
                marking_scheme_test=MarkingSchemeTest(
                    test_number=number,
                    maximum_marks=Decimal("4"),
                    criteria=(criterion,),
                    source_pages=(1,),
                    source_regions=(
                        SchemeSourceRegion(
                            page_number=1,
                            raw_source_text="synthetic rubric",
                        ),
                    ),
                ),
            )
        )
    return AlignedPaper(
        paper_id=PAPER_ID,
        scheme_id=SCHEME_ID,
        tests=tuple(tests),
        aligned_test_numbers=(1, 2, 3, 4),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ollama-model",
        help="Use this already-installed local model instead of the fake provider",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    provider: SubjectiveMarkingProvider
    if args.ollama_model:
        settings = get_settings()
        settings = settings.model_copy(
            update={"ollama_grading_model": args.ollama_model}
        )
        client = OllamaHTTPClient(
            base_url=str(settings.ollama_base_url),
            timeout_seconds=settings.ollama_grading_timeout_seconds,
        )
        info = client.model_info(args.ollama_model)
        provider = OllamaSubjectiveMarkingProvider(
            chat_model=create_grading_chat_model(settings),
            model=info.model,
            model_version=info.digest,
        )
    else:
        provider = SyntheticSubjectiveProvider()
    result = SubjectiveMarkingService(provider).mark_paper(build_subjective_alignment())
    marked = sum(
        item.status is SubjectiveMarkingStatus.MARKED
        for item in result.criterion_results
    )
    review = len(result.criterion_results) - marked
    invalid_or_failed = sum(
        item.status
        in {
            SubjectiveMarkingStatus.INVALID_MODEL_RESPONSE,
            SubjectiveMarkingStatus.PROVIDER_FAILED,
        }
        for item in result.criterion_results
    )
    statuses = Counter(item.status.value for item in result.criterion_results)
    reasons = Counter(
        str(reason) for item in result.criterion_results for reason in item.reason_codes
    )
    cap_violations = sum(
        SubjectiveReasonCode.SCORE_OUT_OF_RANGE in item.reason_codes
        for item in result.criterion_results
    )
    print(
        json.dumps(
            {
                "criteria": len(result.criterion_results),
                "duration_ms": round(result.processing_duration_ms, 3),
                "invalid_or_failed": invalid_or_failed,
                "mark_cap_violations": cap_violations,
                "marked": marked,
                "provider": provider.name,
                "reason_counts": dict(sorted(reasons.items())),
                "review_or_failed": review,
                "status_counts": dict(sorted(statuses.items())),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if invalid_or_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

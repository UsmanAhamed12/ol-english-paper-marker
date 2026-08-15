"""Safe CLI tests for the private evidence-v2 workflow."""

from pathlib import Path

from app.evaluation.evidence_expansion.models import (
    EvidenceCandidateCategory,
    EvidenceExpansionManifest,
    EvidenceExpansionSample,
)
from app.ocr.models import BoundingBox
from scripts.annotate_evidence import EXPANDED_MANIFEST
from scripts.annotate_evidence import build_parser as label_parser
from scripts.prepare_evidence_expansion import _safe_summary, build_parser


def test_expansion_cli_and_labeler_dataset_switch() -> None:
    args = build_parser().parse_args(["prepare", "--paper-count", "8"])
    assert args.paper_count == 8
    label_args = label_parser().parse_args(["--dataset", "evidence-v2"])
    assert label_args.dataset == "evidence-v2"
    assert EXPANDED_MANIFEST.name == "benchmark_manifest.json"


def test_safe_summary_contains_no_source_identity(tmp_path: Path) -> None:
    source = (tmp_path / "private-student-name.png").resolve()
    sample = EvidenceExpansionSample(
        sample_id="evidence_v2_001",
        paper_alias="paper-a",
        page_number=1,
        source_image_path=source,
        source_image_sha256="0" * 64,
        page_width=100,
        page_height=100,
        region=BoundingBox(x=0, y=0, width=50, height=50),
        discovery_category=EvidenceCandidateCategory.PRINTED,
        discovery_reason="synthetic_candidate",
    )
    summary = _safe_summary(EvidenceExpansionManifest(samples=(sample,)), 1, 0)
    assert "private-student-name" not in str(summary)
    assert summary["candidate_categories_are_ground_truth"] is False

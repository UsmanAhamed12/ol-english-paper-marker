"""Launch the loopback-only private evidence visual-labeling interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from app.evaluation.evidence_benchmark.annotation_web import (
    DEFAULT_ANNOTATION_PORT,
    LOOPBACK_HOST,
    create_annotation_server,
)
from app.evaluation.evidence_benchmark.annotations import (
    EvidenceAnnotationRepository,
)
from app.evaluation.evidence_benchmark.manifest import load_evidence_manifest
from app.evaluation.evidence_benchmark.models import EvidenceBenchmarkManifest
from app.evaluation.evidence_benchmark.reverification import (
    EvidenceReverificationRepository,
)
from app.evaluation.evidence_expansion.manifest import (
    load_evidence_expansion_manifest,
)
from app.evaluation.evidence_expansion.models import EvidenceExpansionManifest
from app.evaluation.teacher_evidence.manifest import load_teacher_evidence_manifest
from app.evaluation.teacher_evidence.models import TeacherEvidenceManifest

DEFAULT_MANIFEST = Path("data/evaluation/evidence/benchmark_manifest.json")
EXPANDED_MANIFEST = Path("data/evaluation/evidence_v2/benchmark_manifest.json")
TEACHER_MANIFEST = Path("data/evaluation/evidence_teacher_v1/benchmark_manifest.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--dataset",
        choices=("phase4c4", "evidence-v2", "evidence-teacher-v1"),
        default="phase4c4",
        help="select the private benchmark namespace",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_ANNOTATION_PORT)
    parser.add_argument(
        "--reverify",
        action="store_true",
        help="require explicit approval of every sample in a new private session",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    default_manifests = {
        "phase4c4": DEFAULT_MANIFEST,
        "evidence-v2": EXPANDED_MANIFEST,
        "evidence-teacher-v1": TEACHER_MANIFEST,
    }
    requested_manifest = (
        default_manifests[arguments.dataset]
        if arguments.manifest == DEFAULT_MANIFEST
        else arguments.manifest
    )
    manifest_path = requested_manifest.resolve(strict=True)
    private_root = manifest_path.parent.resolve()
    manifest: (
        EvidenceBenchmarkManifest | EvidenceExpansionManifest | TeacherEvidenceManifest
    )
    if arguments.dataset == "evidence-v2":
        manifest = load_evidence_expansion_manifest(manifest_path)
    elif arguments.dataset == "evidence-teacher-v1":
        manifest = load_teacher_evidence_manifest(manifest_path)
    else:
        manifest = load_evidence_manifest(manifest_path)
    if arguments.reverify and arguments.dataset != "phase4c4":
        raise ValueError("Pending datasets do not use re-verification")
    repository = EvidenceAnnotationRepository(
        manifest,
        private_root / "annotations.json",
        private_root=private_root,
    )
    reverification_repository = None
    if arguments.reverify:
        reverification_repository = EvidenceReverificationRepository(
            private_root / "reverification_session.json",
            private_root=private_root,
        )
        reverification_repository.initialize(repository.load())
    server = create_annotation_server(
        manifest,
        private_root,
        repository,
        port=arguments.port,
        reverification_repository=reverification_repository,
    )
    if reverification_repository is None:
        completed, total = repository.completion()
        progress_label = "Progress"
    else:
        completed = len(reverification_repository.current_ids(repository.load()))
        total = len(manifest.samples)
        progress_label = "Re-verified"
    print(f"Private evidence labeler: http://{LOOPBACK_HOST}:{server.server_port}")
    print(f"{progress_label}: {completed}/{total}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

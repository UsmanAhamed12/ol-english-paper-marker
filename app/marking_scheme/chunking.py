"""Question-aware conversion of parsed rubric criteria into retrieval chunks."""

from __future__ import annotations

import hashlib

from app.marking_scheme.models import (
    MarkingCriterion,
    MarkingSchemeTest,
    StructuredMarkingScheme,
)
from app.vectorstore.models import KnowledgeDocument, MetadataValue

CHUNKER_VERSION = "marking-scheme-chunker-v1"


class MarkingSchemeChunker:
    """Create stable, criterion-scoped chunks with exact Test metadata."""

    def chunk(self, scheme: StructuredMarkingScheme) -> tuple[KnowledgeDocument, ...]:
        """Return one chunk per criterion, or a source-preserving Test fallback."""

        documents: list[KnowledgeDocument] = []
        for scheme_test in scheme.tests:
            if scheme_test.criteria:
                documents.extend(
                    self._criterion_document(scheme, scheme_test, criterion)
                    for criterion in scheme_test.criteria
                )
            else:
                documents.append(self._test_fallback_document(scheme, scheme_test))
        return tuple(documents)

    def _criterion_document(
        self,
        scheme: StructuredMarkingScheme,
        scheme_test: MarkingSchemeTest,
        criterion: MarkingCriterion,
    ) -> KnowledgeDocument:
        source = criterion.provenance.source
        metadata = _test_metadata(scheme, scheme_test)
        metadata.update(
            {
                "chunker_version": CHUNKER_VERSION,
                "criterion_id": criterion.criterion_id,
                "criterion_type": criterion.criterion_type.value,
                "parser_rule": criterion.provenance.parser_rule.value,
                "page_number": source.page_number,
                "has_known_criterion_maximum": criterion.maximum_marks is not None,
            }
        )
        if criterion.maximum_marks is not None:
            metadata["criterion_maximum_marks"] = str(criterion.maximum_marks)
        text = _criterion_text(scheme_test, criterion)
        return KnowledgeDocument(
            document_id=_document_id(
                scheme.provenance.source_sha256,
                scheme_test.test_number,
                criterion.criterion_id,
            ),
            text=text,
            metadata=metadata,
        )

    def _test_fallback_document(
        self,
        scheme: StructuredMarkingScheme,
        scheme_test: MarkingSchemeTest,
    ) -> KnowledgeDocument:
        metadata = _test_metadata(scheme, scheme_test)
        metadata.update(
            {
                "chunker_version": CHUNKER_VERSION,
                "criterion_id": f"test-{scheme_test.test_number:02d}-summary",
                "criterion_type": "test_summary",
                "page_number": scheme_test.source_pages[0],
                "has_known_criterion_maximum": False,
            }
        )
        source_text = "\n".join(
            region.raw_source_text for region in scheme_test.source_regions
        )
        text = _test_header(scheme_test)
        if source_text:
            text = f"{text}\nSource material:\n{source_text}"
        return KnowledgeDocument(
            document_id=_document_id(
                scheme.provenance.source_sha256,
                scheme_test.test_number,
                "test-summary",
            ),
            text=text,
            metadata=metadata,
        )


def _test_metadata(
    scheme: StructuredMarkingScheme,
    scheme_test: MarkingSchemeTest,
) -> dict[str, MetadataValue]:
    metadata: dict[str, MetadataValue] = {
        "document_type": "marking_scheme",
        "source_type": "marking_scheme",
        "source_id": str(scheme.scheme_id),
        "scheme_id": str(scheme.scheme_id),
        "source_sha256": scheme.provenance.source_sha256,
        "test_number": scheme_test.test_number,
        "test_label": f"Test {scheme_test.test_number:02d}",
        "source_pages": ",".join(str(page) for page in scheme_test.source_pages),
        "has_known_test_maximum": scheme_test.maximum_marks is not None,
    }
    if scheme_test.title is not None:
        metadata["test_title"] = scheme_test.title
    if scheme_test.maximum_marks is not None:
        metadata["test_maximum_marks"] = str(scheme_test.maximum_marks)
    return metadata


def _test_header(scheme_test: MarkingSchemeTest) -> str:
    header = f"Test {scheme_test.test_number:02d}"
    if scheme_test.title:
        header = f"{header}: {scheme_test.title}"
    if scheme_test.maximum_marks is not None:
        header = f"{header}\nTest maximum marks: {scheme_test.maximum_marks}"
    return header


def _criterion_text(
    scheme_test: MarkingSchemeTest,
    criterion: MarkingCriterion,
) -> str:
    lines = [_test_header(scheme_test), ""]
    lines.append(
        f"Criterion {criterion.criterion_id} ({criterion.criterion_type.value})"
    )
    lines.append(f"Description: {criterion.description}")
    if criterion.maximum_marks is not None:
        lines.append(f"Criterion maximum marks: {criterion.maximum_marks}")
    if criterion.accepted_answers:
        lines.append(f"Accepted answers: {' / '.join(criterion.accepted_answers)}")
    if criterion.keywords:
        lines.append(f"Keywords: {' / '.join(criterion.keywords)}")
    for component in criterion.components:
        evidence = (
            f"accepted answers: {' / '.join(component.accepted_answers)}"
            if component.accepted_answers
            else f"keywords: {' / '.join(component.keywords)}"
        )
        lines.append(
            f"{component.component_id} ({component.maximum_marks} marks; {evidence})"
        )
    lines.append(f"Source page: {criterion.provenance.source.page_number}")
    lines.append(f"Source text: {criterion.provenance.source.raw_source_text}")
    return "\n".join(lines)


def _document_id(source_sha256: str, test_number: int, item_id: str) -> str:
    identity = f"{CHUNKER_VERSION}:{source_sha256}:test-{test_number:02d}:{item_id}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()

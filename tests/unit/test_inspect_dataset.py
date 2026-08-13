"""Tests for the Phase 0 dataset inspection helpers."""

from __future__ import annotations

import unittest

from scripts.inspect_dataset import (
    ImageRecord,
    PdfRecord,
    parse_pdfimages,
    parse_pdfinfo,
    summarize,
)


class ParsePdfInfoTests(unittest.TestCase):
    def test_parses_colon_separated_fields(self) -> None:
        output = (
            "Pages:          10\nEncrypted:      no\nPage size:      612 x 792 pts\n"
        )

        fields = parse_pdfinfo(output)

        self.assertEqual(fields["Pages"], "10")
        self.assertEqual(fields["Encrypted"], "no")
        self.assertEqual(fields["Page size"], "612 x 792 pts")


class ParsePdfImagesTests(unittest.TestCase):
    def test_skips_headers_and_parses_image_rows(self) -> None:
        header = (
            "page num type width height color comp bpc enc interp "
            "object ID x-ppi y-ppi size ratio"
        )
        image_row = "1 0 image 1636 2367 icc 3 8 jpeg yes 5 0 72 72 1309K 12%"
        output = f"{header}\n{'-' * len(header)}\n{image_row}\n"

        records = parse_pdfimages(output)

        self.assertEqual(
            records,
            (
                ImageRecord(
                    page=1, width=1636, height=2367, encoding="jpeg", x_ppi=72, y_ppi=72
                ),
            ),
        )


class SummarizeTests(unittest.TestCase):
    def test_summarizes_records_deterministically(self) -> None:
        image_a = ImageRecord(1, 1500, 2200, "jpeg", 72, 72)
        image_b = ImageRecord(1, 1700, 2400, "jpeg", 72, 72)
        records = [
            PdfRecord("a.pdf", 100, 8, False, 0, (image_a,)),
            PdfRecord("b.pdf", 200, 10, False, 25, (image_b,)),
        ]

        summary = summarize(records)

        self.assertEqual(summary.pdf_count, 2)
        self.assertEqual(summary.total_pages, 18)
        self.assertEqual(summary.page_count_distribution, {8: 1, 10: 1})
        self.assertEqual(summary.median_pages, 9.0)
        self.assertEqual(summary.textless_pdfs, 1)
        self.assertEqual(summary.image_count, 2)
        self.assertEqual(summary.median_image_width, 1600.0)

    def test_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "No PDF records"):
            summarize([])


if __name__ == "__main__":
    unittest.main()

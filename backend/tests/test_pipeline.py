"""Focused tests for parsing, chunking, and citation formatting."""

import tempfile
import unittest
from pathlib import Path

from app.chunking.structured_chunker import StructuredChunker
from app.formatting.context_formatter import format_retrieved_context
from app.ingestion.parsers import parse_file
from app.models.documents import DocumentSection, ParsedDocument


class ParserTests(unittest.TestCase):
    def test_markdown_parser_preserves_section_headings(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "guide.md"
            path.write_text("# Overview\nFirst paragraph.\n\n## Setup\nInstall the app.", encoding="utf-8")

            parsed = parse_file(path)

        self.assertEqual([section.heading for section in parsed.sections], ["Overview", "Setup"])
        self.assertIn("Install the app.", parsed.sections[1].text)


class ChunkerTests(unittest.TestCase):
    def test_chunks_respect_size_and_preserve_source_metadata(self) -> None:
        document = ParsedDocument(
            document_id="doc-1",
            title="Guide",
            source_path="guide.md",
            file_type="md",
            sections=[
                DocumentSection(
                    heading="Setup",
                    text="First sentence has useful details. " * 7,
                    page_number=2,
                )
            ],
        )
        chunks = StructuredChunker(chunk_size=90, chunk_overlap=12).chunk_document(document)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 90 for chunk in chunks))
        self.assertTrue(all(chunk.section == "Setup" and chunk.page_number == 2 for chunk in chunks))
        self.assertEqual(len({chunk.chunk_id for chunk in chunks}), len(chunks))

    def test_invalid_overlap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            StructuredChunker(chunk_size=100, chunk_overlap=100)


class ContextFormatterTests(unittest.TestCase):
    def test_context_and_citations_keep_source_locations(self) -> None:
        context, sources = format_retrieved_context(
            [{"title": "Guide", "page_number": 3, "section": "Safety", "content": "Wear gloves."}]
        )

        self.assertIn("[1] Guide (Safety, page 3)", context)
        self.assertEqual(sources[0]["citation"], 1)
        self.assertEqual(sources[0]["page_number"], 3)


if __name__ == "__main__":
    unittest.main()

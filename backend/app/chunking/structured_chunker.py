"""Split source sections into overlapping, heading-aware passages."""

import re
import uuid

from app.models.documents import ParsedDocument, TextChunk


class StructuredChunker:
    """Create bounded chunks while preserving paragraph and sentence boundaries."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Chunk size must be positive and overlap smaller than chunk size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: ParsedDocument) -> list[TextChunk]:
        """Chunk all parsed sections and attach stable ids and source metadata."""
        chunks = []
        for section_index, section in enumerate(document.sections):
            prefix = f"{section.heading}\n" if section.heading else ""
            if len(prefix) >= self.chunk_size:
                prefix = prefix[:self.chunk_size - 1]
            body_limit = max(1, self.chunk_size - len(prefix))
            pieces = self._split_into_pieces(section.text, body_limit)
            grouped = self._group_with_overlap(pieces, body_limit)

            for chunk_index, body in enumerate(grouped):
                text = f"{prefix}{body}" if prefix else body
                chunk_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{document.document_id}:{section_index}:{chunk_index}",
                    )
                )
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        title=document.title,
                        source_path=document.source_path,
                        file_type=document.file_type,
                        text=text,
                        section=section.heading,
                        page_number=section.page_number,
                    )
                )
        return chunks

    @staticmethod
    def _split_into_pieces(text: str, max_chars: int) -> list[str]:
        """Prefer paragraphs, then sentences, and only hard-split oversized text."""
        pieces = []
        for paragraph in re.split(r"\n\s*\n", text.strip()):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if len(paragraph) <= max_chars:
                pieces.append(paragraph)
                continue

            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                while len(sentence) > max_chars:
                    boundary = sentence.rfind(" ", 0, max_chars + 1)
                    if boundary <= 0:
                        boundary = max_chars
                    pieces.append(sentence[:boundary].strip())
                    sentence = sentence[boundary:].strip()
                if sentence:
                    pieces.append(sentence)
        return pieces

    def _group_with_overlap(self, pieces: list[str], max_chars: int) -> list[str]:
        """Pack pieces into chunks and repeat a short tail for retrieval continuity."""
        result: list[str] = []
        current: list[str] = []

        for piece in pieces:
            candidate = "\n\n".join(current + [piece])
            if current and len(candidate) > max_chars:
                result.append("\n\n".join(current))
                current = self._overlap_tail(current)
                while current and len("\n\n".join(current + [piece])) > max_chars:
                    current.pop(0)
            current.append(piece)

        if current:
            result.append("\n\n".join(current))
        return result

    def _overlap_tail(self, pieces: list[str]) -> list[str]:
        if self.chunk_overlap == 0:
            return []
        tail = "\n\n".join(pieces)[-self.chunk_overlap:]
        first_word = tail.find(" ")
        if first_word >= 0:
            tail = tail[first_word + 1:]
        return [tail] if tail else []

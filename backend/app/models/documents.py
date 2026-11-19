"""Document and chunk models passed between pipeline stages."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentSection:
    """A heading- or page-aware part of an ingested document."""

    text: str
    heading: str | None = None
    page_number: int | None = None


@dataclass(frozen=True)
class ParsedDocument:
    """A source document after text extraction."""

    document_id: str
    title: str
    source_path: str
    file_type: str
    sections: list[DocumentSection]


@dataclass(frozen=True)
class TextChunk:
    """A searchable text passage and its source metadata."""

    chunk_id: str
    document_id: str
    title: str
    source_path: str
    file_type: str
    text: str
    section: str | None = None
    page_number: int | None = None

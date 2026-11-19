"""Extract readable text from supported files while retaining source locations."""

import hashlib
import logging
import re
from pathlib import Path

from docx import Document as WordDocument
from docx.table import Table as WordTable
from docx.text.paragraph import Paragraph as WordParagraph
from pypdf import PdfReader

from app.models.documents import DocumentSection, ParsedDocument

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def parse_file(file_path: Path) -> ParsedDocument:
    """Parse one PDF, Word, Markdown, or plain-text document."""
    path = file_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Document file does not exist: {path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported document type '{extension or '(no extension)'}': {path.name}. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    if extension == ".pdf":
        sections = _parse_pdf(path)
    elif extension == ".docx":
        sections = _parse_docx(path)
    elif extension == ".md":
        sections = _parse_markdown(path)
    else:
        text = path.read_text(encoding="utf-8-sig")
        sections = [DocumentSection(text=_clean_text(text))]

    sections = [section for section in sections if section.text.strip()]
    if not sections:
        raise ValueError(f"No extractable text was found in: {path}")

    document_id = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    logger.info("Parsed %s into %d section(s)", path.name, len(sections))
    return ParsedDocument(
        document_id=document_id,
        title=path.stem,
        source_path=str(path),
        file_type=extension.lstrip("."),
        sections=sections,
    )


def parse_directory(directory: Path) -> list[ParsedDocument]:
    """Parse supported files recursively; report missing or empty input clearly."""
    root = directory.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Document folder not found: {root}. Set DOCUMENTS_PATH to your source folder."
        )

    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(
            f"No supported documents found in {root}. Add PDF, DOCX, TXT, or Markdown files."
        )
    return [parse_file(path) for path in paths]


def _parse_pdf(path: Path) -> list[DocumentSection]:
    reader = PdfReader(str(path))
    sections = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text() or "")
        if text:
            sections.append(DocumentSection(text=text, page_number=page_number))
    return sections


def _parse_docx(path: Path) -> list[DocumentSection]:
    document = WordDocument(str(path))
    sections = []
    current_heading: str | None = None
    paragraphs: list[str] = []

    def flush() -> None:
        text = _clean_text("\n\n".join(paragraphs))
        if text:
            sections.append(DocumentSection(text=text, heading=current_heading))
        paragraphs.clear()

    for block in document.iter_inner_content():
        if isinstance(block, WordParagraph):
            text = block.text.strip()
            if not text:
                continue
            if block.style and block.style.name.startswith("Heading"):
                flush()
                current_heading = text
            else:
                paragraphs.append(text)
        elif isinstance(block, WordTable):
            for row in block.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    paragraphs.append(" | ".join(cells))
    flush()
    return sections


def _parse_markdown(path: Path) -> list[DocumentSection]:
    sections = []
    current_heading: str | None = None
    lines: list[str] = []

    def flush() -> None:
        text = _clean_text("\n".join(lines))
        if text:
            sections.append(DocumentSection(text=text, heading=current_heading))
        lines.clear()

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            flush()
            current_heading = match.group(1).strip()
        else:
            lines.append(line)
    flush()
    return sections


def _clean_text(text: str) -> str:
    """Normalize whitespace without destroying paragraph boundaries."""
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

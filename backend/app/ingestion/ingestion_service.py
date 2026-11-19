"""Run parsing, structured chunking, embedding, and indexing for a document folder."""

import logging

from app.chunking.structured_chunker import StructuredChunker
from app.config.settings import Settings
from app.embeddings.azure_embeddings import AzureEmbeddingService
from app.ingestion.parsers import parse_directory
from app.search.azure_search_store import AzureSearchStore

logger = logging.getLogger(__name__)
EMBEDDING_BATCH_SIZE = 32


class IngestionService:
    """Orchestrate the offline document-to-search-index pipeline."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._chunker = StructuredChunker(settings.chunk_size, settings.chunk_overlap)
        self._embeddings = AzureEmbeddingService(settings)
        self._search = AzureSearchStore(settings)

    def ingest_directory(self) -> dict[str, int]:
        """Parse configured documents, embed their chunks, and upload them."""
        documents = parse_directory(self._settings.documents_path)
        self._search.ensure_index()

        document_count = 0
        chunk_count = 0
        for document in documents:
            chunks = self._chunker.chunk_document(document)
            for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                batch = chunks[start:start + EMBEDDING_BATCH_SIZE]
                vectors = self._embeddings.embed_texts([chunk.text for chunk in batch])
                self._search.upload_chunks(batch, vectors)
            document_count += 1
            chunk_count += len(chunks)
            logger.info("Indexed %d chunk(s) from %s", len(chunks), document.title)

        return {"documents": document_count, "chunks": chunk_count}

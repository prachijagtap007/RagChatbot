"""Create the vector-enabled search index and run hybrid semantic retrieval."""

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from app.config.settings import Settings
from app.models.documents import TextChunk


class AzureSearchStore:
    """Persist chunk vectors and retrieve ranked passages from Azure AI Search."""

    def __init__(self, settings: Settings) -> None:
        credential = AzureKeyCredential(settings.azure_search_api_key)
        self._index_client = SearchIndexClient(
            endpoint=settings.azure_search_endpoint,
            credential=credential,
        )
        self._search_client = SearchClient(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.search_index_name,
            credential=credential,
        )
        self._index_name = settings.search_index_name
        self._semantic_config = settings.semantic_configuration_name
        self._dimensions = settings.embedding_dimensions

    def ensure_index(self) -> None:
        """Create the required index once; leave existing indexes untouched."""
        try:
            self._index_client.get_index(self._index_name)
        except ResourceNotFoundError:
            self._index_client.create_index(self._build_index())

    def upload_chunks(
        self, chunks: list[TextChunk], vectors: list[list[float]]
    ) -> None:
        """Upload chunks and their vectors as one Azure AI Search batch."""
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one embedding vector.")

        documents = []
        for chunk, vector in zip(chunks, vectors):
            if len(vector) != self._dimensions:
                raise ValueError("Chunk vector does not match the configured embedding dimensions.")
            documents.append(
                {
                    "id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "source_path": chunk.source_path,
                    "file_type": chunk.file_type,
                    "content": chunk.text,
                    "section": chunk.section or "",
                    "page_number": chunk.page_number,
                    "content_vector": vector,
                }
            )

        results = self._search_client.upload_documents(documents=documents)
        failed = [result for result in results if not result.succeeded]
        if failed:
            failed_ids = ", ".join(result.key for result in failed)
            raise RuntimeError(f"Azure AI Search failed to index chunk(s): {failed_ids}")

    def hybrid_search(
        self, query: str, query_vector: list[float], top_k: int
    ) -> list[dict]:
        """Combine lexical and vector retrieval, then apply the semantic ranker."""
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="content_vector",
            weight=1.0,
        )
        results = self._search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            query_type="semantic",
            semantic_configuration_name=self._semantic_config,
            query_caption="extractive",
            top=top_k,
            select=[
                "id", "document_id", "title", "source_path", "file_type",
                "content", "section", "page_number",
            ],
        )
        return [dict(result) for result in results]

    def _build_index(self) -> SearchIndex:
        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SimpleField(
                name="document_id",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SimpleField(name="source_path", type=SearchFieldDataType.String),
            SimpleField(name="file_type", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchableField(name="section", type=SearchFieldDataType.String),
            SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self._dimensions,
                vector_search_profile_name="rag-vector-profile",
            ),
        ]
        semantic = SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name=self._semantic_config,
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name="title"),
                        content_fields=[SemanticField(field_name="content")],
                        keywords_fields=[SemanticField(field_name="section")],
                    ),
                )
            ]
        )
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="rag-hnsw")],
            profiles=[
                VectorSearchProfile(
                    name="rag-vector-profile",
                    algorithm_configuration_name="rag-hnsw",
                )
            ],
        )
        return SearchIndex(
            name=self._index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic,
        )

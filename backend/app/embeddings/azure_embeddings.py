"""Generate query and document vectors with an Azure OpenAI embedding deployment."""

from openai import AzureOpenAI

from app.config.settings import Settings


class AzureEmbeddingService:
    """Small adapter around Azure OpenAI's embeddings endpoint."""

    def __init__(self, settings: Settings) -> None:
        self._client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        self._deployment = settings.embedding_deployment
        self._dimensions = settings.embedding_dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of non-empty strings and verify the configured vector size."""
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("Embedding input must not contain empty text.")

        response = self._client.embeddings.create(
            model=self._deployment,
            input=texts,
            dimensions=self._dimensions,
        )
        vectors = [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
        if len(vectors) != len(texts):
            raise RuntimeError("Azure OpenAI returned an unexpected number of embeddings.")
        if any(len(vector) != self._dimensions for vector in vectors):
            raise RuntimeError(
                "Embedding dimensions do not match AZURE_OPENAI_EMBEDDING_DIMENSIONS."
            )
        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Create a vector for a search query."""
        return self.embed_texts([query])[0]

"""Coordinate query embedding, hybrid retrieval, and grounded answer generation."""

from openai import AzureOpenAI

from app.config.settings import Settings
from app.embeddings.azure_embeddings import AzureEmbeddingService
from app.formatting.context_formatter import format_answer_for_chat, format_retrieved_context
from app.search.azure_search_store import AzureSearchStore


class RagService:
    """Answer questions using retrieved source passages and Azure OpenAI."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embeddings = AzureEmbeddingService(settings)
        self._search = AzureSearchStore(settings)
        self._chat_client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        self._search.ensure_index()

    def answer(self, question: str) -> dict:
        """Retrieve relevant passages and generate an answer grounded in them."""
        query_vector = self._embeddings.embed_query(question)
        results = self._search.hybrid_search(
            query=question,
            query_vector=query_vector,
            top_k=self._settings.retrieval_top_k,
        )
        if not results:
            return format_answer_for_chat(
                "I couldn't find relevant information in the indexed documents.",
                [],
            )

        context, sources = format_retrieved_context(results)
        response = self._chat_client.chat.completions.create(
            model=self._settings.chat_deployment,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the user's question using only the supplied source passages. "
                        "Cite factual statements with their bracketed source number, for example [1]. "
                        "If the passages do not contain the answer, say so plainly. "
                        "Do not invent sources or claim unsupported facts."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nRetrieved source passages:\n{context}",
                },
            ],
        )
        answer = response.choices[0].message.content if response.choices else None
        if not answer or not answer.strip():
            raise RuntimeError("Azure OpenAI returned an empty chat response.")
        return format_answer_for_chat(answer, sources)

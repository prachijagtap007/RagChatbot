"""Environment-backed settings and configuration validation."""

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(RuntimeError):
    """Raised when a required service setting has not been configured."""


def _read_int(name: str, default: int) -> int:
    """Parse an integer setting and identify the variable when its value is invalid."""
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer.") from error


@dataclass(frozen=True)
class Settings:
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str
    chat_deployment: str
    embedding_deployment: str
    embedding_dimensions: int
    azure_search_endpoint: str
    azure_search_api_key: str
    search_index_name: str
    semantic_configuration_name: str
    documents_path: Path
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int

    @classmethod
    def from_env(cls) -> "Settings":
        """Read configuration from environment variables, with safe local defaults."""
        settings = cls(
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "").strip(),
            azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", "").strip(),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21").strip(),
            chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "").strip(),
            embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "").strip(),
            embedding_dimensions=_read_int("AZURE_OPENAI_EMBEDDING_DIMENSIONS", 1536),
            azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT", "").strip(),
            azure_search_api_key=os.getenv("AZURE_SEARCH_API_KEY", "").strip(),
            search_index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", "rag-documents").strip(),
            semantic_configuration_name=os.getenv(
                "AZURE_SEARCH_SEMANTIC_CONFIG", "rag-semantic-config"
            ).strip(),
            documents_path=Path(os.getenv("DOCUMENTS_PATH", "./data/documents")).expanduser(),
            chunk_size=_read_int("CHUNK_SIZE", 1200),
            chunk_overlap=_read_int("CHUNK_OVERLAP", 180),
            retrieval_top_k=_read_int("RETRIEVAL_TOP_K", 8),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Catch configuration mistakes before making a cloud request."""
        required = {
            "AZURE_OPENAI_ENDPOINT": self.azure_openai_endpoint,
            "AZURE_OPENAI_API_KEY": self.azure_openai_api_key,
            "AZURE_OPENAI_CHAT_DEPLOYMENT": self.chat_deployment,
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": self.embedding_deployment,
            "AZURE_SEARCH_ENDPOINT": self.azure_search_endpoint,
            "AZURE_SEARCH_API_KEY": self.azure_search_api_key,
            "AZURE_SEARCH_INDEX_NAME": self.search_index_name,
            "AZURE_SEARCH_SEMANTIC_CONFIG": self.semantic_configuration_name,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ConfigurationError(
                "Configure these values in backend/.env before using Azure services: "
                + ", ".join(missing)
            )
        if self.embedding_dimensions <= 0:
            raise ConfigurationError("AZURE_OPENAI_EMBEDDING_DIMENSIONS must be positive.")
        if self.chunk_size <= 0:
            raise ConfigurationError("CHUNK_SIZE must be positive.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ConfigurationError("CHUNK_OVERLAP must be at least 0 and smaller than CHUNK_SIZE.")
        if self.retrieval_top_k <= 0:
            raise ConfigurationError("RETRIEVAL_TOP_K must be positive.")

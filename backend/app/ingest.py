"""Command-line entry point for indexing the configured document folder."""

import logging

from dotenv import load_dotenv

from app.config.settings import Settings
from app.ingestion.ingestion_service import IngestionService

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    settings = Settings.from_env()
    result = IngestionService(settings).ingest_directory()
    logging.getLogger(__name__).info(
        "Ingestion complete: %d document(s), %d chunk(s).",
        result["documents"],
        result["chunks"],
    )


if __name__ == "__main__":
    main()

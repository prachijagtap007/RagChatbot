"""FastAPI application entry point."""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.routes import router

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env")

app = FastAPI(title="RAG Chatbot API", version="1.0.0")
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    """Basic process health check; Azure readiness is checked on the first chat."""
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the existing chat UI from the same origin as the API."""
    return FileResponse(PROJECT_ROOT / "index.html")

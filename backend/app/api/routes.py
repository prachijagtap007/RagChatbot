"""HTTP request and response models for chat."""

from threading import Lock

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config.settings import ConfigurationError, Settings
from app.services.rag_service import RagService

router = APIRouter()
_rag_service: RagService | None = None
_service_lock = Lock()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class SourceResponse(BaseModel):
    citation: int
    title: str
    page_number: int | None = None
    section: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


def _get_rag_service() -> RagService:
    """Create the cloud clients on first use and cache them for later requests."""
    global _rag_service
    if _rag_service is None:
        with _service_lock:
            if _rag_service is None:
                try:
                    _rag_service = RagService(Settings.from_env())
                except ConfigurationError as error:
                    raise HTTPException(status_code=503, detail=str(error)) from error
    return _rag_service


@router.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Return a grounded answer and the documents used to generate it."""
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="Message must not be blank.")
    service = _get_rag_service()
    result = service.answer(request.message.strip())
    return ChatResponse(**result)

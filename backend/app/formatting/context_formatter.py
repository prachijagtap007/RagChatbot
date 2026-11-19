"""Format ranked source passages for the model and the chat client."""

from typing import Any


def format_retrieved_context(results: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Build numbered source context and matching citation metadata."""
    context_parts = []
    sources = []
    for index, result in enumerate(results, start=1):
        title = str(result.get("title") or "Untitled document")
        page_number = result.get("page_number")
        section = result.get("section")
        location = []
        if section:
            location.append(str(section))
        if page_number:
            location.append(f"page {page_number}")
        location_text = f" ({', '.join(location)})" if location else ""
        content = str(result.get("content") or "").strip()
        context_parts.append(f"[{index}] {title}{location_text}\n{content}")
        sources.append(
            {
                "citation": index,
                "title": title,
                "page_number": page_number,
                "section": section or None,
            }
        )
    return "\n\n".join(context_parts), sources


def format_answer_for_chat(answer: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the response shape expected by the HTML chat client."""
    return {"answer": answer.strip(), "sources": sources}

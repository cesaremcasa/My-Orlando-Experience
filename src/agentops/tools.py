from __future__ import annotations

from typing import Any


def cite_chunk(chunk_id: str, tool_context: Any) -> dict[str, Any]:
    """Validate a citation id against retrieved synthetic chunks."""
    chunks: list[dict[str, Any]] = []
    if tool_context is not None:
        state = getattr(tool_context, "state", None)
        if state is not None:
            chunks = state.get("retrieved_chunks") or []
    for chunk in chunks:
        if str(chunk.get("chunk_id")) == str(chunk_id) or str(chunk.get("source_id")) == str(chunk_id):
            return {
                "ok": True,
                "chunk_id": str(chunk_id),
                "source_id": str(chunk.get("source_id") or ""),
                "excerpt": str(chunk.get("text") or "")[:280],
            }
    return {"ok": False, "chunk_id": str(chunk_id)}

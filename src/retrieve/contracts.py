from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RetrievedChunk:
    """Internal retrieval unit; HTTP response compatibility is handled by API."""

    text: str
    source_document: str
    chunk_id: int | str
    score: float


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]: ...

from __future__ import annotations

from typing import Any, Protocol

from src.agentops.errors import VertexNotProvisioned
from src.agentops.memory.store import MemoryRecord
from src.agentops.settings import vertex_config


class VertexMemoryClient(Protocol):
    async def add(
        self,
        *,
        user_id: str,
        session_id: str,
        content: str,
        provenance: str,
        ttl_seconds: int | None = 86_400,
    ) -> str: ...

    async def search(self, *, user_id: str, query: str, top_k: int = 3) -> list[MemoryRecord]: ...

    async def delete_user(self, user_id: str) -> int: ...


class VertexMemoryStore:
    """Optional Vertex AI Memory Bank adapter. Memory Bank is Pre-GA and not provisioned."""

    def __init__(self, client: VertexMemoryClient | None = None) -> None:
        self._client = client
        self.pre_ga = True

    def _client_or_raise(self) -> VertexMemoryClient:
        if self._client is not None:
            return self._client
        _maybe_import_vertex()
        raise VertexNotProvisioned()

    async def add(
        self,
        *,
        user_id: str,
        session_id: str,
        content: str,
        provenance: str,
        ttl_seconds: int | None = 86_400,
    ) -> str:
        return await self._client_or_raise().add(
            user_id=user_id,
            session_id=session_id,
            content=content,
            provenance=provenance,
            ttl_seconds=ttl_seconds,
        )

    async def search(self, *, user_id: str, query: str, top_k: int = 3) -> list[MemoryRecord]:
        return await self._client_or_raise().search(user_id=user_id, query=query, top_k=top_k)

    async def delete_user(self, user_id: str) -> int:
        return await self._client_or_raise().delete_user(user_id)

    async def purge_expired(self, now: float | None = None) -> int:
        del now
        return 0


class FakeVertexMemoryClient:
    """In-process stand-in. No GCP network calls."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    async def add(
        self,
        *,
        user_id: str,
        session_id: str,
        content: str,
        provenance: str,
        ttl_seconds: int | None = 86_400,
    ) -> str:
        del ttl_seconds
        memory_id = f"vertex-{len(self.records) + 1:04d}"
        self.records.append(
            {
                "memory_id": memory_id,
                "user_id": user_id,
                "session_id": session_id,
                "content": content,
                "provenance": provenance,
            }
        )
        return memory_id

    async def search(self, *, user_id: str, query: str, top_k: int = 3) -> list[MemoryRecord]:
        import hashlib
        import time

        hits = [row for row in self.records if row["user_id"] == user_id]
        query_l = query.lower()
        ranked = [row for row in hits if any(part in row["content"].lower() for part in query_l.split())] or hits
        now = time.time()
        return [
            MemoryRecord(
                memory_id=row["memory_id"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                content_hash=hashlib.sha256(row["content"].encode("utf-8")).hexdigest(),
                provenance=row["provenance"],
                score=1.0,
                created_at=now,
                expires_at=None,
                excerpt=row["content"][:80],
            )
            for row in ranked[:top_k]
        ]

    async def delete_user(self, user_id: str) -> int:
        before = len(self.records)
        self.records = [row for row in self.records if row["user_id"] != user_id]
        return before - len(self.records)


def _maybe_import_vertex() -> None:
    cfg = vertex_config()
    if not cfg["complete"]:
        raise VertexNotProvisioned()
    try:
        import google.cloud.aiplatform  # noqa: F401
    except Exception as exc:
        raise VertexNotProvisioned() from exc

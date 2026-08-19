from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from src.agentops.errors import ConfigurationError
from src.agentops.memory.embedder import Embedder, SentenceTransformerEmbedder, cosine
from src.agentops.settings import data_dir


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    user_id: str
    session_id: str
    content_hash: str
    provenance: str
    score: float
    created_at: float
    expires_at: float | None
    excerpt: str


class MemoryStore(Protocol):
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

    async def purge_expired(self, now: float | None = None) -> int: ...


class LocalMemoryStore:
    """SQLite is authoritative. FAISS IndexFlatIP is rebuilt in memory per search."""

    def __init__(
        self,
        root: Path | None = None,
        embedder: Embedder | None = None,
        use_faiss: bool | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else data_dir()
        if not _is_under_data_dir(self.root):
            raise ValueError("memory storage must stay under the AgentOps data directory")
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "memory.sqlite"
        self.embedder = embedder if embedder is not None else SentenceTransformerEmbedder()
        self.use_faiss = True if use_faiss is None else bool(use_faiss)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    deleted_at REAL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id, deleted_at)"
            )
            conn.commit()

    async def add(
        self,
        *,
        user_id: str,
        session_id: str,
        content: str,
        provenance: str,
        ttl_seconds: int | None = 86_400,
    ) -> str:
        return await asyncio.to_thread(
            self._add_sync,
            user_id,
            session_id,
            content,
            provenance,
            ttl_seconds,
        )

    def _add_sync(
        self,
        user_id: str,
        session_id: str,
        content: str,
        provenance: str,
        ttl_seconds: int | None,
    ) -> str:
        now = time.time()
        memory_id = uuid.uuid4().hex
        expires_at = now + ttl_seconds if ttl_seconds is not None else None
        embedding = self.embedder.embed(content)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    id, user_id, session_id, content, content_hash, provenance,
                    embedding, created_at, expires_at, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    memory_id,
                    user_id,
                    session_id,
                    content,
                    content_hash,
                    provenance,
                    json.dumps(embedding),
                    now,
                    expires_at,
                ),
            )
            conn.commit()
        return memory_id

    async def search(self, *, user_id: str, query: str, top_k: int = 3) -> list[MemoryRecord]:
        return await asyncio.to_thread(self._search_sync, user_id, query, top_k)

    def _search_sync(self, user_id: str, query: str, top_k: int) -> list[MemoryRecord]:
        now = time.time()
        query_vec = self.embedder.embed(query)
        rows = self._active_rows(user_id, now)
        if not rows:
            return []
        if self.use_faiss:
            scored = self._search_faiss(query_vec, rows, top_k)
        else:
            scored = []
            for row in rows:
                score = cosine(query_vec, json.loads(row["embedding"]))
                scored.append((score, row))
            scored.sort(key=lambda item: item[0], reverse=True)
            scored = scored[:top_k]
        return [
            MemoryRecord(
                memory_id=row["id"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                content_hash=row["content_hash"],
                provenance=row["provenance"],
                score=float(score),
                created_at=float(row["created_at"]),
                expires_at=row["expires_at"],
                excerpt=row["content"][:80],
            )
            for score, row in scored
        ]

    async def delete_user(self, user_id: str) -> int:
        return await asyncio.to_thread(self._delete_user_sync, user_id)

    def _delete_user_sync(self, user_id: str) -> int:
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE memories SET deleted_at = ?
                WHERE user_id = ? AND deleted_at IS NULL
                """,
                (now, user_id),
            )
            conn.commit()
            changed = cursor.rowcount
        return int(changed)

    async def purge_expired(self, now: float | None = None) -> int:
        return await asyncio.to_thread(self._purge_expired_sync, now)

    def _purge_expired_sync(self, now: float | None) -> int:
        stamp = time.time() if now is None else now
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE memories SET deleted_at = ?
                WHERE deleted_at IS NULL AND expires_at IS NOT NULL AND expires_at <= ?
                """,
                (stamp, stamp),
            )
            conn.commit()
            changed = cursor.rowcount
        return int(changed)

    def _active_rows(self, user_id: str, now: float) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ?
                  AND deleted_at IS NULL
                  AND (expires_at IS NULL OR expires_at > ?)
                """,
                (user_id, now),
            ).fetchall()
        return [dict(row) for row in rows]

    def _search_faiss(
        self,
        query_vec: list[float],
        rows: list[dict],
        top_k: int,
    ) -> list[tuple[float, dict]]:
        np, faiss = _load_faiss()
        matrix = np.asarray(
            [json.loads(row["embedding"]) for row in rows], dtype="float32"
        )
        index = faiss.IndexFlatIP(int(matrix.shape[1]))
        index.add(matrix)
        scores, indices = index.search(
            np.asarray([query_vec], dtype="float32"), min(top_k, len(rows))
        )
        out: list[tuple[float, dict]] = []
        for score, idx in zip(scores[0], indices[0]):
            if int(idx) < 0:
                continue
            out.append((float(score), rows[int(idx)]))
        return out


def _load_faiss() -> tuple[Any, Any]:
    try:
        import numpy as np

        import faiss
    except Exception as exc:
        raise ConfigurationError() from exc
    return np, faiss


def _is_under_data_dir(path: Path) -> bool:
    configured = data_dir().resolve()
    resolved = path.resolve()
    return resolved == configured or configured in resolved.parents

from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from src.agentops.errors import FeedbackError
from src.agentops.settings import data_dir


@dataclass(frozen=True)
class FeedbackRecord:
    beta_user: str
    response_id: str
    session_id: str
    rating: int
    accepted: bool
    updated: bool
    created_at: float
    updated_at: float


class LocalFeedbackStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else data_dir()
        if not _is_under_data_dir(self.root):
            raise ValueError("feedback storage must stay under the AgentOps data directory")
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "feedback.sqlite"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    beta_user TEXT NOT NULL,
                    response_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    accepted INTEGER NOT NULL,
                    reason TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (beta_user, response_id)
                )
                """
            )
            conn.commit()

    async def submit(
        self,
        *,
        beta_user: str,
        response_id: str,
        session_id: str,
        rating: int,
        accepted: bool,
        reason: str | None,
    ) -> FeedbackRecord:
        return await asyncio.to_thread(
            self._submit_sync,
            beta_user,
            response_id,
            session_id,
            rating,
            accepted,
            reason,
        )

    def _submit_sync(
        self,
        beta_user: str,
        response_id: str,
        session_id: str,
        rating: int,
        accepted: bool,
        reason: str | None,
    ) -> FeedbackRecord:
        now = time.time()
        try:
            return self._write_row(
                beta_user, response_id, session_id, rating, accepted, reason, now
            )
        except FeedbackError:
            raise
        except Exception as exc:
            raise FeedbackError() from exc

    def _write_row(
        self,
        beta_user: str,
        response_id: str,
        session_id: str,
        rating: int,
        accepted: bool,
        reason: str | None,
        now: float,
    ) -> FeedbackRecord:
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                """
                SELECT created_at FROM feedback
                WHERE beta_user = ? AND response_id = ?
                """,
                (beta_user, response_id),
            ).fetchone()
            created_at = float(existing[0]) if existing is not None else now
            conn.execute(
                """
                INSERT INTO feedback (
                    beta_user, response_id, session_id, rating, accepted, reason,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(beta_user, response_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    rating = excluded.rating,
                    accepted = excluded.accepted,
                    reason = excluded.reason,
                    updated_at = excluded.updated_at
                """,
                (
                    beta_user,
                    response_id,
                    session_id,
                    rating,
                    int(accepted),
                    reason,
                    created_at,
                    now,
                ),
            )
            conn.commit()
            count = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE beta_user = ? AND response_id = ?",
                (beta_user, response_id),
            ).fetchone()[0]
        if int(count) != 1:
            raise FeedbackError()
        return FeedbackRecord(
            beta_user=beta_user,
            response_id=response_id,
            session_id=session_id,
            rating=rating,
            accepted=accepted,
            updated=existing is not None,
            created_at=created_at,
            updated_at=now,
        )

    def row_count(self, *, beta_user: str, response_id: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE beta_user = ? AND response_id = ?",
                (beta_user, response_id),
            ).fetchone()
        return int(row[0]) if row else 0


def _is_under_data_dir(path: Path) -> bool:
    configured = data_dir().resolve()
    resolved = path.resolve()
    return resolved == configured or configured in resolved.parents

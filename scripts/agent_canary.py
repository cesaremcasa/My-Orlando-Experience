#!/usr/bin/env python3
"""Opt-in real Grok AgentOps canary. Never prints prompt, answer, memory, or key."""

from __future__ import annotations

import hashlib
import os
import sys

from src.agentops.fakes import FixtureRetriever
from src.agentops.memory.embedder import FakeEmbedder
from src.agentops.memory.store import LocalMemoryStore
from src.agentops.runtime import AgentRuntime, reset_runtime
from src.agentops.settings import data_dir
from src.api import main


def main_check() -> int:
    if not os.getenv("XAI_API_KEY"):
        print("agent canary PENDING: XAI_API_KEY is not configured")
        return 0
    reset_runtime(
        AgentRuntime(
            retriever=FixtureRetriever(),
            memory=LocalMemoryStore(root=data_dir(), embedder=FakeEmbedder(), use_faiss=False),
        )
    )
    from fastapi.testclient import TestClient

    with TestClient(main.app) as client:
        response = client.post(
            "/agent/chat",
            json={"session_id": "canary-session", "message": "When does the synthetic fixture open?"},
            headers={"X-Beta-User": "canary-user"},
        )
    if response.status_code != 200:
        raise SystemExit(f"agent canary failed with HTTP {response.status_code}")
    body = response.json()
    digest = hashlib.sha256(body["response"].encode("utf-8")).hexdigest()
    citation_ids = [str(item.get("source_id") or item.get("chunk_id")) for item in body.get("citations", [])]
    print(
        "agent canary PASS:"
        f" response_sha256={digest}"
        f" response_chars={len(body['response'])}"
        f" citation_ids={citation_ids}"
        f" agent_path={body.get('agent_path')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_check())

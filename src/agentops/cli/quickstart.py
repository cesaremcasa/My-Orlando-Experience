from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from src.agentops.fakes import FakeAgentLlm, FixtureRetriever as AgentFixtureRetriever
from src.agentops.memory.embedder import FakeEmbedder
from src.agentops.memory.store import LocalMemoryStore
from src.agentops.runtime import AgentRuntime, reset_runtime
from src.api import main as api_main
from src.respond.providers import FakeProvider
from src.retrieve.contracts import RetrievedChunk


class _QueryFixtureRetriever:
    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        del query, top_k
        return [
            RetrievedChunk(
                text="Synthetic fixture: Magic Kingdom opens at 09:00 for this test context.",
                source_document="synthetic-orlando-fixture",
                source_id="cc0-fixture-001",
                chunk_id="cc0-fixture-001",
                score=1.0,
            )
        ]


def main() -> int:
    root = Path(os.getenv("ORLANDO_AGENTOPS_DATA_DIR") or tempfile.mkdtemp(prefix="orlando-agentops-"))
    os.environ["ORLANDO_AGENTOPS_DATA_DIR"] = str(root)
    api_main.fusion_engine = _QueryFixtureRetriever()
    api_main.provider = FakeProvider()
    reset_runtime(
        AgentRuntime(
            retriever=AgentFixtureRetriever(),
            memory=LocalMemoryStore(root=root, embedder=FakeEmbedder(), use_faiss=False),
            fake_llm=FakeAgentLlm(),
        )
    )
    with TestClient(api_main.app) as client:
        health = client.get("/health")
        query = client.post("/query", json={"question": "When does the fixture open?"})
        agent_health = client.get("/agent/health")
        chat = client.post(
            "/agent/chat",
            json={"session_id": "quickstart-session", "message": "When does the fixture open?"},
            headers={"X-Beta-User": "quickstart-user"},
        )
        if health.status_code != 200 or health.json().get("status") != "healthy":
            raise SystemExit("fake quickstart /health failed")
        if query.status_code != 200:
            raise SystemExit(f"fake quickstart /query failed with HTTP {query.status_code}")
        body = query.json()
        if body["grounding_status"] != "grounded" or not body["citations"]:
            raise SystemExit("fake quickstart did not produce grounded synthetic citations")
        if agent_health.status_code != 200 or agent_health.json().get("status") != "healthy":
            raise SystemExit("fake quickstart /agent/health failed")
        if chat.status_code != 200:
            raise SystemExit(f"fake AgentOps smoke failed with HTTP {chat.status_code}")
        chat_body = chat.json()
        if chat_body["grounding_status"] != "grounded" or not chat_body["citations"]:
            raise SystemExit("fake AgentOps smoke did not produce grounded synthetic citations")
        feedback = client.post(
            "/feedback",
            json={
                "response_id": chat_body["response_id"],
                "session_id": "quickstart-session",
                "rating": 5,
                "accepted": True,
            },
            headers={"X-Beta-User": "beta-001"},
        )
        if feedback.status_code != 200:
            raise SystemExit(f"fake feedback smoke failed with HTTP {feedback.status_code}")
    print("fake quickstart PASS: /health, /query, /agent/health, /agent/chat, /feedback")
    return 0

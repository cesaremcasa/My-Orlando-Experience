"""Opt-in real Grok AgentOps canary. Never prints prompt, answer, memory, or key."""

from __future__ import annotations

import hashlib
import os


def main() -> int:
    if not os.getenv("XAI_API_KEY"):
        print("Grok canary PENDING: XAI_API_KEY not present in process environment.")
        return 0
    return _run()


def _run() -> int:
    import time

    from fastapi.testclient import TestClient

    from src.agentops.fakes import FixtureRetriever
    from src.agentops.memory.embedder import FakeEmbedder
    from src.agentops.memory.store import LocalMemoryStore
    from src.agentops.runtime import AgentRuntime, reset_runtime
    from src.agentops.settings import data_dir, xai_model
    from src.api import main as api_main

    reset_runtime(
        AgentRuntime(
            retriever=FixtureRetriever(),
            memory=LocalMemoryStore(root=data_dir(), embedder=FakeEmbedder(), use_faiss=False),
        )
    )
    started = time.perf_counter()
    with TestClient(api_main.app) as client:
        response = client.post(
            "/agent/chat",
            json={
                "session_id": "canary-session",
                "message": "When does the synthetic fixture open?",
            },
            headers={"X-Beta-User": "canary-user"},
        )
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    if response.status_code != 200:
        category = "FAIL"
        if response.status_code in {401, 403}:
            category = "FAIL"
        print(
            f"Grok canary {category}: http_status={response.status_code} "
            f"model={xai_model()} latency_ms={latency_ms} "
            "error=provider_http"
        )
        return 1
    body = response.json()
    digest = hashlib.sha256(str(body.get("response") or "").encode("utf-8")).hexdigest()
    citation_ids = [
        str(item.get("source_id") or item.get("chunk_id")) for item in body.get("citations", [])
    ]
    print(
        "Grok canary PASS:"
        f" model={xai_model()}"
        f" response_sha256={digest}"
        f" response_chars={len(str(body.get('response') or ''))}"
        f" citation_ids={citation_ids}"
        f" agent_path={body.get('agent_path')}"
        f" grounding={body.get('grounding_status')}"
        f" safety={'pass' if body.get('grounding_status') == 'grounded' else 'abstain'}"
        f" trace_id={body.get('trace_id')}"
        f" latency_ms={latency_ms}"
    )
    return 0

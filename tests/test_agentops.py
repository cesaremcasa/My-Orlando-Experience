from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from src.agentops.fakes import FakeAgentLlm, FixtureRetriever
from src.agentops.memory.embedder import FakeEmbedder
from src.agentops.memory.store import LocalMemoryStore
from src.agentops.runtime import AgentRuntime, reset_runtime
from src.agentops.settings import xai_model
from src.agentops.tools import cite_chunk
from src.api import main
from src.retrieve.contracts import RetrievedChunk

REPO_ROOT = Path(__file__).resolve().parents[1]
HEADERS = {"X-Beta-User": "beta-alice"}


def _runtime(tmp_path: Path, **kwargs) -> tuple[AgentRuntime, FixtureRetriever, LocalMemoryStore, FakeAgentLlm]:
    retriever = kwargs.pop("retriever", FixtureRetriever())
    memory = kwargs.pop(
        "memory",
        LocalMemoryStore(root=tmp_path, embedder=FakeEmbedder(), use_faiss=False),
    )
    fake_llm = kwargs.pop("fake_llm", FakeAgentLlm())
    runtime = reset_runtime(
        AgentRuntime(retriever=retriever, memory=memory, fake_llm=fake_llm, **kwargs)
    )
    main.fusion_engine = retriever
    return runtime, retriever, memory, fake_llm


@asynccontextmanager
async def _async_client():
    async with main.app.router.lifespan_context(main.app):
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


def test_topology_is_real_sequential_parallel(tmp_path, monkeypatch):
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    runtime, _, _, _ = _runtime(tmp_path)
    runtime.ensure()
    from src.agentops.agents import inspect_topology

    info = inspect_topology(runtime.root)
    assert info["root_is_sequential"]
    assert info["parallel_is_parallel"]
    assert info["children"] == ["context_parallel", "response_agent", "safety_agent"]
    assert info["parallel_children"] == ["retrieval_agent", "memory_agent"]
    assert info["retrieval_is_base"] and info["memory_is_base"]
    assert info["response_is_llm"] and info["safety_is_llm"]
    assert info["safety_schema"] == "SafetyVerdict"
    assert "cite_chunk" in info["response_tools"]


def test_structured_output_and_cite_tool():
    chunk = {
        "chunk_id": "cc0-fixture-001",
        "source_id": "cc0-fixture-001",
        "text": "Synthetic fixture: Magic Kingdom opens at 09:00 for this test context.",
    }

    class _State(dict):
        pass

    class _Ctx:
        state = _State(retrieved_chunks=[chunk])

    found = cite_chunk("cc0-fixture-001", _Ctx())
    missing = cite_chunk("nope", _Ctx())
    assert found["ok"] is True
    assert missing["ok"] is False


def test_agent_chat_grounded_path_and_event_order(tmp_path, monkeypatch):
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    _, _, _, _ = _runtime(tmp_path)
    body = TestClient(main.app).post(
        "/agent/chat",
        json={"session_id": "s-order", "message": "When does the fixture open?"},
        headers=HEADERS,
    ).json()
    assert body["grounding_status"] == "grounded"
    assert body["citations"][0]["source_id"] == "cc0-fixture-001"
    path = body["agent_path"]
    assert path.index("retrieval_agent") < path.index("response_agent")
    assert path.index("memory_agent") < path.index("response_agent")
    assert path.index("response_agent") < path.index("safety_agent")
    assert body["trace_id"] is None


def test_no_evidence_abstains_and_skips_memory_write(tmp_path, monkeypatch):
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    _, retriever, memory, _ = _runtime(tmp_path, retriever=FixtureRetriever([]))
    body = TestClient(main.app).post(
        "/agent/chat",
        json={"session_id": "s-empty", "message": "unknown topic", "remember": True},
        headers=HEADERS,
    ).json()
    assert body["grounding_status"] == "abstained"
    assert body["citations"] == []
    assert retriever.calls == 1
    hits = asyncio.run(memory.search(user_id="beta-alice", query="unknown topic"))
    assert hits == []


def test_safety_fail_does_not_write_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    fake = FakeAgentLlm(
        safety={"verdict": "FAIL", "reason": "ungrounded", "schema_ok": True, "grounded": False}
    )
    _, _, memory, _ = _runtime(tmp_path, fake_llm=fake)
    body = TestClient(main.app).post(
        "/agent/chat",
        json={"session_id": "s-fail", "message": "When does the fixture open?", "remember": True},
        headers=HEADERS,
    ).json()
    assert body["grounding_status"] == "abstained"
    hits = asyncio.run(memory.search(user_id="beta-alice", query="fixture"))
    assert hits == []


def test_memory_persists_expires_soft_deletes_and_isolates(tmp_path, monkeypatch):
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    store = LocalMemoryStore(root=tmp_path, embedder=FakeEmbedder(), use_faiss=False)
    added = asyncio.run(
        store.add(
            user_id="alice",
            session_id="shared",
            content="alice remembers the synthetic park hours fixture",
            provenance="response",
            ttl_seconds=60,
        )
    )
    asyncio.run(
        store.add(
            user_id="bob",
            session_id="shared",
            content="bob remembers a different synthetic snack stand",
            provenance="response",
            ttl_seconds=60,
        )
    )
    alice = asyncio.run(store.search(user_id="alice", query="park hours"))
    bob = asyncio.run(store.search(user_id="bob", query="park hours"))
    assert added in [item.memory_id for item in alice]
    assert all(item.user_id == "alice" for item in alice)
    assert all("snack" not in item.excerpt.lower() for item in alice)
    assert all(item.user_id == "bob" for item in bob)
    expired = asyncio.run(
        store.add(
            user_id="alice",
            session_id="shared",
            content="expired synthetic note",
            provenance="response",
            ttl_seconds=0,
        )
    )
    still = asyncio.run(store.search(user_id="alice", query="expired synthetic note"))
    assert expired not in [item.memory_id for item in still]
    purged = asyncio.run(store.purge_expired())
    assert purged >= 1
    deleted = asyncio.run(store.delete_user("alice"))
    assert deleted >= 1
    assert asyncio.run(store.search(user_id="alice", query="park hours")) == []
    assert asyncio.run(store.search(user_id="bob", query="snack")) 


def test_cross_user_chat_does_not_share_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    _, _, memory, _ = _runtime(tmp_path)
    client = TestClient(main.app)
    first = client.post(
        "/agent/chat",
        json={"session_id": "same-session", "message": "When does the fixture open?", "remember": True},
        headers={"X-Beta-User": "user-a"},
    ).json()
    assert first["grounding_status"] == "grounded"
    second = client.post(
        "/agent/chat",
        json={"session_id": "same-session", "message": "When does the fixture open?"},
        headers={"X-Beta-User": "user-b"},
    ).json()
    a_hits = asyncio.run(memory.search(user_id="user-a", query="fixture"))
    b_hits = asyncio.run(memory.search(user_id="user-b", query="fixture"))
    assert a_hits
    assert not b_hits or all(item.user_id == "user-b" for item in b_hits)
    assert second["grounding_status"] == "grounded"


def test_sanitized_provider_retrieval_memory_and_schema_failures(tmp_path, monkeypatch):
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    secret = "TOP-SECRET-AGENT-TOKEN"

    class BoomRetriever:
        def retrieve(self, query: str, top_k: int = 3):
            raise RuntimeError(secret)

    _, _, _, _ = _runtime(tmp_path, retriever=BoomRetriever())
    retrieval = TestClient(main.app).post(
        "/agent/chat",
        json={"session_id": "s-retr", "message": "When does the fixture open?"},
        headers=HEADERS,
    )
    assert retrieval.status_code == 500
    assert retrieval.json() == {"detail": "Retrieval failed."}
    assert secret not in retrieval.text

    class BoomMemory(LocalMemoryStore):
        async def search(self, *, user_id: str, query: str, top_k: int = 3):
            raise RuntimeError(secret)

    memory = BoomMemory(root=tmp_path, embedder=FakeEmbedder(), use_faiss=False)
    _, _, _, _ = _runtime(tmp_path, memory=memory)
    mem = TestClient(main.app).post(
        "/agent/chat",
        json={"session_id": "s-mem", "message": "When does the fixture open?"},
        headers=HEADERS,
    )
    assert mem.status_code == 500
    assert mem.json() == {"detail": "Memory operation failed."}
    assert secret not in mem.text

    _, _, _, _ = _runtime(tmp_path, fake_llm=FakeAgentLlm(fail_with=RuntimeError(secret)))
    provider = TestClient(main.app).post(
        "/agent/chat",
        json={"session_id": "s-prov", "message": "When does the fixture open?"},
        headers=HEADERS,
    )
    assert provider.status_code == 502
    assert provider.json() == {"detail": "Response provider unavailable"}
    assert secret not in provider.text

    _, _, _, _ = _runtime(tmp_path, fake_llm=FakeAgentLlm(response={"nope": True}))
    schema = TestClient(main.app).post(
        "/agent/chat",
        json={"session_id": "s-schema", "message": "When does the fixture open?"},
        headers=HEADERS,
    )
    assert schema.status_code == 200
    assert schema.json()["grounding_status"] == "abstained"
    assert secret not in schema.text
    assert "ValidationError" not in schema.text


def test_timeout_is_sanitized(tmp_path, monkeypatch):
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    hang = pytest.importorskip("threading").Event()

    class HangingRetriever:
        def retrieve(self, query: str, top_k: int = 3):
            hang.wait(2)
            return []

    monkeypatch.setattr(main, "AGENT_TIMEOUT_SECONDS", 0.05)
    _runtime(tmp_path, retriever=HangingRetriever())
    try:
        failed = TestClient(main.app).post(
            "/agent/chat",
            json={"session_id": "s-time", "message": "When does the fixture open?"},
            headers=HEADERS,
        )
        assert failed.status_code == 504
        assert failed.json() == {"detail": "Agent timed out."}
        assert "Traceback" not in failed.text
    finally:
        hang.set()


def test_agent_cpu_work_does_not_block_event_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    retriever = FixtureRetriever(delay_s=0.15)
    _runtime(tmp_path, retriever=retriever)
    monkeypatch.setattr(main, "AGENT_TIMEOUT_SECONDS", 30.0)
    monkeypatch.setattr(main, "AGENT_MAX_CONCURRENCY", 8)

    async def run() -> None:
        stop = asyncio.Event()
        ticks: list[float] = []

        async def heartbeat() -> None:
            while not stop.is_set():
                ticks.append(time.monotonic())
                try:
                    await asyncio.wait_for(stop.wait(), timeout=0.03)
                except TimeoutError:
                    continue

        hb_task = asyncio.create_task(heartbeat())
        try:
            async with _async_client() as client:
                started = time.monotonic()
                responses = await asyncio.gather(
                    *[
                        client.post(
                            "/agent/chat",
                            json={"session_id": f"s-{index}", "message": "When does the fixture open?"},
                            headers={"X-Beta-User": f"user-{index}"},
                        )
                        for index in range(6)
                    ]
                )
                finished = time.monotonic()
        finally:
            stop.set()
            await hb_task

        assert all(item.status_code == 200 for item in responses)
        during = [tick for tick in ticks if started <= tick <= finished]
        assert len(during) >= 4
        if len(during) >= 2:
            gaps = [later - earlier for earlier, later in zip(during, during[1:])]
            assert max(gaps) < 0.2

    asyncio.run(run())


def test_import_and_both_health_endpoints_need_no_grok_faiss_phoenix_or_gcp():
    env = os.environ.copy()
    for key in (
        "XAI_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "PHOENIX_COLLECTOR_ENDPOINT",
        "PHOENIX_API_KEY",
    ):
        env.pop(key, None)
    env["PYTHONPATH"] = str(REPO_ROOT)
    script = """
import sys
import src.api.main as api_main
from fastapi.testclient import TestClient

blocked = (
    "faiss",
    "google.adk",
    "google.adk.agents",
    "litellm",
    "phoenix",
    "arize.phoenix",
    "arize.phoenix.otel",
    "google.cloud.aiplatform",
    "vertexai",
)
imported = [name for name in blocked if name in sys.modules]
assert imported == [], imported
client = TestClient(api_main.app)
health = client.get("/health")
agent_health = client.get("/agent/health")
assert health.status_code == 200
assert health.json() == {"status": "healthy", "engine_ready": False}
assert agent_health.status_code == 200
assert agent_health.json()["status"] == "healthy"
assert agent_health.json()["agent_ready"] is False
imported = [name for name in blocked if name in sys.modules]
assert imported == [], imported
print("import-health-ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "import-health-ok" in completed.stdout


def test_cc0_fixture_still_synthetic():
    chunk = RetrievedChunk(
        "Synthetic fixture: Magic Kingdom opens at 09:00 for this test context.",
        "synthetic-orlando-fixture",
        "cc0-fixture-001",
        1.0,
        "cc0-fixture-001",
    )
    assert "Disney" not in chunk.text
    assert chunk.source_id == "cc0-fixture-001"


def test_agent_chat_requires_beta_user_and_bounds(tmp_path, monkeypatch):
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    _runtime(tmp_path)
    client = TestClient(main.app)
    missing = client.post("/agent/chat", json={"session_id": "s1", "message": "When does the fixture open?"})
    assert missing.status_code == 422
    blank = client.post(
        "/agent/chat",
        json={"session_id": "   ", "message": "When does the fixture open?"},
        headers=HEADERS,
    )
    assert blank.status_code == 422
    long_session = client.post(
        "/agent/chat",
        json={"session_id": "s" * 129, "message": "When does the fixture open?"},
        headers=HEADERS,
    )
    assert long_session.status_code == 422
    long_message = client.post(
        "/agent/chat",
        json={"session_id": "s1", "message": "m" * 2001},
        headers=HEADERS,
    )
    assert long_message.status_code == 422
    ok = client.post(
        "/agent/chat",
        json={"session_id": "s" * 128, "message": "When does the fixture open?"},
        headers=HEADERS,
    )
    assert ok.status_code == 200


def test_query_and_health_contracts_unchanged(tmp_path, monkeypatch):
    from src.respond.providers import FakeProvider

    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    _runtime(tmp_path)
    monkeypatch.setattr(main, "provider", FakeProvider())
    client = TestClient(main.app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "healthy", "engine_ready": True}
    query = client.post("/query", json={"question": "When does the fixture open?"})
    assert query.status_code == 200
    body = query.json()
    assert {"response", "grounding_score", "latency_ms", "sources", "citations", "grounding_status"} <= set(body)


def test_production_topology_uses_xai_litellm(tmp_path, monkeypatch):
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    memory = LocalMemoryStore(root=tmp_path, embedder=FakeEmbedder(), use_faiss=False)
    runtime = reset_runtime(AgentRuntime(retriever=FixtureRetriever(), memory=memory))
    runtime.ensure()
    model_name = str(getattr(runtime.model, "model", ""))
    assert model_name.startswith("xai/")
    assert "gemini" not in model_name.lower()
    assert xai_model() in model_name
    from google.adk.models.lite_llm import LiteLlm

    assert isinstance(runtime.model, LiteLlm)


def test_production_runtime_selects_lazy_real_embedder(tmp_path, monkeypatch):
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    runtime = reset_runtime(AgentRuntime(retriever=FixtureRetriever(), fake_llm=FakeAgentLlm()))
    runtime.ensure()
    from src.agentops.memory.embedder import SentenceTransformerEmbedder

    assert isinstance(runtime.memory.embedder, SentenceTransformerEmbedder)
    assert runtime.memory.embedder._model is None
    assert runtime.memory.use_faiss is True


def test_memory_persists_across_store_instances(tmp_path, monkeypatch):
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    first = LocalMemoryStore(root=tmp_path, embedder=FakeEmbedder(), use_faiss=False)
    memory_id = asyncio.run(
        first.add(
            user_id="alice",
            session_id="shared",
            content="alice remembers the synthetic park hours fixture",
            provenance="response",
        )
    )
    second = LocalMemoryStore(root=tmp_path, embedder=FakeEmbedder(), use_faiss=False)
    hits = asyncio.run(second.search(user_id="alice", query="park hours"))
    assert memory_id in [item.memory_id for item in hits]
    assert not (tmp_path / "memory.faiss").exists()
    assert (tmp_path / "memory.sqlite").exists()


def test_faiss_similarity_with_fake_embedder_filters_users(tmp_path, monkeypatch):
    pytest.importorskip("faiss")
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    store = LocalMemoryStore(root=tmp_path, embedder=FakeEmbedder(), use_faiss=True)
    asyncio.run(
        store.add(
            user_id="alice",
            session_id="s",
            content="synthetic park hours fixture at Magic Kingdom",
            provenance="response",
        )
    )
    asyncio.run(
        store.add(
            user_id="alice",
            session_id="s",
            content="alice likes synthetic pretzels at a snack stand",
            provenance="response",
        )
    )
    asyncio.run(
        store.add(
            user_id="bob",
            session_id="s",
            content="synthetic park hours fixture at Magic Kingdom",
            provenance="response",
        )
    )
    alice = asyncio.run(store.search(user_id="alice", query="park hours fixture", top_k=2))
    bob = asyncio.run(store.search(user_id="bob", query="park hours fixture", top_k=2))
    assert alice
    assert all(item.user_id == "alice" for item in alice)
    assert "park" in alice[0].excerpt.lower()
    assert all(item.user_id == "bob" for item in bob)
    assert {item.memory_id for item in alice}.isdisjoint({item.memory_id for item in bob})
    assert not (tmp_path / "memory.faiss").exists()


def test_missing_runtime_extras_are_sanitized(tmp_path, monkeypatch):
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    from src.agentops.errors import ConfigurationError

    class MissingExtrasMemory(LocalMemoryStore):
        async def search(self, *, user_id: str, query: str, top_k: int = 3):
            raise ConfigurationError()

    memory = MissingExtrasMemory(root=tmp_path, embedder=FakeEmbedder(), use_faiss=False)
    _runtime(tmp_path, memory=memory)
    response = TestClient(main.app).post(
        "/agent/chat",
        json={"session_id": "s-extras", "message": "When does the fixture open?"},
        headers=HEADERS,
    )
    assert response.status_code == 500
    assert response.json() == {
        "detail": 'AgentOps runtime requires extras. Install with: pip install "my-orlando-experience[agentops]"'
    }
    assert "ImportError" not in response.text
    assert "Traceback" not in response.text


def test_invalid_safety_schema_abstains(tmp_path, monkeypatch):
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    _runtime(tmp_path, fake_llm=FakeAgentLlm(safety={"nope": True}))
    body = TestClient(main.app).post(
        "/agent/chat",
        json={"session_id": "s-safety-schema", "message": "When does the fixture open?", "remember": True},
        headers=HEADERS,
    ).json()
    assert body["grounding_status"] == "abstained"


def test_agent_concurrency_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    retriever = FixtureRetriever(delay_s=0.2)
    _runtime(tmp_path, retriever=retriever)
    monkeypatch.setattr(main, "AGENT_TIMEOUT_SECONDS", 30.0)
    monkeypatch.setattr(main, "AGENT_MAX_CONCURRENCY", 1)

    async def run() -> None:
        async with _async_client() as client:
            started = time.monotonic()
            responses = await asyncio.gather(
                client.post(
                    "/agent/chat",
                    json={"session_id": "s-a", "message": "When does the fixture open?"},
                    headers={"X-Beta-User": "user-a"},
                ),
                client.post(
                    "/agent/chat",
                    json={"session_id": "s-b", "message": "When does the fixture open?"},
                    headers={"X-Beta-User": "user-b"},
                ),
            )
            elapsed = time.monotonic() - started
        assert all(item.status_code == 200 for item in responses)
        assert elapsed >= 0.35

    asyncio.run(run())


def test_agent_canary_redacts_output_and_is_pending_without_key():
    source = (REPO_ROOT / "src" / "agentops" / "cli" / "grok_canary.py").read_text(encoding="utf-8")
    assert "response_sha256" in source
    assert "response_chars" in source
    assert "citation_ids" in source
    assert "agent_path" in source
    assert "os.getenv(\"XAI_API_KEY\")" in source
    env = os.environ.copy()
    env.pop("XAI_API_KEY", None)
    env["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "agent_canary.py")],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PENDING" in completed.stdout
    assert "Synthetic fixture" not in completed.stdout
    assert "When does" not in completed.stdout

from __future__ import annotations

import asyncio
import importlib.util
import os
import tempfile
from pathlib import Path


def main() -> int:
    uid = os.getuid()
    print(f"uid={uid}")
    if os.getenv("ORLANDO_CONTAINER_EVIDENCE") == "1" and uid != 10001:
        raise SystemExit(f"expected UID 10001, got {uid}")
    for name in ("sentence_transformers", "faiss"):
        if importlib.util.find_spec(name) is None:
            raise SystemExit(f"missing dependency {name}")
        print(f"installed={name}")
    if importlib.util.find_spec("google.cloud.aiplatform") is not None:
        raise SystemExit("default image must not include google.cloud.aiplatform")
    print("gcp_extra_absent=true")
    cache = Path.home() / ".cache" / "huggingface"
    snapshots = list(cache.rglob("snapshots")) if cache.exists() else []
    if snapshots:
        raise SystemExit("huggingface model snapshots present; image build must not download models")
    print("no_model_snapshots=true")

    import sys

    before = set(sys.modules)
    import src.api.main as api_main
    loaded = sorted(name for name in ("faiss", "google.adk", "litellm") if name in sys.modules and name not in before)
    if loaded:
        raise SystemExit(f"importing src.api.main loaded {loaded}")
    from src.agentops.memory.embedder import FakeEmbedder, SentenceTransformerEmbedder
    from src.agentops.memory.store import LocalMemoryStore

    embedder = SentenceTransformerEmbedder()
    if embedder._model is not None:
        raise SystemExit("SentenceTransformer model loaded unexpectedly")
    print("st_model_unloaded=true")
    print("api_import_lazy=true")

    root = Path(os.getenv("ORLANDO_AGENTOPS_DATA_DIR") or tempfile.mkdtemp(prefix="orlando-mem-"))
    os.environ.setdefault("ORLANDO_AGENTOPS_DATA_DIR", str(root))
    store = LocalMemoryStore(root=root, embedder=FakeEmbedder(), use_faiss=True)
    memory_id = asyncio.run(
        store.add(
            user_id="beta-001",
            session_id="container",
            content="synthetic park hours fixture",
            provenance="response",
        )
    )
    hits = asyncio.run(store.search(user_id="beta-001", query="park hours fixture", top_k=1))
    if not hits or hits[0].memory_id != memory_id:
        raise SystemExit("FAISS add/search with FakeEmbedder failed")
    print("faiss_fake_embedder_search=ok")

    if os.getenv("ORLANDO_FAKE_RUNTIME") == "1" or os.getenv("ORLANDO_CONTAINER_EVIDENCE") == "1":
        os.environ["ORLANDO_FAKE_RUNTIME"] = "1"
        from fastapi.testclient import TestClient

        from src.agentops.fakes import FakeAgentLlm, FixtureRetriever
        from src.agentops.runtime import AgentRuntime, reset_runtime
        from src.respond.providers import FakeProvider

        api_main.fusion_engine = FixtureRetriever()
        api_main.provider = FakeProvider()
        reset_runtime(
            AgentRuntime(
                retriever=FixtureRetriever(),
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
                json={"session_id": "container-session", "message": "When does the fixture open?"},
                headers={"X-Beta-User": "beta-001"},
            )
            if health.status_code != 200:
                raise SystemExit("/health failed")
            if query.status_code != 200:
                raise SystemExit("/query failed")
            if agent_health.status_code != 200:
                raise SystemExit("/agent/health failed")
            if chat.status_code != 200:
                raise SystemExit("/agent/chat failed")
            feedback = client.post(
                "/feedback",
                json={
                    "response_id": chat.json()["response_id"],
                    "session_id": "container-session",
                    "rating": 5,
                    "accepted": True,
                },
                headers={"X-Beta-User": "beta-001"},
            )
            if feedback.status_code != 200:
                raise SystemExit("/feedback failed")
        print("http_fake_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

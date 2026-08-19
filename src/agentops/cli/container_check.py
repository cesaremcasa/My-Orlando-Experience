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

    from src.agentops.memory.embedder import FakeEmbedder, SentenceTransformerEmbedder
    from src.agentops.memory.store import LocalMemoryStore

    embedder = SentenceTransformerEmbedder()
    if embedder._model is not None:
        raise SystemExit("SentenceTransformer model loaded unexpectedly")
    print("st_model_unloaded=true")

    root = Path(os.getenv("ORLANDO_AGENTOPS_DATA_DIR") or tempfile.mkdtemp(prefix="orlando-mem-"))
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

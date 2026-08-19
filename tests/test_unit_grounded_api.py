from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api import main
from src.respond.providers import FakeProvider
from src.retrieve.contracts import RetrievedChunk


class FakeRetriever:
    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        assert query == "When does Magic Kingdom open?"
        assert top_k == 3
        return [
            RetrievedChunk(
                text="Magic Kingdom opens at 9:00 during December.",
                source_document="Orlando_Park_Hours_Dez2025.pdf",
                chunk_id=7,
                score=0.12,
            )
        ]


def test_query_schema_and_health_without_faiss_or_openai(monkeypatch):
    monkeypatch.setattr(main, "fusion_engine", FakeRetriever())
    monkeypatch.setattr(main, "provider", FakeProvider())
    client = TestClient(main.app)

    health = client.get("/health")
    response = client.post("/query", json={"question": "When does Magic Kingdom open?"})

    assert health.status_code == 200
    assert health.json() == {"status": "healthy", "engine_ready": True}
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"response", "grounding_score", "latency_ms", "sources"}
    assert body["sources"] == ["📌 Source: Magic Kingdom opens"]
    assert body["grounding_score"] > 0


def test_openai_adapter_is_lazy_and_does_not_require_key_at_import(monkeypatch):
    from src.respond.providers import OpenAIProvider

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIProvider()
    try:
        provider.generate(question="q", context="c")
    except RuntimeError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:  # pragma: no cover - provider must fail closed without a key
        raise AssertionError("provider unexpectedly generated without an API key")


def test_retrieved_chunk_preserves_internal_metadata():
    chunk = RetrievedChunk("text", "guide.pdf", "chunk-4", 0.42)
    assert (chunk.text, chunk.source_document, chunk.chunk_id, chunk.score) == (
        "text",
        "guide.pdf",
        "chunk-4",
        0.42,
    )

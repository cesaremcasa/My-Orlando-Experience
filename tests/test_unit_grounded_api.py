from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api import main
from src.respond.providers import FakeProvider
from src.retrieve.contracts import RetrievedChunk


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk] | None = None):
        self.chunks = [
            RetrievedChunk(
                text="Magic Kingdom opens at 9:00 during December.",
                source_document="Orlando_Park_Hours_Dez2025.pdf",
                source_id="park-hours",
                chunk_id=7,
                score=0.12,
            )
        ] if chunks is None else chunks

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        assert top_k == 3
        return self.chunks


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
    assert {"response", "grounding_score", "latency_ms", "sources"}.issubset(body)
    assert body["sources"] == ["park-hours"]
    assert body["grounding_status"] == "grounded"
    assert body["citations"][0] == {
        "document": "Orlando_Park_Hours_Dez2025.pdf",
        "source_id": "park-hours",
        "chunk_id": 7,
        "excerpt": "Magic Kingdom opens at 9:00 during December.",
        "score": 0.12,
    }
    assert body["grounding_score"] > 0


def test_citations_dedupe_sources_and_bound_excerpt(monkeypatch):
    chunks = [
        RetrievedChunk("A" * 500, "guide.pdf", 1, 0.2, "guide"),
        RetrievedChunk("duplicate", "guide.pdf", 1, 0.1, "guide"),
        RetrievedChunk("Second fact", "other.pdf", 2, 0.3, "other"),
    ]
    class ContextProvider:
        def generate(self, *, question: str, context: str) -> str:
            return context

    monkeypatch.setattr(main, "fusion_engine", FakeRetriever(chunks))
    monkeypatch.setattr(main, "provider", ContextProvider())
    body = TestClient(main.app).post("/query", json={"question": "What is the guide?"}).json()
    assert body["sources"] == ["guide", "other"]
    assert len(body["citations"]) == 2
    assert len(body["citations"][0]["excerpt"]) == 280


def test_empty_retrieval_abstains_without_calling_provider(monkeypatch):
    class ExplodingProvider:
        def generate(self, **_kwargs):
            raise AssertionError("provider must not run without context")

    monkeypatch.setattr(main, "fusion_engine", FakeRetriever([]))
    monkeypatch.setattr(main, "provider", ExplodingProvider())
    body = TestClient(main.app).post("/query", json={"question": "unknown"}).json()
    assert body["grounding_status"] == "abstained"
    assert body["response"] == main.ABSTENTION_RESPONSE
    assert body["sources"] == [] and body["citations"] == []


def test_low_grounding_abstains_and_provider_error_is_502(monkeypatch):
    class WeakProvider:
        def generate(self, **_kwargs):
            return "unrelated answer"

    class BrokenProvider:
        def generate(self, **_kwargs):
            raise RuntimeError("secret provider detail")

    monkeypatch.setattr(main, "fusion_engine", FakeRetriever())
    monkeypatch.setattr(main, "provider", WeakProvider())
    weak = TestClient(main.app).post("/query", json={"question": "What is unrelated?"})
    assert weak.status_code == 200
    assert weak.json()["grounding_status"] == "abstained"
    assert weak.json()["sources"] == []

    monkeypatch.setattr(main, "provider", BrokenProvider())
    failed = TestClient(main.app).post("/query", json={"question": "What is this?"})
    assert failed.status_code == 502
    assert failed.json()["detail"] == "Response provider unavailable"
    assert "secret provider detail" not in failed.text


def test_retrieval_error_does_not_leak_exception_or_path(monkeypatch):
    secret = "TOP-SECRET-RETRIEVAL-TOKEN"
    leaked_path = "/secret/data/index/faiss.index"

    class BrokenRetriever:
        def retrieve(self, query: str, top_k: int = 3):
            raise RuntimeError(f"{secret}: {leaked_path}")

    monkeypatch.setattr(main, "fusion_engine", BrokenRetriever())
    monkeypatch.setattr(main, "provider", FakeProvider())
    failed = TestClient(main.app).post("/query", json={"question": "When does Magic Kingdom open?"})
    assert failed.status_code == 500
    assert failed.json() == {"detail": "Retrieval failed."}
    assert secret not in failed.text
    assert leaked_path not in failed.text
    assert "RuntimeError" not in failed.text


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


def test_cc0_fixture_is_small_and_structured():
    fixture = json.loads(
        (Path(__file__).resolve().parents[1] / "tests/fixtures/synthetic_context.json").read_text()
    )
    assert fixture["license"] == "CC0-1.0"
    assert fixture["source_id"] == "cc0-fixture-001"
    assert len(fixture["text"]) < 200

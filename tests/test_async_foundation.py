from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.respond.providers import FakeProvider
from src.retrieve.contracts import RetrievedChunk


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CHUNK = RetrievedChunk(
    text="Magic Kingdom opens at 9:00 during December.",
    source_document="Orlando_Park_Hours_Dez2025.pdf",
    source_id="park-hours",
    chunk_id=7,
    score=0.12,
)


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk] | None = None, delay_s: float = 0.0):
        self.chunks = [FIXTURE_CHUNK] if chunks is None else chunks
        self.delay_s = delay_s
        self.calls = 0
        self._lock = threading.Lock()

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        del query
        assert top_k == 3
        with self._lock:
            self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.chunks


class DelayedFakeProvider(FakeProvider):
    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s

    def generate(self, *, question: str, context: str) -> str:
        if self.delay_s:
            time.sleep(self.delay_s)
        return super().generate(question=question, context=context)


@asynccontextmanager
async def _async_client():
    async with main.app.router.lifespan_context(main.app):
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


def test_health_is_compatible_when_engine_is_uninitialized(monkeypatch):
    monkeypatch.setattr(main, "fusion_engine", None)
    health = TestClient(main.app).get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "healthy", "engine_ready": False}


def test_missing_assets_return_sanitized_503(monkeypatch):
    monkeypatch.setattr(main, "fusion_engine", None)

    def boom():
        raise FileNotFoundError("data/index/faiss.index")

    monkeypatch.setattr(main, "get_fusion_engine", boom)
    response = TestClient(main.app).post("/query", json={"question": "When does Magic Kingdom open?"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Fusion Engine not initialized."}
    assert "faiss" not in response.text.lower()
    assert "FileNotFoundError" not in response.text


def test_import_and_health_require_no_faiss_grok_phoenix_or_gcp():
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

assert api_main.fusion_engine is None
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
health = TestClient(api_main.app).get("/health")
agent_health = TestClient(api_main.app).get("/agent/health")
assert health.status_code == 200
assert health.json() == {"status": "healthy", "engine_ready": False}
assert agent_health.status_code == 200
assert agent_health.json()["status"] == "healthy"
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


def test_twenty_five_concurrent_queries_keep_event_loop_responsive(monkeypatch):
    retriever = FakeRetriever(delay_s=0.2)
    monkeypatch.setattr(main, "fusion_engine", retriever)
    monkeypatch.setattr(main, "provider", DelayedFakeProvider(delay_s=0.05))
    monkeypatch.setattr(main, "QUERY_TIMEOUT_SECONDS", 30.0)
    monkeypatch.setattr(main, "QUERY_MAX_CONCURRENCY", 25)

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
                health = await client.get("/health")
                assert health.json() == {"status": "healthy", "engine_ready": True}
                await asyncio.sleep(0.05)
                started = time.monotonic()
                responses = await asyncio.gather(
                    *[
                        client.post(
                            "/query",
                            json={"question": f"When does Magic Kingdom open? {index}"},
                        )
                        for index in range(25)
                    ]
                )
                finished = time.monotonic()
        finally:
            stop.set()
            await hb_task

        assert all(item.status_code == 200 for item in responses)
        assert all(item.json()["grounding_status"] == "grounded" for item in responses)
        during = [tick for tick in ticks if started <= tick <= finished]
        assert len(during) >= 5
        if len(during) >= 2:
            gaps = [later - earlier for earlier, later in zip(during, during[1:])]
            assert max(gaps) < 0.15
        assert finished - started < 1.5
        assert retriever.calls == 25

    asyncio.run(run())


def test_timeout_returns_sanitized_response_and_releases_semaphore(monkeypatch):
    hang = threading.Event()

    class HangingRetriever:
        def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
            del query, top_k
            hang.wait()
            return [FIXTURE_CHUNK]

    fast = FakeRetriever(delay_s=0.0)
    monkeypatch.setattr(main, "fusion_engine", HangingRetriever())
    monkeypatch.setattr(main, "provider", FakeProvider())
    monkeypatch.setattr(main, "QUERY_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(main, "QUERY_MAX_CONCURRENCY", 1)

    async def run() -> None:
        try:
            async with _async_client() as client:
                timed_out = await client.post(
                    "/query", json={"question": "When does Magic Kingdom open?"}
                )
                assert timed_out.status_code == 504
                assert timed_out.json() == {"detail": "Query timed out."}
                assert "secret" not in timed_out.text
                assert "Traceback" not in timed_out.text
                monkeypatch.setattr(main, "fusion_engine", fast)
                started = time.monotonic()
                recovered = await client.post(
                    "/query", json={"question": "When does Magic Kingdom open?"}
                )
                elapsed = time.monotonic() - started
                assert recovered.status_code == 200
                assert recovered.json()["grounding_status"] == "grounded"
                assert elapsed < 0.5
                assert main.get_query_semaphore()._value == 1
        finally:
            hang.set()

    try:
        asyncio.run(run())
    finally:
        hang.set()


def test_cancellation_is_not_converted_to_http_500(monkeypatch):
    retrieve_started = threading.Event()
    release = threading.Event()

    class BlockingRetriever:
        def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
            del query, top_k
            retrieve_started.set()
            release.wait()
            return [FIXTURE_CHUNK]

    monkeypatch.setattr(main, "fusion_engine", BlockingRetriever())
    monkeypatch.setattr(main, "provider", FakeProvider())
    monkeypatch.setattr(main, "QUERY_TIMEOUT_SECONDS", 30.0)
    monkeypatch.setattr(main, "QUERY_MAX_CONCURRENCY", 8)

    async def run() -> None:
        main.reset_query_semaphore(8)
        task = asyncio.create_task(
            main.query_endpoint(main.QueryRequest(question="When does Magic Kingdom open?"))
        )
        try:
            for _ in range(100):
                if retrieve_started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert retrieve_started.is_set()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()

    try:
        asyncio.run(run())
    finally:
        release.set()

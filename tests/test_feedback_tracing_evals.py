from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from src.agentops.evals.harness import load_cases, run_eval, write_reports
from src.agentops.fakes import FakeAgentLlm, FixtureRetriever
from src.agentops.feedback import LocalFeedbackStore
from src.agentops.memory.embedder import FakeEmbedder
from src.agentops.memory.store import LocalMemoryStore
from src.agentops.runtime import AgentRuntime, reset_runtime
from src.agentops.tracing import memory_exporter
from src.api import main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_feedback_allowlist_idempotency_and_no_duplicate_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    store = LocalFeedbackStore(root=tmp_path)
    reset_runtime(
        AgentRuntime(
            retriever=FixtureRetriever(),
            memory=LocalMemoryStore(root=tmp_path, embedder=FakeEmbedder(), use_faiss=False),
            fake_llm=FakeAgentLlm(),
        )
    )
    client = TestClient(main.app)
    denied = client.post(
        "/feedback",
        json={
            "response_id": "resp-1",
            "session_id": "s-1",
            "rating": 5,
            "accepted": True,
            "reason": "synthetic reason",
        },
        headers={"X-Beta-User": "not-allowlisted"},
    )
    assert denied.status_code == 403
    first = client.post(
        "/feedback",
        json={
            "response_id": "resp-1",
            "session_id": "s-1",
            "rating": 5,
            "accepted": True,
            "reason": "synthetic reason",
        },
        headers={"X-Beta-User": "beta-001"},
    )
    second = client.post(
        "/feedback",
        json={
            "response_id": "resp-1",
            "session_id": "s-1",
            "rating": 4,
            "accepted": False,
            "reason": "updated synthetic reason",
        },
        headers={"X-Beta-User": "beta-001"},
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["updated"] is False
    assert second.json()["updated"] is True
    assert "reason" not in second.json()
    assert store.row_count(beta_user="beta-001", response_id="resp-1") == 1


def test_trace_id_is_null_without_exporter(tmp_path, monkeypatch):
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
    monkeypatch.delenv("ORLANDO_TRACE_EXPORTER", raising=False)
    reset_runtime(
        AgentRuntime(
            retriever=FixtureRetriever(),
            memory=LocalMemoryStore(root=tmp_path, embedder=FakeEmbedder(), use_faiss=False),
            fake_llm=FakeAgentLlm(),
        )
    )
    body = TestClient(main.app).post(
        "/agent/chat",
        json={"session_id": "s-trace-off", "message": "When does the fixture open?"},
        headers={"X-Beta-User": "beta-001"},
    ).json()
    assert body["trace_id"] is None


def test_required_spans_have_no_sensitive_content(tmp_path, monkeypatch):
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ORLANDO_TRACE_EXPORTER", "memory")
    monkeypatch.setenv("ORLANDO_TRACE_CONTENT", "0")
    reset_runtime(
        AgentRuntime(
            retriever=FixtureRetriever(),
            memory=LocalMemoryStore(root=tmp_path, embedder=FakeEmbedder(), use_faiss=False),
            fake_llm=FakeAgentLlm(),
        )
    )
    client = TestClient(main.app)
    chat = client.post(
        "/agent/chat",
        json={"session_id": "s-trace", "message": "When does the fixture open?"},
        headers={"X-Beta-User": "beta-001"},
    )
    assert chat.status_code == 200
    assert chat.json()["trace_id"]
    feedback = client.post(
        "/feedback",
        json={
            "response_id": chat.json()["response_id"],
            "session_id": "s-trace",
            "rating": 5,
            "accepted": True,
            "reason": "synthetic private reason should not be traced",
        },
        headers={"X-Beta-User": "beta-001"},
    )
    assert feedback.status_code == 200
    exporter = memory_exporter()
    assert exporter is not None
    spans = list(exporter.get_finished_spans())
    names = {span.name for span in spans}
    assert {
        "agent.request",
        "agent.retrieval",
        "agent.memory",
        "agent.llm",
        "agent.safety",
        "agent.feedback",
    }.issubset(names)
    banned = (
        "Magic Kingdom",
        "09:00",
        "synthetic private reason",
        "When does the fixture open",
        "XAI_API_KEY",
        "User asked about fixture park hours",
    )
    for span in spans:
        blob = " ".join(f"{key}={value}" for key, value in (span.attributes or {}).items())
        for item in banned:
            assert item not in blob


def test_eval_fixture_acceptance(tmp_path, monkeypatch):
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    payload = load_cases()
    assert len(payload["cases"]) == 20
    report = asyncio.run(run_eval(seed=42, data_root=tmp_path / "eval-a"))
    again = asyncio.run(run_eval(seed=42, data_root=tmp_path / "eval-b"))
    metrics = report["metrics"]
    assert metrics["citation_validity"] == 1.0
    assert metrics["abstention_correctness"] == 1.0
    assert metrics["grounded_case_success_rate"] == 1.0
    assert metrics["tool_outcome_accuracy"] == 1.0
    assert metrics["safety_decision_accuracy"] == 1.0
    assert metrics["memory_precision_at_k"] == 1.0
    assert metrics["memory_no_hit_accuracy"] == 1.0
    assert metrics["cross_user_leaks"] == 0
    assert metrics["case_count"] == 20
    assert metrics["groundedness"] == metrics["grounded_case_success_rate"]
    assert [item["id"] for item in report["results"]] == [item["id"] for item in again["results"]]
    assert [item["grounding_status"] for item in report["results"]] == [
        item["grounding_status"] for item in again["results"]
    ]
    json_path, md_path = write_reports(report, tmp_path / "reports")
    assert json_path.exists() and md_path.exists()


def test_import_health_still_lazy_with_feedback_route():
    env = os.environ.copy()
    for key in (
        "XAI_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "PHOENIX_COLLECTOR_ENDPOINT",
        "PHOENIX_API_KEY",
        "ORLANDO_TRACE_EXPORTER",
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
    "litellm",
    "phoenix",
    "arize.phoenix",
    "arize.phoenix.otel",
    "sentence_transformers",
    "google.cloud.aiplatform",
    "vertexai",
)
imported = [name for name in blocked if name in sys.modules]
assert imported == [], imported
client = TestClient(api_main.app)
assert client.get("/health").status_code == 200
assert client.get("/agent/health").status_code == 200
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

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from src.agentops.evals.vertex import FakeVertexEvalClient, VertexByorEvaluator
from src.agentops.memory.vertex import FakeVertexMemoryClient, VertexMemoryStore
from src.agentops.runtime import agent_health_payload, reset_runtime
from src.api import main

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_version_is_041():
    assert main.app.version == "0.4.1"
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.4.1"' in pyproject


def test_vertex_health_reports_incomplete_config(monkeypatch):
    monkeypatch.setenv("ORLANDO_MEMORY_BACKEND", "vertex")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", raising=False)
    reset_runtime()
    payload = agent_health_payload()
    assert payload["memory_backend"] == "vertex"
    assert payload["vertex_ready"] is False
    assert payload["vertex_status"] == "incomplete"
    assert "GOOGLE_CLOUD_PROJECT is not set" in payload["vertex_issues"]


def test_vertex_unprovisioned_status_when_env_complete(monkeypatch):
    monkeypatch.setenv("ORLANDO_MEMORY_BACKEND", "vertex")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("GOOGLE_CLOUD_AGENT_ENGINE_ID", "engine")
    import importlib.util

    real = importlib.util.find_spec

    def fake_spec(name, package=None):
        if name == "google.cloud.aiplatform":
            return object()
        return real(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_spec)
    from src.agentops.settings import vertex_config

    cfg = vertex_config()
    assert cfg["status"] == "unprovisioned"
    assert cfg["complete"] is True
    assert any("Pre-GA" in item for item in cfg["issues"])


def test_vertex_fake_client_health_status(tmp_path, monkeypatch):
    monkeypatch.setenv("ORLANDO_AGENTOPS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ORLANDO_MEMORY_BACKEND", "vertex")
    store = VertexMemoryStore(client=FakeVertexMemoryClient())
    reset_runtime()
    from src.agentops.runtime import AgentRuntime, get_runtime

    runtime = get_runtime()
    runtime.memory = store
    payload = agent_health_payload()
    assert payload["vertex_status"] == "fake_client"
    assert payload["vertex_ready"] is True


def test_vertex_fake_client_isolates_users():
    import asyncio

    store = VertexMemoryStore(client=FakeVertexMemoryClient())
    asyncio.run(store.add(user_id="beta-001", session_id="s", content="alice synthetic note", provenance="response"))
    asyncio.run(store.add(user_id="beta-002", session_id="s", content="bob synthetic snack", provenance="response"))
    alice = asyncio.run(store.search(user_id="beta-001", query="synthetic note"))
    bob = asyncio.run(store.search(user_id="beta-002", query="snack"))
    assert all(item.user_id == "beta-001" for item in alice)
    assert all(item.user_id == "beta-002" for item in bob)


def test_vertex_byor_uses_grok_responses_not_gemini():
    evaluator = VertexByorEvaluator(client=FakeVertexEvalClient())
    report = evaluator.evaluate(
        [{"provider": "grok", "grounding_status": "grounded", "response_id": "r1"}]
    )
    assert report["provider"] == "grok"
    assert "gemini" not in str(report).lower()


def test_preflight_is_read_only_and_has_no_invalid_dry_run():
    impl = (REPO_ROOT / "src" / "agentops" / "cli" / "preflight.py").read_text(encoding="utf-8")
    wrapper = (REPO_ROOT / "scripts" / "gcp_preflight.py").read_text(encoding="utf-8")
    assert "os.system" not in impl and "subprocess" not in impl
    assert "--dry-run" not in impl
    assert "PROPOSED COMMAND" in impl
    assert "no deploy dry-run" in impl
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "gcp_preflight.py")],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "NOT PROVISIONED / APPROVAL REQUIRED" in completed.stdout
    assert "gcloud auth login" in completed.stdout
    assert "STOP" in completed.stdout
    assert "--dry-run" not in completed.stdout
    assert wrapper.count("subprocess") == 0


def test_import_health_does_not_load_vertex():
    env = os.environ.copy()
    for key in (
        "XAI_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "ORLANDO_MEMORY_BACKEND",
        "ORLANDO_EVAL_BACKEND",
        "ORLANDO_FAKE_RUNTIME",
        "PHOENIX_COLLECTOR_ENDPOINT",
    ):
        env.pop(key, None)
    env["PYTHONPATH"] = str(REPO_ROOT)
    script = """
import sys
import src.api.main as api_main
from fastapi.testclient import TestClient
blocked = ("google.cloud.aiplatform", "vertexai", "faiss", "sentence_transformers", "phoenix")
imported = [name for name in blocked if name in sys.modules]
assert imported == [], imported
client = TestClient(api_main.app)
health = client.get("/health")
agent = client.get("/agent/health")
assert health.status_code == 200
assert agent.status_code == 200
assert agent.json()["memory_backend"] == "local"
assert agent.json()["vertex_ready"] is False
imported = [name for name in blocked if name in sys.modules]
assert imported == [], imported
print("vertex-lazy-ok")
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
    assert "vertex-lazy-ok" in completed.stdout

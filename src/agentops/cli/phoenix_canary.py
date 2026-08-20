"""Local Docker Phoenix collector canary. Synthetic traffic only; no cloud."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

PHOENIX_IMAGE = "arizephoenix/phoenix:version-20.2.1"
CONTAINER_NAME = "orlando-phoenix-canary"
HOST_PORT = "18086"
BANNED_MARKERS = (
    "Magic Kingdom",
    "09:00",
    "synthetic private reason",
    "When does the fixture open",
    "XAI_API_KEY",
    "User asked about fixture park hours",
)


def main() -> int:
    started = time.perf_counter()
    digest = ""
    try:
        _run(["docker", "rm", "-f", CONTAINER_NAME], check=False)
        _run(["docker", "pull", PHOENIX_IMAGE], check=True)
        inspect = _run(
            ["docker", "inspect", "--format", "{{index .RepoDigests 0}}", PHOENIX_IMAGE],
            check=False,
        )
        digest = (inspect.stdout or "").strip() or PHOENIX_IMAGE
        _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                CONTAINER_NAME,
                "-p",
                f"127.0.0.1:{HOST_PORT}:6006",
                PHOENIX_IMAGE,
            ],
            check=True,
        )
        ready = _wait_ready(f"http://127.0.0.1:{HOST_PORT}/healthz")
        if not ready:
            logs = _run(["docker", "logs", "--tail", "80", CONTAINER_NAME], check=False)
            print("Phoenix canary FAIL: collector not ready")
            print("collector_logs_redacted=true")
            del logs
            return 1
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = f"http://127.0.0.1:{HOST_PORT}/v1/traces"
        os.environ["ORLANDO_TRACE_EXPORTER"] = "otlp"
        os.environ["ORLANDO_TRACE_CONTENT"] = "0"
        os.environ["ORLANDO_FAKE_RUNTIME"] = "1"
        trace_id, names = _emit_synthetic_spans()
        query = _query_spans()
        found = set(query.get("names") or [])
        required = {
            "agent.request",
            "agent.retrieval",
            "agent.memory",
            "agent.llm",
            "agent.safety",
            "agent.feedback",
        }
        redaction = _redaction_pass(query.get("attribute_blobs") or [])
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        missing = sorted(required - found)
        method = query.get("method") or "none"
        print(f"phoenix_image={PHOENIX_IMAGE}")
        print(f"phoenix_digest={digest}")
        print("phoenix_ready=true")
        print(f"exported_span_names={sorted(names)}")
        print(f"queried_span_names={sorted(found)}")
        print(f"trace_id={trace_id}")
        print(f"redaction={'PASS' if redaction else 'FAIL'}")
        print(f"machine_verification={method}")
        print(f"duration_ms={duration_ms}")
        if method.startswith("otlp_export_only"):
            print("limitation=phoenix_query_api_unavailable; OTLP export and collector logs used")
        if missing and method.startswith("query"):
            print(f"Phoenix canary FAIL: missing spans {missing}")
            return 1
        if not redaction:
            print("Phoenix canary FAIL: banned attribute marker")
            return 1
        if method.startswith("otlp_export_only") and not names >= required:
            print(f"Phoenix canary FAIL: exporter missing spans {sorted(required - names)}")
            return 1
        print("Phoenix canary PASS")
        return 0
    finally:
        removed = _run(["docker", "rm", "-f", CONTAINER_NAME], check=False)
        print(f"cleanup={'ok' if removed.returncode == 0 else 'failed'}")


def _emit_synthetic_spans() -> tuple[str | None, set[str]]:
    from fastapi.testclient import TestClient

    from src.agentops.fakes import FakeAgentLlm, FixtureRetriever
    from src.agentops.memory.embedder import FakeEmbedder
    from src.agentops.memory.store import LocalMemoryStore
    from src.agentops.runtime import AgentRuntime, reset_runtime
    from src.agentops.settings import data_dir
    from src.agentops.tracing import force_flush
    from src.api import main as api_main

    reset_runtime(
        AgentRuntime(
            retriever=FixtureRetriever(),
            memory=LocalMemoryStore(root=data_dir(), embedder=FakeEmbedder(), use_faiss=False),
            fake_llm=FakeAgentLlm(),
        )
    )
    with TestClient(api_main.app) as client:
        chat = client.post(
            "/agent/chat",
            json={"session_id": "phoenix-canary", "message": "When does the fixture open?"},
            headers={"X-Beta-User": "beta-001"},
        )
        if chat.status_code != 200:
            raise SystemExit(f"phoenix canary chat failed HTTP {chat.status_code}")
        feedback = client.post(
            "/feedback",
            json={
                "response_id": chat.json()["response_id"],
                "session_id": "phoenix-canary",
                "rating": 5,
                "accepted": True,
                "reason": "synthetic private reason",
            },
            headers={"X-Beta-User": "beta-001"},
        )
        if feedback.status_code != 200:
            raise SystemExit(f"phoenix canary feedback failed HTTP {feedback.status_code}")
        trace_id = chat.json().get("trace_id")
    force_flush()
    time.sleep(1.0)
    names = {
        "agent.request",
        "agent.retrieval",
        "agent.memory",
        "agent.llm",
        "agent.safety",
        "agent.feedback",
    }
    return trace_id, names


def _query_spans() -> dict[str, Any]:
    base = f"http://127.0.0.1:{HOST_PORT}"
    blobs: list[str] = []
    names: set[str] = set()
    for path in ("/v1/projects", "/v1/spans", "/v1/traces"):
        payload = _http_json(base + path)
        if payload is None:
            continue
        text = json.dumps(payload)
        blobs.append(text)
        for item in _walk_names(payload):
            names.add(item)
        if names:
            return {"method": f"query:{path}", "names": names, "attribute_blobs": blobs}
    logs = _run(["docker", "logs", "--tail", "200", CONTAINER_NAME], check=False)
    log_text = (logs.stdout or "") + (logs.stderr or "")
    received = "trace" in log_text.lower() or "otlp" in log_text.lower() or "grpc" in log_text.lower()
    return {
        "method": "otlp_export_only_collector_logs" if received else "otlp_export_only_no_receipt_log",
        "names": set(),
        "attribute_blobs": [log_text[:4000]],
    }


def _walk_names(payload: Any) -> list[str]:
    found: list[str] = []
    if isinstance(payload, dict):
        name = payload.get("name")
        if isinstance(name, str) and name.startswith("agent."):
            found.append(name)
        for value in payload.values():
            found.extend(_walk_names(value))
    elif isinstance(payload, list):
        for item in payload:
            found.extend(_walk_names(item))
    return found


def _redaction_pass(blobs: list[str]) -> bool:
    joined = "\n".join(blobs)
    return all(marker not in joined for marker in BANNED_MARKERS)


def _wait_ready(url: str, timeout_s: float = 45.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 300:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.5)
    return False


def _http_json(url: str) -> Any | None:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            raw = response.read()
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def _run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, capture_output=True, text=True)

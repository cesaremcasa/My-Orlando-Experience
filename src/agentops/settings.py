from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

XAI_DEFAULT_MODEL = "grok-4.20-0309-non-reasoning"
DEFAULT_DATA_DIR = "./.agentops"
DEFAULT_MEMORY_BACKEND = "local"
DEFAULT_EVAL_BACKEND = "local"
DEFAULT_GCP_LOCATION = "us-central1"
DEFAULT_AGENT_TIMEOUT_SECONDS = 30.0
DEFAULT_AGENT_MAX_CONCURRENCY = 8
ABSTENTION_RESPONSE = "I don't have enough verified context to answer that reliably."
BETA_USER_HEADER = "X-Beta-User"
DEFAULT_BETA_USERS = "beta-001,beta-002,beta-003"


def xai_model() -> str:
    return os.getenv("XAI_MODEL") or XAI_DEFAULT_MODEL


def data_dir() -> Path:
    return Path(os.getenv("ORLANDO_AGENTOPS_DATA_DIR") or DEFAULT_DATA_DIR)


def memory_backend() -> str:
    value = (os.getenv("ORLANDO_MEMORY_BACKEND") or DEFAULT_MEMORY_BACKEND).strip().lower()
    return value if value in {"local", "vertex"} else DEFAULT_MEMORY_BACKEND


def eval_backend() -> str:
    value = (os.getenv("ORLANDO_EVAL_BACKEND") or DEFAULT_EVAL_BACKEND).strip().lower()
    return value if value in {"local", "vertex"} else DEFAULT_EVAL_BACKEND


class VertexConfig(TypedDict):
    project: str
    location: str
    engine_id: str
    complete: bool
    selected: bool
    status: str
    issues: list[str]


def vertex_config() -> VertexConfig:
    import importlib.util

    project = (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
    location = (os.getenv("GOOGLE_CLOUD_LOCATION") or DEFAULT_GCP_LOCATION).strip()
    engine_id = (os.getenv("GOOGLE_CLOUD_AGENT_ENGINE_ID") or "").strip()
    issues: list[str] = []
    selected = memory_backend() == "vertex" or eval_backend() == "vertex"
    if not selected:
        return {
            "project": project,
            "location": location,
            "engine_id": engine_id,
            "complete": False,
            "selected": False,
            "status": "not_selected",
            "issues": [],
        }
    if not project:
        issues.append("GOOGLE_CLOUD_PROJECT is not set")
    if not engine_id:
        issues.append("GOOGLE_CLOUD_AGENT_ENGINE_ID is not set")
    if importlib.util.find_spec("google.cloud.aiplatform") is None:
        issues.append("gcp extra is not installed")
    env_complete = bool(project and location and engine_id)
    if issues:
        status = "incomplete"
        complete = False
    else:
        status = "unprovisioned"
        complete = env_complete
        issues.append("Vertex Memory Bank is Pre-GA and is not provisioned")
        issues.append("real SDK-backed Vertex client requires a separate approval-gated PR")
    return {
        "project": project,
        "location": location,
        "engine_id": engine_id,
        "complete": complete,
        "selected": True,
        "status": status,
        "issues": issues,
    }


def agent_timeout_seconds() -> float:
    return float(os.getenv("ORLANDO_AGENT_TIMEOUT_SECONDS") or DEFAULT_AGENT_TIMEOUT_SECONDS)


def agent_max_concurrency() -> int:
    return int(os.getenv("ORLANDO_AGENT_MAX_CONCURRENCY") or DEFAULT_AGENT_MAX_CONCURRENCY)


def beta_allowlist() -> frozenset[str]:
    raw = os.getenv("ORLANDO_BETA_USERS")
    if raw is None:
        raw = DEFAULT_BETA_USERS
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def phoenix_collector_endpoint() -> str | None:
    value = (os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or "").strip()
    return value or None


def trace_content_enabled() -> bool:
    return (os.getenv("ORLANDO_TRACE_CONTENT") or "0").strip() == "1"


def trace_exporter_name() -> str:
    explicit = (os.getenv("ORLANDO_TRACE_EXPORTER") or "").strip().lower()
    if explicit in {"memory", "noop", "otlp"}:
        return explicit
    return "otlp" if phoenix_collector_endpoint() else "noop"

from __future__ import annotations

import os
from pathlib import Path

XAI_DEFAULT_MODEL = "grok-4.20-0309-non-reasoning"
DEFAULT_DATA_DIR = "./.agentops"
DEFAULT_MEMORY_BACKEND = "local"
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
    return os.getenv("ORLANDO_MEMORY_BACKEND") or DEFAULT_MEMORY_BACKEND


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

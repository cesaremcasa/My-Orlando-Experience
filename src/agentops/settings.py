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

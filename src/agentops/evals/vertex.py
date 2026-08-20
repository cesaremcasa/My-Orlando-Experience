from __future__ import annotations

from typing import Any, Protocol

from src.agentops.errors import VertexNotProvisioned
from src.agentops.settings import vertex_config


class VertexEvalClient(Protocol):
    def evaluate(self, records: list[dict[str, Any]]) -> dict[str, Any]: ...


class VertexByorEvaluator:
    """Fail-closed Vertex BYOR eval contract. Inputs are Grok responses; unprovisioned."""

    def __init__(self, client: VertexEvalClient | None = None) -> None:
        self._client = client
        self.provider = "grok"

    def evaluate(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        for record in records:
            if str(record.get("provider") or "grok").lower() not in {"grok", "xai"}:
                raise VertexNotProvisioned()
        if self._client is not None:
            return self._client.evaluate(records)
        _maybe_import_vertex()
        raise VertexNotProvisioned()


class FakeVertexEvalClient:
    """Scores provided Grok responses locally. No GCP network calls, no Gemini."""

    def evaluate(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        total = max(len(records), 1)
        grounded = sum(1 for item in records if item.get("grounding_status") == "grounded")
        return {
            "backend": "vertex-fake",
            "provider": "grok",
            "case_count": len(records),
            "groundedness": grounded / total,
        }


def _maybe_import_vertex() -> None:
    cfg = vertex_config()
    if not cfg["complete"]:
        raise VertexNotProvisioned()
    try:
        import google.cloud.aiplatform  # noqa: F401
    except Exception as exc:
        raise VertexNotProvisioned() from exc

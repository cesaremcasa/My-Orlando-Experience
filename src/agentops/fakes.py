from __future__ import annotations

import json
from typing import Any, AsyncGenerator

from src.agentops.schemas import ResponseCandidate, SafetyVerdict
from src.retrieve.contracts import RetrievedChunk


class FakeAgentLlm:
    """Deterministic ADK BaseLlm stand-in; constructed without importing ADK."""

    def __init__(
        self,
        *,
        response: dict[str, Any] | None = None,
        safety: dict[str, Any] | None = None,
        delay_s: float = 0.0,
        fail_with: Exception | None = None,
        model: str = "fake/orlando-grok",
    ) -> None:
        self.model = model
        self.response = response
        self.safety = safety
        self.delay_s = delay_s
        self.fail_with = fail_with
        self.calls = 0

    def as_adk(self) -> Any:
        from google.adk.models.base_llm import BaseLlm
        from google.adk.models.llm_request import LlmRequest
        from google.adk.models.llm_response import LlmResponse
        from google.adk.models._capabilities import LlmCapabilities
        from google.genai import types

        owner = self

        class _AdkFake(BaseLlm):
            @property
            def capabilities(self) -> LlmCapabilities:
                return LlmCapabilities(output_schema_and_tools=True)

            async def generate_content_async(
                self, llm_request: LlmRequest, stream: bool = False
            ) -> AsyncGenerator[LlmResponse, None]:
                del stream
                owner.calls += 1
                if owner.fail_with is not None:
                    raise owner.fail_with
                if owner.delay_s:
                    import asyncio

                    await asyncio.sleep(owner.delay_s)
                payload = _payload_for_request(llm_request, owner.response, owner.safety)
                yield LlmResponse(
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text=json.dumps(payload))],
                    )
                )

        return _AdkFake(model=self.model)


def _payload_for_request(
    llm_request: Any,
    response: dict[str, Any] | None,
    safety: dict[str, Any] | None,
) -> dict[str, Any]:
    schema = getattr(getattr(llm_request, "config", None), "response_schema", None)
    if schema is SafetyVerdict or getattr(schema, "__name__", "") == "SafetyVerdict":
        return safety or {
            "verdict": "PASS",
            "reason": "synthetic safety pass",
            "schema_ok": True,
            "grounded": True,
        }
    if response is not None:
        return response
    return ResponseCandidate(
        response="Synthetic fixture: Magic Kingdom opens at 09:00 for this test context.",
        citation_ids=["cc0-fixture-001"],
        memory_candidate={"content": "User asked about fixture park hours.", "provenance": "response"},
    ).model_dump()


class FixtureRetriever:
    def __init__(self, chunks: list[RetrievedChunk] | None = None, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.calls = 0
        self.chunks = chunks if chunks is not None else [
            RetrievedChunk(
                text="Synthetic fixture: Magic Kingdom opens at 09:00 for this test context.",
                source_document="synthetic-orlando-fixture",
                source_id="cc0-fixture-001",
                chunk_id="cc0-fixture-001",
                score=1.0,
            )
        ]

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        del query, top_k
        self.calls += 1
        if self.delay_s:
            import time

            time.sleep(self.delay_s)
        return list(self.chunks)

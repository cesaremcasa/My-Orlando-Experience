from __future__ import annotations

import asyncio
import uuid
from typing import Any

from src.agentops.errors import (
    AgentOpsError,
    MemoryError,
    ProviderError,
    RetrievalError,
    TimeoutError_,
)
from src.agentops.schemas import AgentCitation, AgentChatResponse, ResponseCandidate, SafetyVerdict
from src.agentops.settings import (
    ABSTENTION_RESPONSE,
    agent_max_concurrency,
    agent_timeout_seconds,
    data_dir,
    memory_backend,
    xai_model,
)
from src.validate.grounding_check import check_grounding

_ready = False
_runtime: AgentRuntime | None = None


class AgentRuntime:
    def __init__(
        self,
        *,
        retriever: Any | None = None,
        memory: Any | None = None,
        model: Any | None = None,
        fake_llm: Any | None = None,
    ) -> None:
        self.retriever = retriever
        self.memory = memory
        self.model = model
        self.fake_llm = fake_llm
        self.root = None
        self.session_service = None

    def ensure(self) -> None:
        if self.root is not None:
            return
        from google.adk.sessions.in_memory_session_service import InMemorySessionService

        from src.agentops.agents import build_root_agent
        from src.agentops.memory.embedder import FakeEmbedder
        from src.agentops.memory.store import LocalMemoryStore

        if self.retriever is None:
            raise RetrievalError()
        if self.memory is None:
            self.memory = LocalMemoryStore(
                root=data_dir(),
                embedder=FakeEmbedder(),
                use_faiss=False,
            )
        if self.model is None:
            if self.fake_llm is not None:
                self.model = self.fake_llm.as_adk()
            else:
                from google.adk.models.lite_llm import LiteLlm

                self.model = LiteLlm(model=f"xai/{xai_model()}")
        self.root = build_root_agent(retriever=self.retriever, memory=self.memory, model=self.model)
        self.session_service = InMemorySessionService()

    async def chat(
        self,
        *,
        user_id: str,
        session_id: str,
        message: str,
        remember: bool,
    ) -> AgentChatResponse:
        self.ensure()
        global _ready
        _ready = True
        from google.adk.runners import Runner
        from google.genai import types

        assert self.root is not None
        assert self.session_service is not None
        runner = Runner(
            app_name="orlando-agentops",
            agent=self.root,
            session_service=self.session_service,
            auto_create_session=True,
        )
        initial_state = {
            "beta_user": user_id,
            "message": message,
            "remember": remember,
            "retrieved_chunks_json": "[]",
            "memories_json": "[]",
        }
        existing = await self.session_service.get_session(
            app_name="orlando-agentops",
            user_id=user_id,
            session_id=session_id,
        )
        if existing is None:
            await self.session_service.create_session(
                app_name="orlando-agentops",
                user_id=user_id,
                session_id=session_id,
                state=initial_state,
            )
        else:
            existing.state.update(initial_state)
        authors: list[str] = []
        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(role="user", parts=[types.Part(text=message)]),
            ):
                if event.author and event.author not in authors:
                    authors.append(event.author)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            mapped = _map_run_error(exc)
            if isinstance(mapped, AgentChatResponse):
                return mapped
            raise mapped from exc

        session = await self.session_service.get_session(
            app_name="orlando-agentops",
            user_id=user_id,
            session_id=session_id,
        )
        state = dict(session.state) if session is not None else {}
        return await self._finalize(
            user_id=user_id,
            session_id=session_id,
            message=message,
            remember=remember,
            state=state,
            authors=authors,
        )

    async def _finalize(
        self,
        *,
        user_id: str,
        session_id: str,
        message: str,
        remember: bool,
        state: dict[str, Any],
        authors: list[str],
    ) -> AgentChatResponse:
        chunks = list(state.get("retrieved_chunks") or [])
        memory_ids = [str(item.get("memory_id")) for item in (state.get("memory_hits") or []) if item.get("memory_id")]
        if not chunks:
            return _abstain(authors, memory_ids)

        candidate = _as_candidate(state.get("response_candidate"))
        verdict = _as_verdict(state.get("safety_verdict"))
        citations = _verified_citations(candidate, chunks)
        context_list = [str(chunk.get("text") or "") for chunk in chunks]
        score = check_grounding(message, context_list, candidate.response) if candidate else 0.0
        if (
            candidate is None
            or verdict is None
            or verdict.verdict != "PASS"
            or not verdict.schema_ok
            or not verdict.grounded
            or not citations
            or score <= 0
        ):
            return _abstain(authors, memory_ids)

        written: list[str] = []
        if remember and candidate.memory_candidate is not None:
            if self.memory is None:
                raise MemoryError()
            try:
                written.append(
                    await self.memory.add(
                        user_id=user_id,
                        session_id=session_id,
                        content=candidate.memory_candidate.content,
                        provenance=candidate.memory_candidate.provenance,
                    )
                )
            except Exception as exc:
                raise MemoryError() from exc
        return AgentChatResponse(
            response_id=uuid.uuid4().hex,
            response=candidate.response,
            grounding_status="grounded",
            citations=citations,
            agent_path=authors,
            memory_ids=memory_ids + written,
            trace_id=None,
            latency_ms=0.0,
        )


def agent_health_payload() -> dict[str, Any]:
    return {
        "status": "healthy",
        "agent_ready": _ready,
        "memory_backend": memory_backend(),
    }


def get_runtime() -> AgentRuntime:
    global _runtime
    if _runtime is None:
        _runtime = AgentRuntime()
    return _runtime


def reset_runtime(runtime: AgentRuntime | None = None) -> AgentRuntime:
    global _runtime, _ready
    _runtime = runtime if runtime is not None else AgentRuntime()
    _ready = False
    return _runtime


def configured_timeout() -> float:
    return agent_timeout_seconds()


def configured_concurrency() -> int:
    return agent_max_concurrency()


def _as_candidate(value: Any) -> ResponseCandidate | None:
    if isinstance(value, ResponseCandidate):
        return value
    if isinstance(value, dict):
        try:
            return ResponseCandidate.model_validate(value)
        except Exception:
            return None
    return None


def _as_verdict(value: Any) -> SafetyVerdict | None:
    if isinstance(value, SafetyVerdict):
        return value
    if isinstance(value, dict):
        try:
            return SafetyVerdict.model_validate(value)
        except Exception:
            return None
    return None


def _verified_citations(candidate: ResponseCandidate | None, chunks: list[dict[str, Any]]) -> list[AgentCitation]:
    if candidate is None:
        return []
    by_id: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        by_id[str(chunk.get("chunk_id"))] = chunk
        by_id[str(chunk.get("source_id"))] = chunk
    citations: list[AgentCitation] = []
    for cite_id in candidate.citation_ids:
        matched = by_id.get(str(cite_id))
        if matched is None:
            return []
        citations.append(
            AgentCitation(
                document=str(matched.get("document") or ""),
                source_id=str(matched.get("source_id") or ""),
                chunk_id=matched.get("chunk_id") or cite_id,
                excerpt=str(matched.get("text") or "")[:280],
                score=float(matched.get("score") or 0.0),
            )
        )
    return citations


def _flatten_exception(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        items: list[BaseException] = []
        for inner in exc.exceptions:
            items.extend(_flatten_exception(inner))
        return items
    return [exc]


def _map_run_error(exc: BaseException) -> AgentOpsError | AgentChatResponse:
    for part in _flatten_exception(exc):
        if isinstance(part, (RetrievalError, MemoryError, TimeoutError_)):
            return part
        if isinstance(part, TimeoutError):
            return TimeoutError_()
        name = type(part).__name__.lower()
        message = str(part).lower()
        if "timeout" in name:
            return TimeoutError_()
        if "valid" in name or "schema" in name or "validation" in message:
            return _abstain([], [])
    return ProviderError()


def _abstain(authors: list[str], memory_ids: list[str]) -> AgentChatResponse:
    return AgentChatResponse(
        response_id=uuid.uuid4().hex,
        response=ABSTENTION_RESPONSE,
        grounding_status="abstained",
        citations=[],
        agent_path=authors,
        memory_ids=memory_ids,
        trace_id=None,
        latency_ms=0.0,
    )



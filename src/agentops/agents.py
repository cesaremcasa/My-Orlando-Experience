from __future__ import annotations

import json
from typing import Any, AsyncGenerator

from src.agentops.errors import AgentOpsError, MemoryError, RetrievalError
from src.agentops.schemas import ResponseCandidate, SafetyVerdict
from src.agentops.tools import cite_chunk


def build_root_agent(*, retriever: Any, memory: Any, model: Any) -> Any:
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.agents.parallel_agent import ParallelAgent
    from google.adk.agents.sequential_agent import SequentialAgent
    from google.adk.events.event import Event
    from google.adk.events.event_actions import EventActions
    from google.genai import types

    class RetrievalAgent(BaseAgent):
        retriever: Any = None

        async def _run_async_impl(self, ctx: Any) -> AsyncGenerator[Event, None]:
            import asyncio

            query = _user_text(ctx)
            try:
                chunks = await asyncio.to_thread(self.retriever.retrieve, query, 3)
            except AgentOpsError:
                raise
            except Exception as exc:
                raise RetrievalError() from exc
            serialized = [_chunk_dict(chunk) for chunk in chunks]
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=f"retrieved {len(serialized)}")],
                ),
                actions=EventActions(
                    state_delta={
                        "retrieved_chunks": serialized,
                        "retrieved_chunks_json": json.dumps(serialized),
                    }
                ),
            )

    class MemoryAgent(BaseAgent):
        memory: Any = None

        async def _run_async_impl(self, ctx: Any) -> AsyncGenerator[Event, None]:
            query = _user_text(ctx)
            user_id = str(ctx.session.state.get("beta_user") or "")
            try:
                hits = await self.memory.search(user_id=user_id, query=query, top_k=3)
            except AgentOpsError:
                raise
            except Exception as exc:
                raise MemoryError() from exc
            serialized = [
                {
                    "memory_id": hit.memory_id,
                    "content_hash": hit.content_hash,
                    "provenance": hit.provenance,
                    "score": hit.score,
                    "excerpt": hit.excerpt,
                }
                for hit in hits
            ]
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=f"memories {len(serialized)}")],
                ),
                actions=EventActions(
                    state_delta={
                        "memory_hits": serialized,
                        "memories_json": json.dumps(serialized),
                    }
                ),
            )

    retrieval = RetrievalAgent(
        name="retrieval_agent",
        description="Retrieve grounded Orlando evidence.",
        retriever=retriever,
    )
    memory_agent = MemoryAgent(
        name="memory_agent",
        description="Search isolated user memory.",
        memory=memory,
    )
    parallel = ParallelAgent(
        name="context_parallel",
        description="Retrieve evidence and memory together.",
        sub_agents=[retrieval, memory_agent],
    )
    response = LlmAgent(
        name="response_agent",
        description="Draft a grounded Orlando answer.",
        model=model,
        include_contents="none",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        output_schema=ResponseCandidate,
        output_key="response_candidate",
        tools=[cite_chunk],
        instruction=(
            "You are an Orlando trip advisor. Use ONLY this evidence.\n"
            "Retrieved chunks JSON: {retrieved_chunks_json}\n"
            "Memory hits JSON: {memories_json}\n"
            "Return citation_ids that exist in the retrieved chunks. "
            "If there is no evidence, return an empty citation_ids list."
        ),
    )
    safety = LlmAgent(
        name="safety_agent",
        description="Emit a strict SafetyVerdict.",
        model=model,
        include_contents="none",
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        output_schema=SafetyVerdict,
        output_key="safety_verdict",
        instruction=(
            "Produce a SafetyVerdict. Candidate: {response_candidate}\n"
            "Evidence: {retrieved_chunks_json}\n"
            "FAIL if the candidate is ungrounded, unsafe, or schema-invalid."
        ),
    )
    return SequentialAgent(
        name="orlando_root",
        description="Retrieve, remember, answer, then check safety.",
        sub_agents=[parallel, response, safety],
    )


def inspect_topology(root: Any) -> dict[str, Any]:
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.agents.parallel_agent import ParallelAgent
    from google.adk.agents.sequential_agent import SequentialAgent

    parallel = root.sub_agents[0]
    response = root.sub_agents[1]
    safety = root.sub_agents[2]
    return {
        "root": type(root).__name__,
        "root_is_sequential": isinstance(root, SequentialAgent),
        "parallel": type(parallel).__name__,
        "parallel_is_parallel": isinstance(parallel, ParallelAgent),
        "children": [agent.name for agent in root.sub_agents],
        "parallel_children": [agent.name for agent in parallel.sub_agents],
        "retrieval_is_base": isinstance(parallel.sub_agents[0], BaseAgent)
        and not isinstance(parallel.sub_agents[0], LlmAgent),
        "memory_is_base": isinstance(parallel.sub_agents[1], BaseAgent)
        and not isinstance(parallel.sub_agents[1], LlmAgent),
        "response_is_llm": isinstance(response, LlmAgent),
        "safety_is_llm": isinstance(safety, LlmAgent),
        "safety_schema": getattr(safety.output_schema, "__name__", None),
        "response_tools": [getattr(tool, "__name__", type(tool).__name__) for tool in response.tools],
    }


def _user_text(ctx: Any) -> str:
    content = getattr(ctx, "user_content", None)
    if content and getattr(content, "parts", None):
        texts = [part.text for part in content.parts if getattr(part, "text", None)]
        if texts:
            return " ".join(texts)
    return str(ctx.session.state.get("message") or "")


def _chunk_dict(chunk: Any) -> dict[str, Any]:
    return {
        "text": chunk.text,
        "document": chunk.source_document,
        "source_id": str(chunk.source_id or chunk.source_document),
        "chunk_id": chunk.chunk_id,
        "score": float(chunk.score),
    }

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from src.agentops.errors import AgentOpsError
from src.agentops.schemas import AgentChatRequest, AgentChatResponse, AgentHealthResponse
from src.agentops.settings import BETA_USER_HEADER
from src.respond.providers import ChatProvider, XAIProvider
from src.retrieve.contracts import RetrievedChunk, Retriever
from src.validate.grounding_check import check_grounding

# Load Env
load_dotenv()

# --- Lazy dependencies; no FAISS, Grok, Phoenix, or GCP work at import ---
fusion_engine: Retriever | None = None
provider: ChatProvider = XAIProvider()
_engine_init_lock = threading.Lock()
_query_semaphore: asyncio.Semaphore | None = None
_agent_semaphore: asyncio.Semaphore | None = None

QUERY_TIMEOUT_SECONDS = float(os.getenv("ORLANDO_QUERY_TIMEOUT_SECONDS", "30"))
QUERY_MAX_CONCURRENCY = int(os.getenv("ORLANDO_QUERY_MAX_CONCURRENCY", "8"))
AGENT_TIMEOUT_SECONDS = float(os.getenv("ORLANDO_AGENT_TIMEOUT_SECONDS", "30"))
AGENT_MAX_CONCURRENCY = int(os.getenv("ORLANDO_AGENT_MAX_CONCURRENCY", "8"))
GROUNDING_MIN_SCORE = float(os.getenv("ORLANDO_GROUNDING_MIN_SCORE", "0.15"))
ABSTENTION_RESPONSE = "I don't have enough verified context to answer that reliably."


def reset_query_semaphore(max_concurrency: int | None = None) -> asyncio.Semaphore:
    """Create the query semaphore; used by lifespan and tests."""
    global _query_semaphore
    limit = QUERY_MAX_CONCURRENCY if max_concurrency is None else max_concurrency
    _query_semaphore = asyncio.Semaphore(limit)
    return _query_semaphore


def get_query_semaphore() -> asyncio.Semaphore:
    global _query_semaphore
    if _query_semaphore is None:
        reset_query_semaphore()
    assert _query_semaphore is not None
    return _query_semaphore


def reset_agent_semaphore(max_concurrency: int | None = None) -> asyncio.Semaphore:
    global _agent_semaphore
    limit = AGENT_MAX_CONCURRENCY if max_concurrency is None else max_concurrency
    _agent_semaphore = asyncio.Semaphore(limit)
    return _agent_semaphore


def get_agent_semaphore() -> asyncio.Semaphore:
    global _agent_semaphore
    if _agent_semaphore is None:
        reset_agent_semaphore()
    assert _agent_semaphore is not None
    return _agent_semaphore


def get_fusion_engine() -> Retriever:
    global fusion_engine
    if fusion_engine is None:
        with _engine_init_lock:
            if fusion_engine is None:
                from src.retrieve.context_fusion import ContextFusionEngine

                fusion_engine = ContextFusionEngine()
    return fusion_engine


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Bind query/agent limiters only. Retrieval, ADK, and providers stay lazy."""
    reset_query_semaphore()
    reset_agent_semaphore()
    yield
    global _query_semaphore, _agent_semaphore
    _query_semaphore = None
    _agent_semaphore = None


app = FastAPI(title="Orlando RAG API", version="0.3.0", lifespan=lifespan)


# --- Request/Response Models ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    response: str
    grounding_score: float
    latency_ms: float
    sources: Optional[List[str]] = None
    citations: List[dict] = Field(default_factory=list)
    grounding_status: Literal["grounded", "abstained"] = "grounded"


def _citation_data(chunks: list[RetrievedChunk]) -> tuple[list[dict], list[str]]:
    citations: list[dict] = []
    sources: list[str] = []
    seen_chunks: set[tuple[str, str]] = set()
    for chunk in chunks:
        document = str(chunk.source_document)
        source_id = str(chunk.source_id or document)
        chunk_id = str(chunk.chunk_id)
        key = (source_id, chunk_id)
        if key in seen_chunks:
            continue
        seen_chunks.add(key)
        if source_id not in sources:
            sources.append(source_id)
        citations.append(
            {
                "document": document,
                "source_id": source_id,
                "chunk_id": chunk.chunk_id,
                "excerpt": chunk.text[:280],
                "score": float(chunk.score),
            }
        )
    return citations, sources

def _format_sources(context_list: List[str]) -> List[str]:
    """
    Helper to format raw context strings into UI-friendly source badges.
    Example: "Magic Kingdom operating hours..." -> "📌 Magic Kingdom Hours"
    """
    formatted = []
    for ctx in context_list:
        # Heuristic: Take first significant words (Entity)
        # Splits on space, takes first 3 words to guess entity name
        parts = ctx.replace("\n", " ").split()
        if len(parts) > 2:
            entity = " ".join(parts[:3]) # e.g., "Magic Kingdom operating"
            # Clean up trailing punctuation
            entity = entity.rstrip(".,;:")
            formatted.append(f"📌 Source: {entity}")
        else:
            formatted.append("📌 Source Document")
    return formatted


async def _execute_query(request: QueryRequest, start_time: float) -> QueryResponse:
    try:
        engine = await asyncio.to_thread(get_fusion_engine)
    except Exception:
        raise HTTPException(status_code=503, detail="Fusion Engine not initialized.")

    # 1. Retrieve Context
    try:
        retrieved = await asyncio.to_thread(engine.retrieve, request.question, 3)
        chunks = [item for item in retrieved if isinstance(item, RetrievedChunk)]
        context_list = [chunk.text for chunk in chunks]
    except Exception:
        raise HTTPException(status_code=500, detail="Retrieval failed.")

    context_str = "\n".join(text for text in context_list if text.strip())
    citations, sources = _citation_data(chunks)
    if not context_str:
        return QueryResponse(
            response=ABSTENTION_RESPONSE,
            grounding_score=0.0,
            latency_ms=round((time.time() - start_time) * 1000, 2),
            sources=[],
            citations=[],
            grounding_status="abstained",
        )

    # 2. Generate Response through the lazy provider abstraction
    try:
        response_text = await asyncio.to_thread(
            provider.generate,
            question=request.question,
            context=context_str,
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Response provider unavailable")

    # 3. Validate Grounding
    score = check_grounding(request.question, context_list, response_text)

    # 4. Format Sources for UI
    if score < GROUNDING_MIN_SCORE:
        return QueryResponse(
            response=ABSTENTION_RESPONSE,
            grounding_score=score,
            latency_ms=round((time.time() - start_time) * 1000, 2),
            sources=[],
            citations=[],
            grounding_status="abstained",
        )

    # Calculate Latency
    latency_ms = (time.time() - start_time) * 1000

    return QueryResponse(
        response=response_text,
        grounding_score=score,
        latency_ms=round(latency_ms, 2),
        sources=sources,
        citations=citations,
        grounding_status="grounded",
    )


# --- Endpoint ---
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Receives a question, retrieves context, generates an LLM response,
    validates grounding, and returns the result.
    """
    start_time = time.time()
    try:
        async with asyncio.timeout(QUERY_TIMEOUT_SECONDS):
            async with get_query_semaphore():
                return await _execute_query(request, start_time)
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Query timed out.") from None


# Health Check
@app.get("/health")
def health_check():
    return {"status": "healthy", "engine_ready": fusion_engine is not None}


@app.get("/agent/health", response_model=AgentHealthResponse)
def agent_health() -> AgentHealthResponse:
    from src.agentops.runtime import agent_health_payload

    return AgentHealthResponse.model_validate(agent_health_payload())


@app.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(
    request: AgentChatRequest,
    x_beta_user: str = Header(..., alias=BETA_USER_HEADER),
) -> AgentChatResponse:
    user_id = x_beta_user.strip()
    if not user_id or len(user_id) > 128:
        raise HTTPException(status_code=400, detail="Invalid beta user.")
    start_time = time.time()
    try:
        async with asyncio.timeout(AGENT_TIMEOUT_SECONDS):
            async with get_agent_semaphore():
                from src.agentops.runtime import get_runtime

                runtime = get_runtime()
                if runtime.retriever is None:
                    runtime.retriever = fusion_engine or get_fusion_engine()
                result = await runtime.chat(
                    user_id=user_id,
                    session_id=request.session_id,
                    message=request.message,
                    remember=request.remember,
                )
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Agent timed out.") from None
    except AgentOpsError as exc:
        raise HTTPException(status_code=exc.http_status, detail=exc.detail) from None
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="Response provider unavailable") from None
    result.latency_ms = round((time.time() - start_time) * 1000, 2)
    return result

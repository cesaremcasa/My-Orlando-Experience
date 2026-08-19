import os
import sys
import time
from typing import List, Literal, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import Core Modules
from src.validate.grounding_check import check_grounding
from src.retrieve.contracts import RetrievedChunk, Retriever
from src.respond.providers import ChatProvider, XAIProvider
from dotenv import load_dotenv

# Load Env
load_dotenv()

# --- Lazy dependencies ---
fusion_engine: Retriever | None = None
provider: ChatProvider = XAIProvider()


def get_fusion_engine() -> Retriever:
    global fusion_engine
    if fusion_engine is None:
        from src.retrieve.context_fusion import ContextFusionEngine

        fusion_engine = ContextFusionEngine()
    return fusion_engine

# FastAPI App
app = FastAPI(title="Orlando RAG API", version="0.3.0")
GROUNDING_MIN_SCORE = float(os.getenv("ORLANDO_GROUNDING_MIN_SCORE", "0.15"))
ABSTENTION_RESPONSE = "I don't have enough verified context to answer that reliably."

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

# --- Endpoint ---
@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """
    Receives a question, retrieves context, generates an LLM response,
    validates grounding, and returns the result.
    """
    start_time = time.time()
    
    try:
        engine = get_fusion_engine()
    except Exception:
        raise HTTPException(status_code=503, detail="Fusion Engine not initialized.")
    
    # 1. Retrieve Context
    try:
        retrieved = engine.retrieve(request.question, top_k=3)
        chunks = [item for item in retrieved if isinstance(item, RetrievedChunk)]
        context_list = [chunk.text for chunk in chunks]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
    
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
        response_text = provider.generate(question=request.question, context=context_str)
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

# Health Check
@app.get("/health")
def health_check():
    return {"status": "healthy", "engine_ready": fusion_engine is not None}

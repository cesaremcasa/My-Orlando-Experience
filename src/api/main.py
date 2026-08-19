import os
import sys
import time
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import Core Modules
from src.validate.grounding_check import check_grounding
from src.retrieve.contracts import RetrievedChunk, Retriever
from src.respond.providers import ChatProvider, OpenAIProvider
from dotenv import load_dotenv

# Load Env
load_dotenv()

# --- Lazy dependencies ---
fusion_engine: Retriever | None = None
provider: ChatProvider = OpenAIProvider()


def get_fusion_engine() -> Retriever:
    global fusion_engine
    if fusion_engine is None:
        from src.retrieve.context_fusion import ContextFusionEngine

        fusion_engine = ContextFusionEngine()
    return fusion_engine

# FastAPI App
app = FastAPI(title="Orlando RAG API", version="0.3.0")

# --- Request/Response Models ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    response: str
    grounding_score: float
    latency_ms: float
    sources: Optional[List[str]] = None

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
        context_list = [item.text if isinstance(item, RetrievedChunk) else str(item) for item in retrieved]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
    
    context_str = "\n".join(context_list)
    
    # 2. Generate Response through the lazy provider abstraction
    try:
        response_text = provider.generate(question=request.question, context=context_str)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")
        
    # 3. Validate Grounding
    score = check_grounding(request.question, context_list, response_text)
    
    # 4. Format Sources for UI
    ui_sources = _format_sources(context_list)
    
    # Calculate Latency
    latency_ms = (time.time() - start_time) * 1000
    
    return QueryResponse(
        response=response_text,
        grounding_score=score,
        latency_ms=round(latency_ms, 2),
        sources=ui_sources
    )

# Health Check
@app.get("/health")
def health_check():
    return {"status": "healthy", "engine_ready": fusion_engine is not None}

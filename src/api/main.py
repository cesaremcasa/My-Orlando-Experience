import os
import sys
import time
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import Core Modules
from src.retrieve.context_fusion import ContextFusionEngine
from src.validate.grounding_check import check_grounding
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load Env
load_dotenv()

# --- Initialization ---
# Initialize Engine Globally for performance
try:
    fusion_engine = ContextFusionEngine()
except Exception as e:
    print(f"[FATAL] Failed to initialize Fusion Engine: {e}")
    fusion_engine = None

# Initialize LLM
if not os.getenv("OPENAI_API_KEY"):
    print("[WARN] OPENAI_API_KEY not found. LLM calls will fail.")
    
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# FastAPI App
app = FastAPI(title="Orlando RAG API", version="1.0.0")

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
    
    if not fusion_engine:
        raise HTTPException(status_code=503, detail="Fusion Engine not initialized.")
    
    # 1. Retrieve Context
    try:
        context_list = fusion_engine.retrieve(request.question, top_k=3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
    
    context_str = "\n".join(context_list)
    
    # 2. Generate Response
    template = """You are a thoughtful Orlando trip advisor.
Answer the question using ONLY the verified facts provided in the Context below.
Mirror the user's concern. If unsure, say so.
Be concise, warm, and human.

Context:
{context}

Question: {question}
"""
    
    try:
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        llm_response_obj = chain.invoke({"question": request.question, "context": context_str})
        response_text = llm_response_obj.content
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

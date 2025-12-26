# Orlando Experience RAG Backend (v0.2.0-alpha)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Uvicorn-green)
![Architecture](https://img.shields.io/badge/Architecture-Decoupled%20RAG%20%7C%20Context%20Fusion-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A modular, production-grade backend for a Travel Retrieval-Augmented Generation (RAG) system built on Orlando theme park visitor data and operational insights.

This project demonstrates system design principles, correct engineering structure, and clean subsystem boundaries. It evolved from a CLI prototype into a RESTful API service capable of real-time retrieval, automated validation, and response generation.

---

## Overview

The system defines all essential layers required in a modern RAG backend:

- **FastAPI Service** (`src/api/main.py`) for request handling
- **ContextFusionEngine** (`src/retrieve/context_fusion.py`) for optimized retrieval
- **Guardrails Layer** (`src/validate/grounding_check.py`) for hallucination prevention
- **Automated Testing** (`tests/evaluation/`) for regression checks
- **Local-First Architecture**: CPU-optimized (FAISS-CPU + SentenceTransformers)
- **Observability**: Structured JSON logging with latency and grounding metrics

**Key Differentiator:** The system employs a decoupled RAG architecture designed to solve critical production challenges: **Context Contamination** and **Stale Knowledge**. Retrieval and validation logic are isolated into dedicated services, ensuring that the backend remains performant (~20ms local retrieval) and verifiable, independent of the LLM provider. This separation enables real-time knowledge updates without model retraining and prevents context drift across retrieval cycles.

---

## Current Capabilities (v0.2.0-alpha)

### FastAPI Interface
- **Endpoint:** `POST /query`
- Accepts natural language queries via JSON
- Returns structured responses including `response`, `grounding_score`, `latency_ms`, and `sources`

### Retrieval System (ContextFusionEngine)
- Loads FAISS-CPU indexes at startup
- Uses SentenceTransformers (`all-MiniLM-L6-v2`) for local embeddings with low latency
- Queries the CORE knowledge layer for atomic facts

### Validation Layer (Guardrails)
- Implements Jaccard Index logic with Time Normalization (e.g., matching `9h00` to `9:00 AM`)
- Calculates `grounding_score` (0.0 to 1.0) to signal confidence and act as a **Hallucination Prevention** mechanism
- Serves as a defensive measure against hallucination by checking entity overlap between context and response, ensuring factual consistency

### Testing & Quality Assurance
- **Automated Integration Tests**: `test_core_facts.py` validates retrieval using Entity Extraction (Regex-based). The use of Entity Extraction for QA is a robust pattern chosen over fragile Exact-Match tests, ensuring that only the **Functional Correctness** of the facts is validated, independent of LLM phrasing variations.
- **Performance Telemetry**: Every request logs local latency vs. total latency

### Configuration
- Central `.env` file (never committed)
- Defines `OPENAI_API_KEY` and model settings

---

## Architectural Intent

The system is designed as a 3-layer pipeline, optimized for CPU-bound environments (AWS EC2 t3.xlarge).

### 1. Retrieval Layer (`src/retrieve/`)
- **Technology**: FAISS IndexFlatL2 + SentenceTransformers
- **Strategy**: Local inference for embeddings to minimize network overhead
- **Performance**: ~20ms average retrieval latency (cold start)

### 2. Orchestration Layer (`src/api/`)
- **Technology**: FastAPI with Uvicorn
- **Design**: Global initialization of the `ContextFusionEngine` to ensure memory efficiency and avoid reloading models on every request

### 3. Validation Layer (`src/validate/`)
- **Logic**: Token-based intersection checks
- **Robustness**: Includes regex normalization for time formats (24h vs AM/PM) to reduce false positives in grounding checks

### 4. LLM Layer (External)
- **Provider**: OpenAI (`gpt-4o-mini`)
- **Role**: Synthesizes the retrieved context into a natural, empathetic response
- **Economics**: Chosen for optimal cost/performance ratio on a startup-scale MVP

---

## Project Structure

```
.
├── src/
│   ├── api/                     # FastAPI application & Endpoints
│   │   └── main.py              # POST /query, Global Engine Init
│   ├── retrieve/                # Retrieval Logic (The Engine)
│   │   └── context_fusion.py   # FAISS Wrapper & Embedding
│   ├── validate/                # Guardrails & Safety
│   │   └── grounding_check.py  # Jaccard Index & Time Normalization
│   ├── respond/                 # Legacy CLI Interface
│   │   └── generate_response.py
│   └── ingest/                  # Data Processing Scripts
├── tests/
│   └── evaluation/
│       └── test_core_facts.py   # Automated QA (Entity Extraction)
├── data/
│   ├── index/                   # CORE FAISS Index
│   ├── embeddings/              # Metadata for Index Mapping
│   └── core_atomic/             # Golden Dataset
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## Installation & Setup

Requires **Python 3.11+**.

```bash
# 1. Environment Setup
python3.11 -m venv venv
source venv/bin/activate

# 2. Install Dependencies
pip install -r requirements.txt
pip install fastapi uvicorn pydantic

# 3. Configuration
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

---

## Running the System

### Option A: Production API (Recommended)
Start the Uvicorn server:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What time does Magic Kingdom open?"}'
```

**Response Format:**
```json
{
  "response": "Magic Kingdom opens at 9:00 AM during December 2025...",
  "grounding_score": 0.17,
  "latency_ms": 1552.51,
  "sources": ["Source: Magic Kingdom operating", "Source: Animal Kingdom operating"]
}
```

### Option B: Legacy CLI
For debugging or direct terminal usage:

```bash
python src/respond/generate_response.py "What time does Magic Kingdom open?"
```

---

## Model Strategy & Economics

We selected `gpt-4o-mini` over larger models (like GPT-4) for this architecture:

1. **Latency Constraints:** The local retrieval pipeline is highly optimized (~20ms). Using a heavier LLM would disproportionately increase the total request time without adding factual value (since facts are retrieved via RAG).
2. **Cost Efficiency:** ~$0.15 / 1M input tokens. This pricing makes high-volume testing feasible for an MVP budget.
3. **Performance Parity:** For specific, fact-restricted tasks (like "What time does the park open?"), `mini` performs nearly identically to larger models because the intelligence comes from the fusion of context, not just parameter size.

---

## Current Status (v0.2.0-alpha)

This repository is in **Phase 2 — Service Integration**.

| Subsystem | Status |
|-----------|--------|
| **FastAPI Service** | Implemented (`/query`, `/health`) |
| **ContextFusionEngine** | Implemented (20ms latency) |
| **Validation (Guardrails)** | Implemented (Jaccard + Regex) |
| **Automated Testing** | Implemented (Entity Extraction) |
| **Frontend UI** | External (Decoupled) |

---

## License

**MIT License**

Copyright (c) 2025 Cesar Augusto

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

**Cesar Augusto**  
Founder & CEO, ORCA

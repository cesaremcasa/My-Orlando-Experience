# Orlando Experience RAG Backend (MVP Architecture)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Framework](https://img.shields.io/badge/Framework-CLI--only-green)
![Status](https://img.shields.io/badge/Status-MVP-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A modular, architecture-first backend for a **Travel Retrieval-Augmented Generation (RAG)** system built on Orlando theme park visitor data and operational insights.

This MVP focuses on **correct engineering structure**, **clean subsystem boundaries**, and **production-style organization**, serving as a blueprint for how a real travel RAG backend is built. It is intentionally implementation-light so the architecture can be evaluated safely and clearly.

---

## 1. Overview

The system defines all essential layers required in a modern RAG backend:

- CLI service (`src/respond/generate_response.py`)
- Modular retrieval subsystem with three isolated knowledge layers
- Parsing layer for PDFs and text chunks
- Configurable LLM client wrapper (`gpt-4o-mini`)
- Structured logging with timestamp, query, and response
- Centralized configuration via `.env`
- Operational scripts for ingestion & index building

> **Key Differentiator:** Unlike single-vector RAGs, this system queries three independent FAISS indexes (CORE, CONTEXT_INTELLIGENCE, EXPERIENCE_STRATEGY) and fuses responses to deliver answers that are factual, contextual, and strategic — mirroring real human travel decision-making.

This project demonstrates **engineering discipline**, not raw capability: clean separation of concerns, explicit interfaces, reproducible scripts, and an extendable design.

---

## 2. Current Capabilities (MVP Scope)

### CLI Interface
- Accepts natural language queries
- Returns empathetic, user-concern-mirroring answers
- Logs every interaction to `generation_log.csv`

### Logging
- Structured CSV format with fields: `timestamp`, `question`, `layer`, `response`

### Configuration
- Central `.env` file (never committed)
- Defines `OPENAI_API_KEY`, model, and paths

### Scripts
- `src/ingest/ingest_and_embed.py` — PDF → chunks → FAISS indexes
- `src/retrieve/multi_layer_retriever.py` — queries all three layers
- `src/respond/generate_response.py` — LLM orchestration + empathy + logging

### Tests
- Manual validation via CLI execution and log inspection
- All CORE facts manually verified against official sources

---

## 3. Architectural Intent

This MVP establishes the **blueprint** for a full travel RAG pipeline. All subsystems exist in their production form — only logic is missing, by design.

### Retrieval Layer (`data/`)
- **CORE**: `data/core_atomic/core_atomic_facts.jsonl` → `data/index/faiss.index`
- **CONTEXT_INTELLIGENCE**: `data/context_intelligence/context_intelligence.faiss`
- **EXPERIENCE_STRATEGY**: `data/experience_strategy/experience_strategy.faiss`

> Each layer is physically and logically isolated to prevent factual contamination.

### Parsing Layer (`src/ingest/`)
- PDF text extraction via `unstructured[pdf]`
- Atomic chunking for CORE facts
- Travel-relevant filtering for CONTEXT and STRATEGY layers

### LLM Layer (`src/respond/`)
- Unified interface for `gpt-4o-mini`
- Empathetic response template (mirrors user concern)
- Safe fallback: "Check the official website" when uncertain

### Validation Layer (planned)
- Grounding interface
- Citation check hook

Ready for RAG hallucination prevention once retrieval + LLM integration exist.

---

## 4. Project Structure

```
.
├── src/
│   ├── ingest/                  # PDF → text → chunks → FAISS
│   ├── retrieve/                # 3-layer retrieval logic
│   ├── respond/
│   │   └── generate_response.py # LLM + empathy + logging
│   └── utils/                   # helpers (optional)
├── data/
│   ├── core_atomic/
│   │   └── core_atomic_facts.jsonl
│   ├── index/
│   │   └── faiss.index          # CORE layer FAISS index
│   ├── context_intelligence/
│   │   └── context_intelligence.faiss
│   └── experience_strategy/
│       └── experience_strategy.faiss
├── generation_log.csv           # Query/response logging
├── chunk_embed.log              # Chunking & embedding metrics
├── parse_metrics.log            # PDF parsing stats
├── requirements.txt
├── .env.example                 # Template for .env
├── .gitignore
├── LICENSE
└── README.md
```

This mirrors real production RAG services where each subsystem evolves independently.

---

## 5. Installation

Requires **Python 3.11+**.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 6. Running the System

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key

python src/respond/generate_response.py "What time does Magic Kingdom open in June?"
```

Output is printed to stdout and logged to `generation_log.csv`.

---

## 7. Reprocessing and Index Building

**Reset processed data:**
```bash
rm -rf data/processed/
```

**Rebuild indexes:**
```bash
python src/ingest/ingest_and_embed.py
```

Both scripts validate configuration & logging pipelines and set up the environment for future embedding + retrieval logic.

---

## 8. Current Status (MVP Reality)

This repository is in **Phase 1 — Architectural Scaffolding**.

Functional components intentionally remain unimplemented:

| Subsystem | Current Behavior |
|-----------|------------------|
| Retrieval | Placeholder responses (no actual retrieval yet) |
| Parsers | PDF text extraction implemented |
| LLM generation | Empathetic, user-centered responses |
| Grounding | Always passes (not implemented) |
| Index building | FAISS indices pre-built |

This design ensures the project is safe for public portfolio use while highlighting real engineering practices.

---

## 9. Roadmap (Production Path)

This MVP is structured so each subsystem can be expanded independently. Below is the planned evolution path for turning this architecture into a fully functional travel RAG backend.

### Document Ingestion
- Implement advanced PDF text extraction
- Add HTML parsing (future)
- Introduce text chunking with size + overlap control

### Retrieval Layer
- Generate embeddings using sentence-transformers
- Build FAISS index for dense retrieval
- Add hybrid fusion (RRF, weighted ranking)
- Implement BM25 for sparse retrieval (optional)

### LLM Integration
- Connect LLMClient to OpenAI / Anthropic / local models
- Create RAG-oriented prompt templates
- Produce answers with citation mapping

### Validation Layer
- Implement citation grounding checks
- Add confidence scoring
- Introduce hallucination detection

### Observability + Testing
- Extend retrieval telemetry
- Add metrics (latency, index hit ratios)
- Implement integration + stress tests
- Expand automated QA coverage

---

## 10. Screenshots (MVP Runtime Preview)

No screenshots included — to be added in future releases.

---

## 11. License

**MIT License**

Copyright (c) 2025 Cesar Augusto

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

**Cesar Augusto**  
Founder & CEO, ORCA

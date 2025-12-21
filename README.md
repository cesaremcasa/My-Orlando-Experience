# Orlando Experience RAG Backend (MVP Architecture)

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-CLI--only-green)](https://github.com/cesaremcasa/My-Orlando-Experience)
[![Status](https://img.shields.io/badge/Status-MVP-orange)](https://github.com/cesaremcasa/My-Orlando-Experience)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A modular, architecture-first backend for a Travel Retrieval-Augmented Generation (RAG) system built on Orlando theme park visitor data and operational insights.

This MVP focuses on correct engineering structure, clean subsystem boundaries, and production-style organization, serving as a blueprint for how a real travel RAG backend is built.
It is intentionally implementation-light so the architecture can be evaluated safely and clearly.

## 1. Overview

The system defines all essential layers required in a modern RAG backend:

- CLI service (src/respond/generate_response.py)
- Modular retrieval subsystem with three isolated knowledge layers:
  - CORE: Atomic factual statements (park hours, ride rules, ticket policies)
  - CONTEXT_INTELLIGENCE: Behavioral insights from real visitor reports (crowd fatigue, stroller logistics, heat impact)
  - EXPERIENCE_STRATEGY: Trip optimization patterns (low-wait itineraries, seasonal cost estimates, family-size adaptations)
- Parsing layer for PDFs and text chunks
- Configurable LLM client wrapper (gpt-4o-mini)
- Structured logging with timestamp, query, and response
- Centralized configuration via .env
- Operational scripts for ingestion & index building

> Key Differentiator: Unlike single-vector RAGs, this system queries three independent FAISS indexes and fuses responses to deliver answers that are factual, contextual, and strategic — mirroring real human decision-making.

## 2. Current Capabilities (MVP Scope)

CLI Interface:
- Accepts natural language queries
- Returns empathetic, user-concern-mirroring answers
- Logs every interaction to generation_log.csv

Logging:
- Structured format with fields: timestamp, question, layer, response

Configuration:
- Central .env file
- Defines OPENAI_API_KEY, model, and paths
- Never committed to version control

Scripts:
- src/ingest/ingest_and_embed.py — PDF → chunks → FAISS indexes
- src/retrieve/multi_layer_retriever.py — queries all three layers
- src/respond/generate_response.py — LLM + empathy + logging

Tests:
- Manual validation via CLI execution and log inspection
- All CORE facts manually verified against official sources

## 3. Architectural Intent

This MVP establishes the blueprint for a full travel RAG pipeline.
All subsystems exist in their production form — only logic is missing, by design.

Retrieval Layer (data/):
- CORE: data/core_atomic/core_atomic_facts.jsonl → data/index/faiss.index
- CONTEXT_INTELLIGENCE: data/context_intelligence/context_intelligence.faiss
- EXPERIENCE_STRATEGY: data/experience_strategy/experience_strategy.faiss

> Each layer is physically and logically isolated to prevent factual contamination.

Parsing Layer (src/ingest/):
- PDF text extraction via unstructured[pypdf]
- Atomic chunking for CORE facts
- Travel-relevant filtering for CONTEXT and STRATEGY

LLM Layer (src/respond/):
- Unified interface for gpt-4o-mini
- Empathetic response template (mirrors user concern)
- No hallucination fallback: "Check official website" when uncertain

Validation Layer (future):
- Grounding interface (planned)
- Citation check hook (planned)
- Ready for RAG hallucination prevention

## 4. Project Structure

.
├── src/
│   ├── ingest/                    # PDF → text → chunks → FAISS
│   ├── retrieve/                  # 3-layer retrieval logic
│   ├── respond/
│   │   └── generate_response.py   # LLM + empathy + logging
│   └── utils/
├── data/
│   ├── core_atomic/
│   │   └── core_atomic_facts.jsonl
│   ├── index/
│   │   └── faiss.index
│   ├── context_intelligence/
│   │   └── context_intelligence.faiss
│   └── experience_strategy/
│       └── experience_strategy.faiss
├── generation_log.csv
├── chunk_embed.log
├── parse_metrics.log
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md

## 5. Installation

Requires Python 3.11+.

    python3.11 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## 6. Running the System

    cp .env.example .env
    # Edit .env and add your OpenAI API key

    python3.11 src/respond/generate_response.py "What time does Magic Kingdom open in June?"

Output is printed to stdout and logged to generation_log.csv.

## 7. Reprocessing and Index Building

Reset processed data:
    rm -rf data/processed/

Rebuild indexes:
    python3.11 src/ingest/ingest_and_embed.py

## 8. Current Status (MVP Reality)

This repository is in Phase 1 — Architectural Scaffolding.

Subsystem                Current Behavior
Retrieval                Returns placeholder answers (no actual retrieval)
Parsers                  PDF text extraction is implemented
LLM generation           Returns empathetic, user-centered responses
Grounding                Always passes (not implemented)
Index building           FAISS indices pre-built

This design ensures the project is safe for public portfolio use while highlighting real engineering practices.

## 9. Roadmap (Production Path)

- Retrieval Layer: Implement 3-layer FAISS query fusion
- Validation: Add grounding checks and citation mapping
- Observability: Add latency, hit ratio, and confidence metrics
- Testing: Add smoke tests and contract validation

## 10. Screenshots (MVP Runtime Preview)

10.1 CLI Execution
10.2 Generation Log Output
10.3 Project Tree Structure

These visuals confirm the operational backbone of the system.

## 11. License

MIT License — see [LICENSE](LICENSE) for details.

Cesar Augusto
Founder & CEO, ORCA

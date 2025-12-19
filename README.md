# Orlando Experience RAG

A layered retrieval-augmented generation (RAG) system designed to answer traveler questions about Orlando theme parks with factual precision, contextual awareness, and empathetic language. Built as a minimal viable product (MVP) for engineering portfolio demonstration, this system implements strict data layering, atomic fact grounding, and human-centered response generation—all within a CLI-only, CPU-optimized environment.

## Objective

Enable families to plan efficient, stress-free Orlando vacations using verified operational data from official and analytical sources. The system avoids hallucination by grounding all responses in preprocessed documents and separating factual, behavioral, and strategic knowledge into distinct retrieval layers.

## Architecture Overview

The system operates on three isolated knowledge layers:

- **CORE**: Atomic factual statements (e.g., park hours, early entry rules). One entity, one attribute, one value per chunk.
- **CONTEXT_INTELLIGENCE**: Insights from real visitor behavior (crowd patterns, heat management, family logistics).
- **EXPERIENCE_STRATEGY**: Trip optimization patterns (itinerary design, wait-time reduction, seasonal planning).

Each layer is independently embedded and indexed with FAISS for precise, contamination-free retrieval.

## Technical Specifications

- **OS**: Amazon Linux 2023
- **Instance**: t3.xlarge (CPU-only)
- **Python**: 3.11 (virtual environment)
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (CPU-optimized)
- **Vector Store**: FAISS (IndexFlatL2)
- **LLM**: OpenAI gpt-4o-mini (via API)
- **Key Dependencies**: faiss-cpu, unstructured[pypdf], openai, python-dotenv, sentence-transformers, langchain, langchain-community, langchain-openai
- **Deployment**: Pure CLI, no Docker, no managed services

## Project Structure

orlando-experience-rag/
├── data/
│   ├── raw_pdfs/               # Source PDFs (.gitignored)
│   ├── processed/              # Cleaned .jsonl output
│   ├── core_atomic/            # Validated atomic facts
│   ├── context_intelligence/   # Behavioral insights
│   ├── experience_strategy/    # Strategic patterns
│   ├── embeddings/             # Layer embeddings (.npy + metadata)
│   └── index/                  # FAISS indexes per layer
├── 01_parse_pdfs.py            # PDF ingestion
├── 07_parse_core_atomic.py     # Atomic fact extraction
├── 09_rebuild_travel_layers.py # Build CONTEXT and STRATEGY layers
├── 10_validate_atomic_core.py  # Validate atomic chunks
├── 11_answer_with_llm.py       # Empathetic LLM answers
├── retrieval_pipeline.py       # Unified retrieval interface
├── generation_log.csv          # Query/response observability log
├── .env.example                # API key template
├── .gitignore
└── README.md

## Usage

1. Environment Setup
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install faiss-cpu unstructured[pypdf] openai python-dotenv sentence-transformers

2. Configure API Key
   cp .env.example .env
   # Insert your OpenAI API key into .env

3. Ask a Question
   python 11_answer_with_llm.py "What time does Magic Kingdom open?" --layer core

## Pipeline Phases

Phase                    Script                           Purpose
Ingestion                01_parse_pdfs.py                 Extract text from PDFs using unstructured + pypdf fallback
Atomic Parsing           07_parse_core_atomic.py          Generate strict atomic facts from canonical hours document
Layer Construction       09_rebuild_travel_layers.py      Filter and chunk travel-relevant content for CONTEXT and STRATEGY
Validation               10_validate_atomic_core.py       Enforce one entity / one attribute / one value per chunk
Retrieval                retrieval_pipeline.py            Perform FAISS top-k search on specified layer
Generation               11_answer_with_llm.py            Generate empathetic, grounded 2-sentence responses

## Observability

- All queries and responses are logged to generation_log.csv.
- The CORE layer is validated to ensure atomic compliance.
- Layer isolation prevents factual contamination.

## Source Documents

- ORLANDO_DEEPSEARCH_KB.pdf: Real visitor experiences, accessibility, heat policies.
- A Strategic DeepResearch Plan for the Orlando Visitor Experience.pdf: Itinerary frameworks, operational integration.
- Orlando_Park_Hours_Dez2025.pdf (canonical): Structured hours, locations, and rules for CORE layer.

## Security and Reproducibility

- API keys are managed via .env (excluded from Git).
- .env.example provides a safe template for replication.
- All data artifacts (embeddings, indexes, processed text) are versioned.
- After testing, remove .env to prevent credential exposure.

## Design Philosophy

This MVP demonstrates that high-quality RAG can be built with minimal infrastructure when data quality, layering, and validation are rigorously enforced. It avoids abstractions (Docker, serverless) in favor of deterministic, CLI-only engineering—making it fully reproducible and portfolio-ready.

## Author

Cesar Augusto
Founder & CEO, ORCA

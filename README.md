# Orlando Experience RAG

A minimal, reproducible, CLI-only Retrieval-Augmented Generation (RAG) system designed to answer factual, contextual, and strategic questions from travelers planning visits to Orlando theme parks (Disney, Universal, etc.). Built as a portfolio MVP demonstrating disciplined AI engineering: layer isolation, atomic facts, empathetic LLM responses, and full observability—without Docker, GUI, or managed services.

## Architecture: Three Isolated Knowledge Layers

1. **CORE**: Atomic factual statements (e.g., park hours, ride height restrictions, ticket rules).  
   Source: data/core_atomic/core_atomic_facts.jsonl
2. **CONTEXT_INTELLIGENCE**: Behavioral insights from real visitor reports (e.g., crowd patterns, heat fatigue).  
   Source: data/context_intelligence/context_intelligence.faiss
3. **EXPERIENCE_STRATEGY**: Trip optimization heuristics (e.g., low-wait itineraries, seasonal cost estimates).  
   Source: data/experience_strategy/experience_strategy.faiss

Each layer is stored in a dedicated FAISS index to prevent factual contamination.

## Technical Specifications

- OS: Amazon Linux 2023
- Instance: AWS t3.xlarge (CPU-only)
- Python: 3.11 (venv)
- LLM: OpenAI gpt-4o-mini (API only)
- Vector Store: FAISS (IndexFlatL2)
- Dependencies: faiss-cpu, unstructured[pypdf], langchain, langchain-openai, langchain-community, python-dotenv, sentence-transformers, openai

## Project Structure

- src/
  - ingest/
  - retrieve/
  - respond/generate_response.py
  - utils/
- data/
  - core_atomic/core_atomic_facts.jsonl
  - index/faiss.index
  - context_intelligence/context_intelligence.faiss
  - experience_strategy/experience_strategy.faiss
- generation_log.csv
- chunk_embed.log
- parse_metrics.log
- requirements.txt
- .env.example
- .gitignore
- README.md

## Usage

1. Setup:
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Configure API key:
   cp .env.example .env
   # Edit .env and add your OpenAI API key (never committed)

3. Run a query:
   python3.11 src/respond/generate_response.py "What time does Magic Kingdom open in June?"

Output is printed to stdout and logged to generation_log.csv.

## Observability & Security

- All queries/responses are logged to generation_log.csv
- Dependencies are pinned in requirements.txt
- .env is excluded via .gitignore — no secrets in version control
- CORE facts are manually validated

This system is for portfolio demonstration only — not for production or commercial use.

Cesar Augusto
Founder & CEO, ORCA

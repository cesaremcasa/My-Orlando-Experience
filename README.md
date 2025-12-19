# Orlando Experience RAG

A layered retrieval-augmented generation (RAG) system designed to answer traveler questions about Orlando theme parks with factual precision, contextual awareness, and empathetic language. Built as a minimal viable product (MVP) for engineering portfolio demonstration, this system implements strict data layering, atomic fact grounding, and human-centered response generation—all within a CLI-only, CPU-optimized environment.

## Objective

Enable families to plan efficient, stress-free Orlando vacations using verified operational data from official and analytical sources. The system avoids hallucination by grounding all responses in preprocessed documents and separating factual, behavioral, and strategic knowledge into distinct retrieval layers.

## Architecture Overview

The system operates on three isolated knowledge layers:

- **CORE**: Contains atomic factual statements (e.g., park hours, locations, early entry rules). Each chunk represents one entity, one attribute, and one value.
- **CONTEXT_INTELLIGENCE**: Derived from visitor behavior observations. Addresses pain points such as crowd management, heat stress, and family logistics.
- **EXPERIENCE_STRATEGY**: Focuses on trip optimization patterns, including itinerary structuring, wait-time reduction, and seasonal planning.

Each layer is independently indexed using FAISS and can be queried separately to ensure response fidelity.

## Technical Specifications

- **Operating System**: Amazon Linux 2023
- **Instance Type**: t3.xlarge (CPU-only, no GPU)
- **Python Version**: 3.11 (isolated via virtual environment)
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional, optimized for CPU)
- **Vector Database**: FAISS (IndexFlatL2)
- **LLM Backend**: OpenAI `gpt-4o-mini` (via REST API)
- **Key Dependencies**: `faiss-cpu`, `unstructured[pypdf]`, `openai`, `python-dotenv`, `sentence-transformers`, `langchain`, `langchain-community`, `langchain-openai`
- **Deployment Model**: CLI-only, no Docker, no cloud abstractions

## Project Structure
orlando-experience-rag/
├── data/
│   ├── raw_pdfs/               # Source documents (excluded from version control)
│   ├── processed/              # Cleaned text output (.jsonl)
│   ├── core_atomic/            # Atomic facts (strictly validated entries)
│   ├── context_intelligence/   # Visitor behavior insights
│   ├── experience_strategy/    # Strategic trip-planning patterns
│   ├── embeddings/             # Embeddings for each layer (.npy + metadata)
│   └── index/                  # FAISS indexes (one per layer)
├── 01_parse_pdfs.py
├── 07_parse_core_atomic.py
├── 09_rebuild_travel_layers.py
├── 10_validate_atomic_core.py
├── 11_answer_with_llm.py
├── retrieval_pipeline.py
├── generation_log.csv
├── .env.example
├── .gitignore
└── README.md
text## Setup and Installation

1. Create and activate virtual environment:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate

Install dependencies:Bashpip install --upgrade pip
pip install faiss-cpu unstructured[pypdf] openai python-dotenv sentence-transformers langchain langchain-community langchain-openai
Configure API key (only when testing):Bashcp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-...

Data Processing Pipeline
Run scripts in order (they are idempotent):
Bashpython 01_parse_pdfs.py
python 07_parse_core_atomic.py
python 09_rebuild_travel_layers.py
python 10_validate_atomic_core.py  # Validates CORE atomicity
Usage Examples
Bash# Factual query (maximum precision)
python 11_answer_with_llm.py "What time does Magic Kingdom open on December 25, 2025?" --layer core

# Contextual advice
python 11_answer_with_llm.py "How to handle crowds with small children in December?" --layer context

# Strategic planning
python 11_answer_with_llm.py "Best park order for a 4-day trip in December 2025?" --layer strategy

# Combined layers
python 11_answer_with_llm.py "Plan a Christmas Day at Disney for a family with toddlers" --layer all
Pipeline Phases








































PhaseScriptDescriptionIngestion01_parse_pdfs.pyExtracts text from PDFs using unstructured + pypdf fallbackAtomic Parsing07_parse_core_atomic.pyGenerates strict atomic facts from canonical hours documentLayer Construction09_rebuild_travel_layers.pyBuilds travel-relevant chunks for CONTEXT and STRATEGY layersValidation10_validate_atomic_core.pyEnforces one entity / one attribute / one value per CORE chunkRetrievalretrieval_pipeline.pyFAISS top-k retrieval from specified layerGeneration11_answer_with_llm.pyEmpathetic 2-sentence response strictly grounded in retrieved context
Observability & Safety

All queries/responses logged to generation_log.csv
Layer isolation prevents fact contamination
.env excluded from git and should be deleted after testing

Source Documents

Orlando_Park_Hours_Dez2025.pdf → Canonical source for CORE layer (December 2025 hours)
ORLANDO_DEEPSEARCH_KB.pdf → Real visitor experiences and operational challenges
A Strategic DeepResearch Plan for the Orlando Visitor Experience.pdf → Optimization frameworks

Design Philosophy
This MVP demonstrates engineering discipline in constrained environments:

Zero infrastructure abstractions
Pure CLI execution on single EC2 instance
High-fidelity RAG achieved through rigorous data structuring and layer isolation
Fully reproducible, secure, and portfolio-ready

Author
Cesar Augusto
Founder & CEO, ORCA

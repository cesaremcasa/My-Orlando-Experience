#!/usr/bin/env python3
"""
Build two auxiliary FAISS layers for Orlando Experience RAG:

1. CONTEXT_INTELLIGENCE
   - Purpose: Analytical/systemic context to enrich reasoning (does NOT replace facts).
   - Source: 
        "A Strategic DeepResearch Plan for the Orlando Visitor Experience.pdf"
        "ORLANDO_DEEPSEARCH_KB.pdf"

2. EXPERIENCE_STRATEGY
   - Purpose: Strategic insights, patterns, trends (supplements but never replaces facts/context).
   - Source: Same two documents (same content, independent embedding pipeline).

Both layers are built independently using the same embedding model (all-MiniLM-L6-v2)
but stored in separate FAISS indexes and metadata files.

Output:
- data/context_intelligence/context_intelligence.faiss
- data/context_intelligence/context_intelligence_metadata.jsonl
- data/experience_strategy/experience_strategy.faiss
- data/experience_strategy/experience_strategy_metadata.jsonl
"""
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Reuse exact same chunking logic as Phase 4
def simple_token_split(text: str, chunk_size: int = 512, overlap: int = 50) -> list:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
    return chunks

def load_documents_by_name(filenames: list) -> list:
    """Load specific .jsonl files from data/processed/ by base filename."""
    docs = []
    for name in filenames:
        jsonl_path = Path("data/processed") / f"{name}.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                doc = json.loads(f.readline())
                docs.append(doc)
        else:
            print(f"⚠️ Warning: {jsonl_path} not found")
    return docs

def embed_and_save_layer(docs: list, layer_name: str, output_dir: Path):
    """Build FAISS index and metadata for a given layer."""
    output_dir.mkdir(exist_ok=True)
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.max_seq_length = 512

    all_chunks = []
    all_embeddings = []

    for doc in docs:
        chunks = simple_token_split(doc["content"])
        embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False).astype('float32')
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            all_chunks.append({
                "source_file": doc["file_name"],
                "chunk_id": i,
                "text": chunk,
                "char_count": len(chunk)
            })
            all_embeddings.append(emb)

    # Save metadata
    meta_path = output_dir / f"{layer_name}_metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for item in all_chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Save FAISS index
    embeddings_np = np.array(all_embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    faiss.write_index(index, str(output_dir / f"{layer_name}.faiss"))

    print(f"✅ Built {layer_name}: {len(all_chunks)} chunks | dim={embeddings_np.shape[1]}")

def main():
    # Define source documents (by base filename, without .pdf)
    source_docs = [
        "A Strategic DeepResearch Plan for the Orlando Visitor Experience",
        "ORLANDO_DEEPSEARCH_KB"
    ]

    docs = load_documents_by_name(source_docs)

    if not docs:
        print("❌ No source documents found. Ensure filenames match exactly.")
        return

    # Build CONTEXT_INTELLIGENCE layer
    embed_and_save_layer(
        docs=docs,
        layer_name="context_intelligence",
        output_dir=Path("data/context_intelligence")
    )

    # Build EXPERIENCE_STRATEGY layer (same docs, independent pipeline)
    embed_and_save_layer(
        docs=docs,
        layer_name="experience_strategy",
        output_dir=Path("data/experience_strategy")
    )

    print("\n🎯 Auxiliary layers built successfully.")
    print("   - CONTEXT_INTELLIGENCE: enriches factual answers with systemic context.")
    print("   - EXPERIENCE_STRATEGY: provides strategic patterns/trends for higher-order reasoning.")
    print("   - Both are independent and do NOT replace core facts.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 5: Retrieval & Query Pipeline
- Load FAISS index and metadata
- Embed user query
- Perform similarity search
- Return top-k relevant chunks with metadata
"""
import os
import json
import time
import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Paths
INDEX_PATH = "data/index/faiss.index"
METADATA_PATH = "data/embeddings/metadata.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"

def load_resources():
    """Load FAISS index and metadata."""
    index = faiss.read_index(INDEX_PATH)
    metadata = []
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    model = SentenceTransformer(MODEL_NAME)
    return index, metadata, model

def retrieve(query: str, top_k: int = 3) -> list:
    """Perform retrieval and return top-k results."""
    index, metadata, model = load_resources()
    
    # Embed query
    start = time.time()
    query_vec = model.encode([query], convert_to_numpy=True).astype('float32')
    
    # Search
    scores, indices = index.search(query_vec, top_k)
    elapsed = time.time() - start
    
    # Format results
    results = []
    for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if idx == -1:
            continue  # FAISS returns -1 for missing neighbors
        doc = metadata[idx]
        results.append({
            "rank": i + 1,
            "file_name": doc["file_name"],
            "chunk_id": doc["chunk_id"],
            "text": doc["text"][:300] + "..." if len(doc["text"]) > 300 else doc["text"],
            "score": float(score),
            "retrieval_time_sec": round(elapsed, 3)
        })
    return results

def main():
    parser = argparse.ArgumentParser(description="Orlando Experience RAG – Retrieval CLI")
    parser.add_argument("query", type=str, help="Your question about Orlando attractions")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to return")
    args = parser.parse_args()

    results = retrieve(args.query, top_k=args.top_k)
    
    print(f"\n🔍 Query: {args.query}")
    print(f"{'─' * 80}")
    for res in results:
        print(f"[Rank {res['rank']}] {res['file_name']} (score: {res['score']:.2f})")
        print(f"    {res['text']}\n")
    print(f"⏱️  Retrieval time: {results[0]['retrieval_time_sec']}s")

if __name__ == "__main__":
    main()

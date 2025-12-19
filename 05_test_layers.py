#!/usr/bin/env python3
"""
Test retrieval from specific Orlando Experience layers:
- core: data/index/faiss.index (factual)
- context: data/context_intelligence/context_intelligence.faiss (systemic analysis)
- strategy: data/experience_strategy/experience_strategy.faiss (trends/patterns)
"""
import argparse
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL = "all-MiniLM-L6-v2"

def load_index_and_metadata(layer):
    if layer == "core":
        index_path = "data/index/faiss.index"
        meta_path = "data/embeddings/metadata.jsonl"
    elif layer == "context":
        index_path = "data/context_intelligence/context_intelligence.faiss"
        meta_path = "data/context_intelligence/context_intelligence_metadata.jsonl"
    elif layer == "strategy":
        index_path = "data/experience_strategy/experience_strategy.faiss"
        meta_path = "data/experience_strategy/experience_strategy_metadata.jsonl"
    else:
        raise ValueError("Layer must be: core, context, or strategy")
    
    index = faiss.read_index(index_path)
    metadata = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    return index, metadata

def retrieve(query, layer="core", top_k=2):
    index, metadata = load_index_and_metadata(layer)
    model = SentenceTransformer(MODEL)
    query_vec = model.encode([query], convert_to_numpy=True).astype('float32')
    scores, indices = index.search(query_vec, top_k)
    
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        doc = metadata[idx]
        results.append({
            "layer": layer,
            "source": doc.get("file_name", doc.get("source_file")),
            "text_preview": (doc["text"][:250] + "...") if len(doc["text"]) > 250 else doc["text"],
            "score": float(score)
        })
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Question to test")
    parser.add_argument("--layer", choices=["core", "context", "strategy"], default="core")
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()

    print(f"\n🔍 Query: '{args.query}'")
    print(f"   Layer: {args.layer.upper()}")
    print("-" * 70)
    
    results = retrieve(args.query, layer=args.layer, top_k=args.top_k)
    for i, res in enumerate(results, 1):
        print(f"[{i}] Source: {res['source']}")
        print(f"    Preview: {res['text_preview']}")
        print(f"    Score: {res['score']:.2f}\n")

if __name__ == "__main__":
    main()

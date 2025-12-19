#!/usr/bin/env python3
"""
Rebuild CORE FAISS index from atomic facts only.
"""
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

ATOMIC_PATH = Path("data/core_atomic/core_atomic_facts.jsonl")
EMBEDDINGS_DIR = Path("data/embeddings")
INDEX_DIR = Path("data/index")
EMBEDDINGS_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

def load_atomic_facts():
    facts = []
    with open(ATOMIC_PATH, "r", encoding="utf-8") as f:
        for line in f:
            facts.append(json.loads(line))
    return facts

def rebuild_atomic_core():
    facts = load_atomic_facts()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    texts = [fact["text"] for fact in facts]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype('float32')
    
    # Save metadata
    with open(EMBEDDINGS_DIR / "metadata.jsonl", "w", encoding="utf-8") as f:
        for fact in facts:
            f.write(json.dumps(fact, ensure_ascii=False) + "\n")
    
    # Save embeddings
    np.save(EMBEDDINGS_DIR / "embeddings.npy", embeddings)
    
    # Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    
    print(f"✅ Atomic CORE index built with {len(facts)} facts.")

if __name__ == "__main__":
    rebuild_atomic_core()

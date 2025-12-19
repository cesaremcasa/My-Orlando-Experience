#!/usr/bin/env python3
"""
Rebuild CONTEXT_INTELLIGENCE and EXPERIENCE_STRATEGY layers
with travel-planning-focused chunks ONLY.
Filters out non-travel content (revenue, tech specs, brand partnerships).
"""
import json
import re
from pathlib import Path

def is_travel_relevant(text: str) -> bool:
    """Keep only chunks about visitor experience, logistics, crowds, family dynamics."""
    text_lower = text.lower()
    travel_keywords = [
        "visitor", "family", "crowd", "itinerary", "logistics", "wait time", 
        "parking", "transportation", "weather", "fatigue", "burnout", 
        "first-time", "planning", "schedule", "hotel", "dining", "attraction",
        "experience", "trip", "orlando", "disney", "universal", "legoland"
    ]
    irrelevant_keywords = [
        "revenue", "income", "profit", "pin trading", "rfid", "wristband", 
        "brand partnership", "licensing", "gift shop", "merchandise", "epic universe"
    ]
    
    if any(kw in text_lower for kw in irrelevant_keywords):
        return False
    return any(kw in text_lower for kw in travel_keywords)

def extract_clean_chunks_from_kb():
    kb_path = Path("data/processed/ORLANDO_DEEPSEARCH_KB.jsonl")
    if not kb_path.exists():
        print("⚠️ ORLANDO_DEEPSEARCH_KB.jsonl not found")
        return []
    
    with open(kb_path, "r", encoding="utf-8") as f:
        doc = json.loads(f.readline())
    
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", doc["content"])
    cleaned = []
    
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 30 or len(sent) > 300:
            continue
        if is_travel_relevant(sent):
            cleaned.append(sent)
    
    return cleaned

def save_layer(name: str, chunks: list):
    layer_dir = Path(f"data/{name}")
    layer_dir.mkdir(exist_ok=True)
    
    # Save metadata
    with open(layer_dir / f"{name}_metadata.jsonl", "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            f.write(json.dumps({
                "source_file": "ORLANDO_DEEPSEARCH_KB.pdf",
                "chunk_id": i,
                "text": chunk
            }, ensure_ascii=False) + "\n")
    
    # Embed and save FAISS
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False).astype('float32')
    
    np.save(layer_dir / "embeddings.npy", embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(layer_dir / f"{name}.faiss"))
    
    print(f"✅ {name}: {len(chunks)} travel-focused chunks")

def main():
    travel_chunks = extract_clean_chunks_from_kb()
    if not travel_chunks:
        print("❌ No travel-relevant chunks found")
        return
    
    save_layer("context_intelligence", travel_chunks)
    save_layer("experience_strategy", travel_chunks)  # Same content, separate index

if __name__ == "__main__":
    main()

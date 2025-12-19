#!/usr/bin/env python3
"""
Rebuild CORE FACTUAL index including the canonical park hours document.
Excludes analytical documents.
"""
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

FACTUAL_DOCS = [
    "01.22.10_parkingenforce",
    "2012_UNI_brochure",
    "Guia-de-Bolso-Orlando",
    "Happiest-Handbook",
    "MK_0923_EN",
    "Orlando Travel Guide (1)",
    "universal-citywalk-orlando-map",
    "universal-studios-florida-park-map-english",
    "virtual-guide",
    "Orlando_Park_Hours_Dez2025"  # NEW
]

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

def load_factual_docs():
    docs = []
    for name in FACTUAL_DOCS:
        jsonl_path = Path("data/processed") / f"{name}.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                doc = json.loads(f.readline())
                docs.append(doc)
        else:
            print(f"⚠️ Missing: {jsonl_path}")
    return docs

def rebuild_core_index():
    docs = load_factual_docs()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    model.max_seq_length = 512

    all_chunks = []
    all_embeddings = []

    for doc in docs:
        chunks = simple_token_split(doc["content"])
        embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False).astype('float32')
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            all_chunks.append({
                "file_name": doc["file_name"],
                "chunk_id": i,
                "text": chunk,
                "char_count": len(chunk)
            })
            all_embeddings.append(emb)

    # Save
    with open("data/embeddings/metadata.jsonl", "w", encoding="utf-8") as f:
        for item in all_chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    np.save("data/embeddings/embeddings.npy", np.array(all_embeddings).astype('float32'))
    
    index = faiss.IndexFlatL2(np.array(all_embeddings).shape[1])
    index.add(np.array(all_embeddings).astype('float32'))
    faiss.write_index(index, "data/index/faiss.index")

    print(f"✅ Core index rebuilt with {len(all_chunks)} chunks from {len(docs)} documents (including park hours).")

if __name__ == "__main__":
    rebuild_core_index()

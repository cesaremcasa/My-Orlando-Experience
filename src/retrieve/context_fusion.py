import faiss
import json
import sys
import os
from sentence_transformers import SentenceTransformer

class ContextFusionEngine:
    """
    Manages FAISS index and SentenceTransformer model for fast CPU retrieval.
    """
    def __init__(self):
        self.index_path = "data/index/faiss.index"
        self.metadata_path = "data/embeddings/metadata.jsonl"
        self.model_name = "all-MiniLM-L6-v2"
        
        print(f"[ContextFusion] Loading Index from {self.index_path}...")
        self.index = faiss.read_index(self.index_path)
        
        print(f"[ContextFusion] Loading Metadata from {self.metadata_path}...")
        self.metadata = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))
                
        print(f"[ContextFusion] Loading Model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print("[ContextFusion] Initialization Complete.")

    def retrieve(self, query, top_k=3):
        """Embeds query and searches FAISS index."""
        query_vector = self.model.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for idx in indices[0]:
            if idx == -1: continue
            # Retrieve text and create a nice display string
            doc = self.metadata[idx]
            text = doc.get("text", "")
            source = doc.get("source_document", "Unknown Source")
            
            # Clean up source name for display
            if "/" in source: source = source.split("/")[-1]
            
            results.append(text)
            
        return results

import os
import sys
import json
import re
import faiss
import pytest
import numpy as np
from sentence_transformers import SentenceTransformer

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# --- CONFIGURATION ---
INDEX_PATH = "data/index/faiss.index"
METADATA_PATH = "data/embeddings/metadata.jsonl"
GOLDEN_DATA_PATH = "data/core_atomic/core_atomic_facts.jsonl"
TOP_K_RETRIEVAL = 3

def extract_entity_value(retrieved_text: str, expected_entity: str) -> bool:
    """
    Validates that the expected entity appears in the retrieved text.
    Uses regex to find the entity name.
    """
    if re.search(re.escape(expected_entity), retrieved_text, re.IGNORECASE):
        return True
    return False

@pytest.fixture(scope="module")
def retrieval_system():
    """
    Loads the real FAISS index, metadata, and initializes the SentenceTransformer model.
    """
    # 1. Load Real FAISS Index
    if not os.path.exists(INDEX_PATH):
        pytest.fail(f"FAISS Index not found at {INDEX_PATH}")
    index = faiss.read_index(INDEX_PATH)

    # 2. Load Metadata
    metadata = []
    if not os.path.exists(METADATA_PATH):
        pytest.fail(f"Metadata not found at {METADATA_PATH}")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    
    # 3. Initialize SentenceTransformer (Must match the one used to create the index)
    # Verified: all-MiniLM-L6-v2 matches index dimension 384
    model = SentenceTransformer('all-MiniLM-L6-v2')

    return index, metadata, model

@pytest.fixture(scope="module")
def golden_dataset():
    """
    Loads the ground truth atomic facts.
    """
    if not os.path.exists(GOLDEN_DATA_PATH):
        pytest.fail(f"Golden Dataset not found at {GOLDEN_DATA_PATH}")
    
    facts = []
    with open(GOLDEN_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            facts.append(json.loads(line))
    return facts

def test_core_facts_retrieval(retrieval_system, golden_dataset):
    """
    Integration Test:
    - Iterates through the Golden Dataset.
    - Embeds queries using SentenceTransformer (Real).
    - Searches REAL FAISS index.
    - Validates results using Entity Extraction (NO Exact Match).
    """
    index, metadata, model = retrieval_system
    facts = golden_dataset

    print(f"\n--- Starting Integration Test on {len(facts)} Facts ---")

    # Test a subset
    test_sample = facts if len(facts) <= 10 else facts[:10]

    passed_count = 0
    failed_count = 0

    for fact in test_sample:
        entity = fact.get("entity")
        query_text = fact.get("text")
        
        # 1. Embed Query (Real Local Model)
        query_vector = model.encode([query_text], convert_to_numpy=True).astype('float32')

        # 2. Search FAISS Index (Real Retrieval)
        distances, indices = index.search(query_vector, TOP_K_RETRIEVAL)

        # 3. Validate Results
        found_entity = False
        retrieved_texts = []

        for idx in indices[0]:
            if idx == -1: continue
            doc = metadata[idx]
            retrieved_texts.append(doc.get("text", ""))

        # Check if the expected entity exists in any of the top-k retrieved texts
        for text in retrieved_texts:
            if extract_entity_value(text, entity):
                found_entity = True
                break
        
        if found_entity:
            passed_count += 1
            print(f"✅ PASS: Entity '{entity}' found.")
        else:
            failed_count += 1
            print(f"❌ FAIL: Entity '{entity}' NOT found.")
            print(f"   Query: {query_text}")

    print(f"\n--- Test Summary: {passed_count} Passed, {failed_count} Failed ---")
    
    # Assert success rate
    success_rate = passed_count / len(test_sample)
    assert success_rate >= 0.5, f"Retrieval success rate {success_rate:.2f} is below threshold 0.5"

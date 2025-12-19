#!/usr/bin/env python3
"""
Validates that all chunks in data/core_atomic/core_atomic_facts.jsonl
adhere to the atomic factual specification.
"""
import json
from pathlib import Path

CORE_ATOMIC_PATH = Path("data/core_atomic/core_atomic_facts.jsonl")
REQUIRED_FIELDS = {"entity", "attribute", "value", "source_document", "year_or_season", "text"}

def validate_chunk(chunk: dict, idx: int) -> list:
    """Validate a single atomic chunk. Return list of errors."""
    errors = []
    
    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in chunk:
            errors.append(f"Missing field: {field}")
    
    # Check atomicity
    if not isinstance(chunk.get("entity"), str) or not chunk["entity"].strip():
        errors.append("Entity must be a non-empty string")
    
    if not isinstance(chunk.get("attribute"), str) or not chunk["attribute"].strip():
        errors.append("Attribute must be a non-empty string")
    
    if not isinstance(chunk.get("value"), str) or not chunk["value"].strip():
        errors.append("Value must be a non-empty string")
    
    if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
        errors.append("Text must be a non-empty string")
    
    # Check text structure (should be a single factual sentence)
    text = chunk["text"].strip()
    if text.count(".") > 1 or "\n" in text or len(text.split()) > 40:
        errors.append("Text should be a single factual sentence (not a paragraph or list)")

    return [f"[Chunk {idx}] {err}" for err in errors]

def main():
    if not CORE_ATOMIC_PATH.exists():
        print("❌ ERROR: core_atomic_facts.jsonl not found")
        return 1

    with open(CORE_ATOMIC_PATH, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    print(f"🔍 Validating {len(chunks)} atomic chunks...")

    all_errors = []
    for i, chunk in enumerate(chunks, 1):
        errors = validate_chunk(chunk, i)
        all_errors.extend(errors)

    if all_errors:
        print("\n❌ VALIDATION FAILED:")
        for err in all_errors:
            print(f"  - {err}")
        return 1
    else:
        print("✅ ALL CHUNKS ARE ATOMIC AND VALID")
        return 0

if __name__ == "__main__":
    exit(main())

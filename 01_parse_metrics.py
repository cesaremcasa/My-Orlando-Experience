#!/usr/bin/env python3
"""
Phase 3 Observability: Summarize ingestion results.
"""
import json
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

def main():
    files = list(PROCESSED_DIR.glob("*.jsonl"))
    total_chars = 0
    total_words = 0
    for f in sorted(files):
        with open(f, "r") as fp:
            doc = json.loads(fp.readline())
            total_chars += doc["char_count"]
            total_words += doc["word_count"]
            print(f"{doc['file_name']:<40} {doc['char_count']:>8} chars | {doc['word_count']:>6} words")
    print("-" * 70)
    print(f"TOTAL: {len(files)} documents | {total_chars:,} chars | {total_words:,} words")

if __name__ == "__main__":
    main()

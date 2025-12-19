#!/usr/bin/env python3
"""
Phase 3B: Atomic CORE Layer Ingestion
- Parse only factual PDFs
- Extract atomic facts: one entity, one attribute, one value
- Output: data/core_atomic/ as .jsonl with enriched metadata
"""
import json
import re
from pathlib import Path

# Factual PDFs (base names without .pdf)
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
    "Orlando_Park_Hours_Dez2025"
]

# Output dir
ATOMIC_DIR = Path("data/core_atomic")
ATOMIC_DIR.mkdir(exist_ok=True)

def extract_atomic_facts(text: str, source_doc: str) -> list:
    """Extract atomic facts using rule-based parsing."""
    facts = []
    year_or_season = "December 2025"  # Fixed per your canonical doc

    # Normalize text
    text = re.sub(r"\s+", " ", text.strip())

    # Rule 1: Handle canonical hours doc explicitly
    if "Orlando_Park_Hours_Dez2025" in source_doc:
        park_rules = [
            ("Magic Kingdom", r"Magic Kingdom: ([^,]+)"),
            ("EPCOT", r"EPCOT: ([^,]+)"),
            ("Hollywood Studios", r"Hollywood Studios: ([^,]+)"),
            ("Animal Kingdom", r"Animal Kingdom: ([^,]+)"),
            ("Typhoon Lagoon", r"Typhoon Lagoon: ([^,]+)"),
            ("Blizzard Beach", r"Blizzard Beach: ([^,]+)"),
            ("Universal Studios Florida", r"Universal Studios Florida: ([^,]+)"),
            ("Islands of Adventure", r"Islands of Adventure: ([^,]+)"),
            ("Universal Volcano Bay", r"Universal Volcano Bay: ([^,]+)"),
            ("SeaWorld Orlando", r"SeaWorld Orlando: ([^,]+)"),
            ("Aquatica Orlando", r"Aquatica Orlando: ([^,]+)"),
            ("Busch Gardens Tampa", r"Busch Gardens Tampa: ([^,]+)"),
            ("LEGOLAND Florida", r"LEGOLAND Florida \(Winter Haven\): ([^,]+)"),
            ("Peppa Pig Theme Park", r"Peppa Pig Theme Park: ([^,]+)"),
        ]
        
        for park, pattern in park_rules:
            match = re.search(pattern, text)
            if match:
                hours = match.group(1).strip()
                facts.append({
                    "entity": park,
                    "attribute": "operating_hours",
                    "value": hours,
                    "source_document": source_doc,
                    "year_or_season": year_or_season,
                    "text": f"{park} operating hours during {year_or_season}: {hours}."
                })
        
        # Early entry rules
        if "Early Entry" in text:
            facts.append({
                "entity": "Walt Disney World Resort",
                "attribute": "early_entry_rule",
                "value": "30 minutes before official opening for Disney hotel guests",
                "source_document": source_doc,
                "year_or_season": year_or_season,
                "text": "Walt Disney World Resort offers Early Entry (30 minutes before official opening) for hotel guests during December 2025."
            })
        if "Early Park Admission" in text:
            facts.append({
                "entity": "Universal Orlando Resort",
                "attribute": "early_admission_rule",
                "value": "1 hour before official opening for hotel guests",
                "source_document": source_doc,
                "year_or_season": year_or_season,
                "text": "Universal Orlando Resort offers Early Park Admission (1 hour before official opening) for hotel guests during December 2025."
            })
        
        # Locations
        if "Winter Haven" in text:
            facts.append({
                "entity": "LEGOLAND Florida",
                "attribute": "location",
                "value": "Winter Haven, Florida",
                "source_document": source_doc,
                "year_or_season": year_or_season,
                "text": "LEGOLAND Florida is located in Winter Haven, Florida."
            })
            facts.append({
                "entity": "Peppa Pig Theme Park",
                "attribute": "location",
                "value": "Winter Haven, Florida",
                "source_document": source_doc,
                "year_or_season": year_or_season,
                "text": "Peppa Pig Theme Park is located in Winter Haven, Florida."
            })

    # Rule 2: Generic factual extraction (fallback for other PDFs)
    else:
        # Extract sentences ending with periods
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        for sent in sentences:
            # Heuristic: short factual sentences (< 120 chars) likely atomic
            if 20 < len(sent) < 120 and any(kw in sent.lower() for kw in ["open", "close", "hour", "time", "located", "entry"]):
                facts.append({
                    "entity": "unknown",
                    "attribute": "generic_factual",
                    "value": sent,
                    "source_document": source_doc,
                    "year_or_season": "unknown",
                    "text": sent
                })

    return facts

def process_core_atomic():
    all_facts = []
    for doc_name in FACTUAL_DOCS:
        jsonl_path = Path("data/processed") / f"{doc_name}.jsonl"
        if not jsonl_path.exists():
            continue
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            doc = json.loads(f.readline())
        
        facts = extract_atomic_facts(doc["content"], doc["file_name"])
        all_facts.extend(facts)
        print(f"Extracted {len(facts)} atomic facts from {doc['file_name']}")

    # Save as single .jsonl
    with open(ATOMIC_DIR / "core_atomic_facts.jsonl", "w", encoding="utf-8") as f:
        for fact in all_facts:
            f.write(json.dumps(fact, ensure_ascii=False) + "\n")

    print(f"\n✅ Total atomic facts extracted: {len(all_facts)}")
    print(f"Saved to: data/core_atomic/core_atomic_facts.jsonl")

if __name__ == "__main__":
    process_core_atomic()

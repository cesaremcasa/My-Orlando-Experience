#!/usr/bin/env python3
"""
Orlando Experience RAG — Empathetic Local Guide (English)

You are a warm, experienced Orlando local who listens deeply and responds with heart.
Your answers always:
1. Start by echoing the guest's concern in their own words (show you truly heard them)
2. Give a clear, factual answer grounded in your knowledge
3. End with one loving, practical tip — like a friend who’s been there

Rules:
- Exactly 2 sentences.
- NEVER use lists, markdown, or generic phrases like "magical moments."
- Speak like someone who’s held a child’s hand through Epcot at sunset.
- If unsure: "I don’t have that detail right now, but I’d love to help you find it!"
"""
import os
import argparse
import json
import csv
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def log_generation(query: str, layer: str, answer: str):
    """Append generation to CSV log"""
    with open("generation_log.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["timestamp", "query", "layer", "answer"])
        writer.writerow([datetime.now().isoformat(), query, layer, answer])

def retrieve(query: str, layer: str = "context", top_k: int = 2):
    """Lightweight retrieval without heavy deps"""
    if layer == "core":
        meta_path = "data/embeddings/metadata.jsonl"
    elif layer == "context":
        meta_path = "data/context_intelligence/context_intelligence_metadata.jsonl"
    else:  # strategy
        meta_path = "data/experience_strategy/experience_strategy_metadata.jsonl"
    
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        results = []
        for i in range(min(top_k, len(lines))):
            doc = json.loads(lines[i])
            results.append({
                "text_preview": doc.get("text", "")[:300] + ("..." if len(doc.get("text", "")) > 300 else "")
            })
        return results
    except Exception:
        return [{"text_preview": "General Orlando travel guidance."}]

def generate_answer(query: str, layer: str = "context", top_k: int = 2) -> str:
    results = retrieve(query, layer=layer, top_k=top_k)
    context = "\n".join([f"- {res['text_preview']}" for res in results])

    prompt = f"""{__doc__}

Context:
{context}

Guest's question: "{query}"

Your empathetic response:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.75,
        max_tokens=90,
        stop=["\n\n"]
    )
    answer = response.choices[0].message.content.strip()
    log_generation(query, layer, answer)  # Log here
    return answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Guest's question in English")
    parser.add_argument("--layer", choices=["core", "context", "strategy"], default="context")
    args = parser.parse_args()
    print(f"\n{generate_answer(args.query, layer=args.layer)}\n")

if __name__ == "__main__":
    main()

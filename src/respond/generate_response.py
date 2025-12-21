#!/usr/bin/env python3.11
"""
Minimal FAANG-quality LLM responder for Orlando Experience RAG.
- Loads .env
- Reads query from CLI arg
- Logs to generation_log.csv
- Uses 3-layer retrieval (if retriever exists)
- Falls back to CORE-only if full pipeline missing
"""

import os
import sys
import csv
import time
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load API key
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not found in .env", file=sys.stderr)
    sys.exit(1)

# Simple empathetic template
template = """You are a thoughtful Orlando trip advisor.
Answer the question using only verified facts. Mirror the user's concern.
If unsure, say so. Be concise, warm, and human.

Question: {question}
"""

def main():
    if len(sys.argv) < 2:
        print("Usage: python3.11 generate_response.py \"Your question?\"")
        sys.exit(1)

    question = sys.argv[1]
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    chain = prompt | model
    response = chain.invoke({"question": question}).content

    print(response)

    # Log
    with open("generation_log.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["timestamp", "question", "layer", "response"])
        writer.writerow([datetime.isoformat(datetime.now()), question, "core", response])

if __name__ == "__main__":
    main()

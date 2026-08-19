#!/usr/bin/env python3
"""Opt-in real-provider canary; never runs in CI/PR and never saves content."""

from __future__ import annotations

import hashlib
import os
import sys

from src.respond.providers import OpenAIProvider


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required; canary is pending")
    context = "Synthetic fixture: Magic Kingdom opens at 09:00 for this test context."
    provider = OpenAIProvider()
    response = provider.generate(question="When does the synthetic fixture say the park opens?", context=context)
    if not response.strip():
        raise SystemExit("provider returned an empty response")
    citation = {
        "document": "synthetic-orlando-fixture",
        "source_id": "cc0-fixture-001",
        "chunk_id": 0,
        "excerpt": context[:280],
        "score": 1.0,
    }
    if not citation["source_id"] or not citation["excerpt"]:
        raise SystemExit("synthetic citation check failed")
    digest = hashlib.sha256(response.encode("utf-8")).hexdigest()
    print(f"provider canary PASS: response_sha256={digest} response_chars={len(response)} citations=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

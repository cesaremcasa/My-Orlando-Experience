#!/usr/bin/env python3
"""Run the local API path with synthetic retrieval and a deterministic fake provider."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src import api as _api_package  # noqa: F401 - verifies the installed namespace
from src.api import main
from src.respond.providers import FakeProvider
from src.retrieve.contracts import RetrievedChunk


class FixtureRetriever:
    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        del query, top_k
        return [
            RetrievedChunk(
                "Synthetic fixture: Magic Kingdom opens at 09:00 for this test context.",
                "synthetic-orlando-fixture",
                "cc0-fixture-001",
                0,
                1.0,
            )
        ]


def main_check() -> int:
    main.fusion_engine = FixtureRetriever()
    main.provider = FakeProvider()
    with TestClient(main.app) as client:
        response = client.post("/query", json={"question": "When does the fixture open?"})
    if response.status_code != 200:
        raise SystemExit(f"fake quickstart failed with HTTP {response.status_code}")
    body = response.json()
    if body["grounding_status"] != "grounded" or not body["citations"]:
        raise SystemExit("fake quickstart did not produce grounded synthetic citations")
    print("fake quickstart PASS: /query returned grounded synthetic citation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_check())

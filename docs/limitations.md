# Known limitations

- GCP is not provisioned. Vertex classes are fail-closed integration contracts plus test fakes, not a production Memory Bank/BYOR client.
- Vertex AI Memory Bank is Pre-GA. A real SDK-backed client needs a separate approval-gated PR and live GCP canary.
- Production local embeddings use SentenceTransformer on first memory use; CI uses FakeEmbedder and does not download a model.
- FAISS for AgentOps is in-memory per search, not a persisted index.
- ADK `SequentialAgent`/`ParallelAgent` emit deprecation warnings; topology is preserved.
- Grounding is lexical overlap, not an independent truth guarantee.
- Phoenix Compose is optional and local.
- Locked third-party advisories (langchain, transformers, etc.) are recorded by pip-audit and not silently upgraded.

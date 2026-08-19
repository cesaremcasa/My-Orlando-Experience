# Known limitations

- GCP is not provisioned. Vertex adapters are interfaces plus fakes.
- Vertex AI Memory Bank is Pre-GA.
- Production local embeddings use SentenceTransformer on first memory use; CI uses FakeEmbedder and does not download a model.
- FAISS for AgentOps is in-memory per search, not a persisted index.
- ADK `SequentialAgent`/`ParallelAgent` emit deprecation warnings; topology is preserved.
- Grounding is lexical overlap, not an independent truth guarantee.
- Phoenix Compose is optional and local.
- Locked third-party advisories (langchain, transformers, etc.) are recorded by pip-audit and not silently upgraded.

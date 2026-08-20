# Architecture (v0.4.1)

My Orlando AgentOps is a local-first FastAPI service (portfolio-grade reference, not a provisioned deployment):

1. `POST /query` — grounded retrieval baseline (unchanged).
2. `POST /agent/chat` — ADK Sequential/Parallel graph: retrieval+memory in parallel, Grok `LiteLlm` response, structured safety, memory write only after PASS.
3. `POST /feedback` — allowlisted beta feedback in local SQLite.
4. Optional Phoenix/OTLP export when `PHOENIX_COLLECTOR_ENDPOINT` is set.

Memory: SQLite is authoritative. Local search rebuilds an in-memory FAISS `IndexFlatIP` from user-filtered rows. Vertex Memory Bank is a fail-closed unprovisioned Pre-GA scaffold, not a working production client.

Provider: xAI Grok only. A real Grok canary is pending. Vertex BYOR eval is a fail-closed contract that would consume Grok responses if later implemented. Gemini is not a production fallback.

GCP: **NOT PROVISIONED / APPROVAL REQUIRED**. Informational identity may be `orlando-506100` / `us-central1`. Billing is not enabled. Agent Engine ID is unset. Local backends remain selected.

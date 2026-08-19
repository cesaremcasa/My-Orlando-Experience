# ADR 0001: Local-first AgentOps with Grok, optional Vertex adapters

## Decision
Ship a working local AgentOps path (SQLite + in-memory FAISS + Google ADK + xAI Grok). Keep Vertex Memory Bank and Vertex BYOR eval behind explicit backends. Do not provision GCP in this release.

## Why
The objective is working evidence, not architectural ceremony. Memory Bank is Pre-GA. Grok remains the only response generator.

## Consequences
- Local is the default (`ORLANDO_MEMORY_BACKEND=local`, `ORLANDO_EVAL_BACKEND=local`).
- Vertex adapters initialize only when selected and fail closed if not provisioned.
- Cloud Run packaging and a dry-run preflight exist; no `gcloud` mutation is performed.

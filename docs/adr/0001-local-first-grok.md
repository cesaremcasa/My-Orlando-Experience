# ADR 0001: Local-first AgentOps with Grok, optional Vertex adapters

## Decision
Ship a working local AgentOps path (SQLite + in-memory FAISS + Google ADK + xAI Grok). Keep Vertex Memory Bank and Vertex BYOR eval as fail-closed unprovisioned scaffolds behind explicit backends. Do not provision GCP in this release.

## Why
The objective is working evidence, not architectural ceremony. Memory Bank is Pre-GA. Grok remains the only response generator.

## Consequences
- Local is the default (`ORLANDO_MEMORY_BACKEND=local`, `ORLANDO_EVAL_BACKEND=local`). `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` are informational only.
- Vertex scaffolds initialize only when selected and fail closed if not provisioned.
- Cloud Run packaging and a read-only preflight exist; no `gcloud` mutation is performed. Cloud Run has no deploy dry-run in the installed SDK.

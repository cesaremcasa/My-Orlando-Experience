# Security and privacy

- No secrets in git. `.env` is gitignored. Never print `XAI_API_KEY`.
- Future Cloud Run deployments should load `XAI_API_KEY` from Secret Manager only.
- Default `ORLANDO_TRACE_CONTENT=0`. Spans may hold IDs, hashes, sizes, scores, status, and latency — not prompts, responses, retrieved text, memory content, feedback reasons, or keys.
- Memory is isolated by `X-Beta-User`. SQLite is authoritative. FAISS is rebuilt from user-filtered rows.
- HTTP failures are sanitized. Feedback stores no real PII; only synthetic beta identities.
- Vertex Memory Bank is Pre-GA and not provisioned.

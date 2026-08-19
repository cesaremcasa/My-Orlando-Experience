# SLOs (local target)

| Signal | Target |
|---|---|
| `/health` and `/agent/health` | 200 without loading FAISS, Grok, Phoenix, or GCP |
| Fake `/agent/chat` | grounded synthetic fixture, sanitized errors |
| Timeout | `ORLANDO_AGENT_TIMEOUT_SECONDS` default 30s → 504 |
| Concurrency | `ORLANDO_AGENT_MAX_CONCURRENCY` default 8 |
| Cross-user memory | zero leakage |
| Citation validity (eval fixture) | 100% on grounded cases |
| Abstention (no-evidence fixture) | 100% |

These are local evidence targets, not Cloud Run SLOs. GCP is not provisioned.

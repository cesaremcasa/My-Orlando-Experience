# Evaluation methodology

Local CLI: `uv run python scripts/eval_agentops.py --seed 42`.

- 20 synthetic CC0 cases covering grounded answers, abstention, citations, memory, isolation, injection, and failures.
- Metrics: citation validity, groundedness, abstention correctness, memory precision@k, tool success, safety pass rate, p50, p95.
- Same seed and inputs produce the same case outcomes. Latency percentiles may jitter.
- Vertex BYOR eval, if selected later, receives Grok responses already generated. It does not call Gemini.
- Phoenix dataset upload is optional and local-only.

# Evaluation methodology

Local CLI: `orlando-agentops-eval --seed 42 --out <dir>` or `uv run python scripts/eval_agentops.py --seed 42`.

- 20 synthetic CC0 cases covering grounded answers, abstention, citations, memory, isolation, injection, and failures.
- `citation_validity`: grounded fixture cases with valid citations.
- `grounded_case_success_rate`: fixture outcome agreement for expected-grounded cases. Deprecated alias: `groundedness` (not an independent groundedness judge).
- `abstention_correctness`: no-evidence cases abstain.
- `memory_precision_at_k`: relevant retrieved labels / retrieved memories.
- `memory_no_hit_accuracy`: expected empty-hit cases.
- `tool_outcome_accuracy`: actual tool ok/error vs expected (expected failures count as correct).
- `safety_decision_accuracy`: public pass/abstain/error vs fixture expected decision.
- `p50_ms` / `p95_ms`: wall-clock latency, not byte-deterministic.
- Same seed produces the same case IDs, decisions, citations, memory outcomes, and non-timing metrics.
- Vertex BYOR eval remains an unprovisioned scaffold. Phoenix dataset upload is optional and local-only. Live Phoenix collector is not yet canaried.

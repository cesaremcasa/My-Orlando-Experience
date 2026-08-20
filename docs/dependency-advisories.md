# Dependency advisory triage

Machine-readable register: `docs/dependency-advisories.json`.

CI runs `scripts/audit_triage.py` against the locked requirements. The job is green only when every pip-audit finding is in that register. New IDs fail the build. This is **accepted advisories**, not a clean bill of health.

Review deadline: 2026-09-30.

Upgraded in v0.4.1:

- `python-dotenv` 1.0.1 → 1.2.2 (direct base; `set_key` symlink advisory)
- `pytest` 8.3.5 → 9.0.3 (direct dev; `/tmp/pytest-of-{user}` advisory)

Not upgraded:

- `langchain-core` / `langchain-openai` remain in the `rag` extra because `src/respond/providers.py` uses `ChatOpenAI` and `ChatPromptTemplate` for `/query`. Demonstrated AgentOps runtime does not use LangChain serialization, prompt loading, or image token counting. A compatible LangChain 1.2.x jump is deferred to an approval-gated follow-up.
- `transformers` remains a `sentence-transformers` transitive pin. High-severity findings require a 5.x migration that is out of scope.

Default container installs `.[agentops]` and does not include LangChain.

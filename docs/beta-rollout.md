# Beta rollout

Default allowlist: `ORLANDO_BETA_USERS=beta-001,beta-002,beta-003`.

- `/agent/chat` requires `X-Beta-User`.
- `/feedback` requires an allowlisted `X-Beta-User`; others receive 403.
- Identities are synthetic. No PII. Feedback is local SQLite, idempotent on `(beta_user, response_id)`.
- Real Grok remains a manual canary. Fake runtime (`ORLANDO_FAKE_RUNTIME=1`) is for local/container evidence only.

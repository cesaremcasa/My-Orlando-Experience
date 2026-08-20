# GCP approval gate

**Status: NOT PROVISIONED / APPROVAL REQUIRED**

Operator-stated facts (not verified by repository code):

- `GOOGLE_CLOUD_PROJECT=orlando-506100` is informational and is not required for local mode
- `GOOGLE_CLOUD_LOCATION=us-central1`
- `GOOGLE_CLOUD_AGENT_ENGINE_ID` is unset
- billing is not enabled
- Vertex AI API is not intentionally enabled
- no Agent Engine, Memory Bank, or Cloud Run service exists

Print the gate:

```bash
orlando-agentops-gcp-preflight
# or: uv run python scripts/gcp_preflight.py
```

The command never executes `gcloud run deploy`. Cloud Run has no deploy dry-run in the installed Google Cloud SDK. The printed command is labeled `PROPOSED COMMAND — DO NOT RUN WITHOUT APPROVAL`.

The command does not run `gcloud auth login`, enable APIs, deploy Cloud Run, create service accounts, write Secret Manager secrets, or create Vertex Agent Engine / Memory Bank resources. Vertex remains a fail-closed unprovisioned scaffold.

XAI_API_KEY, if used in a future authorized Cloud Run service, must come from Secret Manager. This repository does not write that secret.

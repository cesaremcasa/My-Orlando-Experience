# GCP approval gate

**Status: NOT PROVISIONED / APPROVAL REQUIRED**

Print the gate:

```bash
uv run python scripts/gcp_preflight.py
```

The command does not run `gcloud auth login`, enable APIs, deploy Cloud Run, create service accounts, write Secret Manager secrets, or create Vertex Agent Engine / Memory Bank resources.

XAI_API_KEY, if used in a future authorized Cloud Run service, must come from Secret Manager. This repository does not write that secret.

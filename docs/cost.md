# Cost posture

Nothing in v0.4.0 creates billable GCP resources.

If later approved, cost drivers would be Cloud Run CPU/RAM, Artifact Registry, Secret Manager, Vertex Agent Engine, Vertex Memory Bank (Pre-GA), and xAI Grok tokens. Run `uv run python scripts/gcp_preflight.py` for the approval list. Do not enable APIs or deploy without that approval.

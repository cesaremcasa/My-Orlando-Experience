# Cost posture

Nothing in v0.4.0 or the v0.4.1 evidence PR creates billable GCP resources. Project `orlando-506100` exists as identity only. Billing is not enabled.

If later approved, cost drivers would be Cloud Run CPU/RAM, Artifact Registry, Secret Manager, Vertex Agent Engine, Vertex Memory Bank (Pre-GA), and xAI Grok tokens. Run `orlando-agentops-gcp-preflight` for the approval list. Do not enable APIs or deploy without that approval. A separate approval-gated PR is required for billing, Vertex AI API enablement, Agent Engine creation, a real SDK-backed client, IAM, Memory Bank canary, and BYOR evaluation canary.

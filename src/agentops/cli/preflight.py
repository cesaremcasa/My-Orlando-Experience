from __future__ import annotations

from src.agentops.settings import DEFAULT_GCP_LOCATION, vertex_config


def render() -> str:
    cfg = vertex_config()
    project = cfg["project"] or "<GOOGLE_CLOUD_PROJECT>"
    location = cfg["location"] or DEFAULT_GCP_LOCATION
    engine = cfg["engine_id"] or "<GOOGLE_CLOUD_AGENT_ENGINE_ID>"
    lines = [
        "My Orlando AgentOps v0.4.1 GCP preflight (read-only)",
        "GCP STATUS: NOT PROVISIONED / APPROVAL REQUIRED",
        "Vertex client implementation: fail-closed unprovisioned scaffold (Pre-GA Memory Bank).",
        "",
        "Required APIs (not enabled by this command):",
        "- run.googleapis.com",
        "- aiplatform.googleapis.com",
        "- secretmanager.googleapis.com",
        "- artifactregistry.googleapis.com",
        "- logging.googleapis.com",
        "",
        "Required IAM roles (not granted by this command):",
        "- roles/run.admin",
        "- roles/secretmanager.secretAccessor",
        "- roles/aiplatform.user",
        "- roles/logging.logWriter",
        "- roles/artifactregistry.writer",
        "",
        "Expected Secret Manager secret names (not written by this command):",
        "- XAI_API_KEY",
        "",
        "Intended region/project configuration:",
        f"- GOOGLE_CLOUD_PROJECT={project}",
        f"- GOOGLE_CLOUD_LOCATION={location}",
        f"- GOOGLE_CLOUD_AGENT_ENGINE_ID={engine}",
        "",
        "PROPOSED COMMAND — DO NOT RUN WITHOUT APPROVAL:",
        f"gcloud run deploy orlando-agentops --image=<image> --region={location} "
        f"--project={project} --no-allow-unauthenticated",
        "",
        "Cloud Run has no deploy dry-run in the installed Google Cloud SDK.",
        "This preflight never executes gcloud run deploy.",
        "",
        "Estimated components requiring cost approval:",
        "- Cloud Run service",
        "- Artifact Registry storage",
        "- Secret Manager",
        "- Vertex AI Agent Engine (Pre-GA)",
        "- Vertex AI Memory Bank (Pre-GA)",
        "- optional Cloud Logging",
        "",
        "STOP. This preflight does not run:",
        "- gcloud auth login",
        "- gcloud services enable",
        "- gcloud run deploy",
        "- service account creation",
        "- Secret Manager writes",
        "- Vertex Agent Engine creation",
        "- Vertex Memory Bank creation",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    print(render(), end="")
    return 0

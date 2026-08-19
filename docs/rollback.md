# Rollback

This release is a git tag and draft GitHub release only.

- Revert application traffic by deploying the previous artifact (`v0.3.0`) if a later GCP deploy exists. None exists now.
- Local rollback: check out the previous main SHA and reinstall `.[rag,agentops]`.
- Memory SQLite lives in `ORLANDO_AGENTOPS_DATA_DIR` (default `./.agentops`). Keep a copy before destructive experiments.
- Do not force-push `main`. Do not rewrite tags that have artifacts attached.

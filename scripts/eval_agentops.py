#!/usr/bin/env python3
"""Deterministic local AgentOps evaluation CLI. Not an administrative HTTP API."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.agentops.evals.harness import DEFAULT_SEED, run_eval, write_reports
from src.agentops.settings import data_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic AgentOps evaluation harness.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--upload-phoenix", action="store_true", help="Optional local Phoenix upload.")
    args = parser.parse_args()
    out_dir = args.out or (data_dir() / "evals")
    report = asyncio.run(
        run_eval(seed=args.seed, data_root=data_dir() / "eval-memory", upload_phoenix=args.upload_phoenix)
    )
    json_path, md_path = write_reports(report, out_dir)
    metrics = report["metrics"]
    if metrics["citation_validity"] < 1.0 or metrics["abstention_correctness"] < 1.0 or metrics["cross_user_leaks"]:
        raise SystemExit(
            f"evaluation failed citation_validity={metrics['citation_validity']} "
            f"abstention_correctness={metrics['abstention_correctness']} "
            f"cross_user_leaks={metrics['cross_user_leaks']}"
        )
    print(f"eval PASS: {json_path} {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Thin wrapper around the installed orlando-agentops-gcp-preflight command."""

from src.agentops.cli.preflight import main

if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Thin wrapper around the installed orlando-agentops-quickstart command."""

from src.agentops.cli.quickstart import main

if __name__ == "__main__":
    raise SystemExit(main())

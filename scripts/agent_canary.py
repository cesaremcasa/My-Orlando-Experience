#!/usr/bin/env python3
"""Thin wrapper around the installed orlando-agentops-grok-canary command."""

from src.agentops.cli.grok_canary import main

if __name__ == "__main__":
    raise SystemExit(main())

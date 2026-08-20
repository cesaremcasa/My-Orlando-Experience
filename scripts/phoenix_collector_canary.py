#!/usr/bin/env python3
"""Thin wrapper around the local Phoenix collector canary."""

from src.agentops.cli.phoenix_canary import main

if __name__ == "__main__":
    raise SystemExit(main())

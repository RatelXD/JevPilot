"""Compatibility entry point for the explicitly live two-mode runner."""
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# How to run:
#   uv run python benchmarks/tasks/run_minimal_slice.py --mode compare --task T-00 --repeat 1 --live

import sys
from pathlib import Path

import anyio

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.tasks.run_suite import main

if __name__ == "__main__":
    raise SystemExit(anyio.run(main))

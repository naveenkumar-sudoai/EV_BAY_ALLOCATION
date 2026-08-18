#!/usr/bin/env python3
"""Reproducibility entry point.

Runs the experiment suite.  Usage::

    .venv/bin/python run_experiments.py --experiment all
    .venv/bin/python run_experiments.py --experiment a
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ev_sim.experiments import main  # noqa: E402

if __name__ == "__main__":
    main()

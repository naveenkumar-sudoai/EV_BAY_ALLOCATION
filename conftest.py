"""Pytest bootstrap.

Makes the ``src/`` layout importable as ``ev_sim`` so the test suite can run
without a separate install step. Running ``pytest`` from the repository root
will pick this file up and prepend ``src`` to ``sys.path``.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

"""Summary metrics computed over the raw per-car records.

All functions here are pure: they operate on arrays/lists and do not import the
simulation engine, keeping the metric layer trivially unit-testable and
reusable by the experiment runner.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def jains_fairness(values: Sequence[float]) -> float:
    """Jain's fairness index.

    ``J = (sum x)^2 / (n * sum x^2)``, bounded in ``[1/n, 1]`` with ``1``
    meaning all values are identical (perfectly fair).  An all-zero input is
    treated as perfectly fair (everyone waited equally).  Returns ``NaN`` for an
    empty input.
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    if n == 0:
        return float("nan")
    sum_x = float(x.sum())
    sum_x2 = float((x ** 2).sum())
    if sum_x2 == 0.0:
        return 1.0
    return (sum_x ** 2) / (n * sum_x2)


def summarize(
    wait_times: Sequence[float],
    bay_utilization: Sequence[float],
    starvation_threshold: float = 30.0,
    n_censored: int = 0,
) -> dict:
    """One-shot summary of a single simulation replicate.

    ``n_censored`` is the number of cars still queued (never started) at the
    horizon — their wait time is right-censored and excluded from the wait
    metrics.  Returns a flat dict of the headline metrics used by all
    experiments.
    """
    wt = np.asarray(wait_times, dtype=float)
    util = np.asarray(bay_utilization, dtype=float)

    if wt.size == 0:
        return {
            "n_served": 0,
            "n_censored": int(n_censored),
            "mean_wait_min": float("nan"),
            "median_wait_min": float("nan"),
            "max_wait_min": float("nan"),
            "jain_fairness": float("nan"),
            "starvation_count": 0,
            "mean_bay_utilization": float(np.mean(util)) if util.size else float("nan"),
        }

    return {
        "n_served": int(wt.size),
        "n_censored": int(n_censored),
        "mean_wait_min": float(wt.mean()),
        "median_wait_min": float(np.median(wt)),
        "max_wait_min": float(wt.max()),
        "jain_fairness": jains_fairness(wt),
        "starvation_count": int((wt >= starvation_threshold).sum()),
        "mean_bay_utilization": float(np.mean(util)) if util.size else float("nan"),
    }


def bay_utilization_fraction(busy_times: Sequence[float], active_duration: float) -> list[float]:
    """Per-bay utilization fraction over the steady-state window."""
    if active_duration <= 0.0:
        return [float("nan")] * len(busy_times)
    return [float(t / active_duration) for t in busy_times]

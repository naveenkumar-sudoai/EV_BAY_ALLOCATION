"""Unit tests for the metrics layer."""
from __future__ import annotations

import math

import numpy as np
import pytest

from ev_sim.metrics import bay_utilization_fraction, jains_fairness, summarize


def test_jain_perfectly_equal_is_one():
    assert jains_fairness([1, 1, 1]) == pytest.approx(1.0)
    assert jains_fairness([2.0, 2.0]) == pytest.approx(1.0)


def test_jain_known_reference():
    # For [1, 2, 3]: sum=6, sum_sq=14 -> 36 / (3*14) = 0.857142...
    assert jains_fairness([1, 2, 3]) == pytest.approx(36.0 / 42.0)


def test_jain_all_zero_is_one():
    assert jains_fairness([0.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_jain_single_element_is_one():
    assert jains_fairness([5.0]) == pytest.approx(1.0)


def test_jain_empty_is_nan():
    assert math.isnan(jains_fairness([]))


def test_jain_decreases_with_inequality():
    # Adding a large outlier must lower fairness.
    assert jains_fairness([10, 10, 10, 100]) < jains_fairness([10, 10, 10, 11])


def test_jain_is_bounded_below_by_one_over_n():
    x = [0.0, 0.0, 0.0, 100.0]
    assert jains_fairness(x) >= 1.0 / len(x)


def test_summarize_starvation_count():
    s = summarize(wait_times=[5, 10, 35, 40], bay_utilization=[0.5, 0.5],
                  starvation_threshold=30.0)
    assert s["starvation_count"] == 2
    assert s["mean_wait_min"] == pytest.approx(22.5)
    assert s["median_wait_min"] == pytest.approx(22.5)
    assert s["max_wait_min"] == 40.0
    assert s["n_served"] == 4
    assert s["mean_bay_utilization"] == pytest.approx(0.5)


def test_summarize_empty_wait_times():
    s = summarize(wait_times=[], bay_utilization=[0.3], n_censored=7)
    assert s["n_served"] == 0
    assert s["n_censored"] == 7
    assert math.isnan(s["mean_wait_min"])


def test_bay_utilization_fraction():
    assert bay_utilization_fraction([100.0, 50.0], 200.0) == pytest.approx([0.5, 0.25])


def test_bay_utilization_zero_duration_is_nan():
    out = bay_utilization_fraction([10.0], 0.0)
    assert math.isnan(out[0])

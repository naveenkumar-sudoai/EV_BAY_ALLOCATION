"""Unit tests for the Car/Bay data models and vehicle-mix sampling."""
from __future__ import annotations

import numpy as np
import pytest

from ev_sim.models import (
    BATTERY_TIERS,
    Bay,
    Car,
    sample_battery_capacity,
    sample_max_accept_kw,
)


def test_kwh_needed_computation():
    car = Car(0, 0.0, 40.0, 30.0, 100.0, 50.0)
    assert car.kwh_needed == pytest.approx(40.0 * 0.70)


def test_kwh_needed_zero_when_target_below_initial():
    car = Car(0, 0.0, 40.0, 60.0, 50.0, 50.0)
    assert car.kwh_needed == 0.0


def test_wait_and_total_time_properties():
    car = Car(0, arrival_time=10.0, battery_capacity_kwh=40.0,
              initial_soc_pct=20.0, target_soc_pct=100.0, max_accept_kw=50.0)
    assert car.wait_time is None  # not served yet
    car.service_start_time = 15.0
    car.service_end_time = 55.0
    assert car.wait_time == pytest.approx(5.0)
    assert car.charge_time == pytest.approx(40.0)
    assert car.total_time_in_system == pytest.approx(45.0)
    assert car.served is True


def test_bay_occupancy_state_machine():
    bay = Bay(0, 60.0)
    assert bay.is_free(0.0) is True
    car = Car(1, 0.0, 40.0, 20.0, 100.0, 50.0)
    bay.occupy(car, free_time=45.0)
    assert bay.currently_occupied is True
    assert bay.current_car is car
    assert bay.is_free(30.0) is False
    # Occupied until release runs, even once predicted_free_time is reached.
    assert bay.is_free(45.0) is False
    bay.release(45.0)
    assert bay.currently_occupied is False
    assert bay.current_car is None
    assert bay.predicted_free_time == 45.0
    assert bay.is_free(45.0) is True


def test_tier_weights_sum_to_one():
    total = sum(t[1] for t in BATTERY_TIERS)
    assert total == pytest.approx(1.0)
    names = [t[0] for t in BATTERY_TIERS]
    assert names == ["small", "mid", "premium"]


def test_market_split_weights_are_50_35_15():
    weights = {t[0]: t[1] for t in BATTERY_TIERS}
    assert weights["mid"] == pytest.approx(0.50)
    assert weights["small"] == pytest.approx(0.35)
    assert weights["premium"] == pytest.approx(0.15)


def test_sample_battery_capacity_respects_tier_ranges():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        cap, tier = sample_battery_capacity(rng)
        lo, hi = {t[0]: (t[2], t[3]) for t in BATTERY_TIERS}[tier]
        assert lo <= cap <= hi


def test_sample_market_split_is_roughly_correct():
    rng = np.random.default_rng(42)
    counts = {"small": 0, "mid": 0, "premium": 0}
    n = 20_000
    for _ in range(n):
        _, tier = sample_battery_capacity(rng)
        counts[tier] += 1
    # Tolerances loose enough to avoid flakiness at n=20k.
    assert counts["mid"] / n == pytest.approx(0.50, abs=0.03)
    assert counts["small"] / n == pytest.approx(0.35, abs=0.03)
    assert counts["premium"] / n == pytest.approx(0.15, abs=0.03)


def test_sample_max_accept_kw_respects_tier_ranges():
    rng = np.random.default_rng(1)
    for name, _, _, _, lo, hi in BATTERY_TIERS:
        for _ in range(200):
            kw = sample_max_accept_kw(rng, name)
            assert lo <= kw <= hi


def test_sample_max_accept_kw_rejects_unknown_tier():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_max_accept_kw(rng, "nope")

"""Unit tests for the piecewise CC/CV charge-curve model."""
from __future__ import annotations

import math

import pytest

from ev_sim.charge_curve import (
    CC_SOC_PCT,
    TAIL_FACTOR,
    time_from_0_to_soc_hours,
    time_to_charge,
)


# --- reference points -------------------------------------------------------
# A 40 kWh battery at 40 kW effective power: T_full = 1 h = 60 min.
CAP = 40.0
PWR = 40.0
T_FULL_MIN = 60.0


def test_zero_to_eighty_is_linear_cc_regime():
    # CC regime: 0 -> 80% takes 0.8 * T_full.
    assert time_to_charge(0, 0, 80, PWR, CAP) == pytest.approx(0.8 * T_FULL_MIN)


def test_zero_to_seventy_reference():
    # First 70% is pure CC: 0.7 * T_full.
    assert time_to_charge(0, 0, 70, PWR, CAP) == pytest.approx(0.7 * T_FULL_MIN)


def test_full_charge_is_one_and_a_half_times_nominal():
    # 0 -> 100% = CC (0.8) + CV (0.7) = 1.5 * T_full.
    assert time_to_charge(0, 0, 100, PWR, CAP) == pytest.approx(1.5 * T_FULL_MIN)


def test_last_twenty_percent_equals_first_seventy_percent():
    """The defining calibration: last 20% takes as long as the first 70%."""
    last_20 = time_to_charge(0, 80, 100, PWR, CAP)
    first_70 = time_to_charge(0, 0, 70, PWR, CAP)
    assert last_20 == pytest.approx(first_70)
    assert last_20 == pytest.approx(TAIL_FACTOR * T_FULL_MIN)


def test_curve_is_monotonic_in_soc():
    """time_from_0_to_soc must be non-decreasing across the whole SOC range."""
    times = [
        time_from_0_to_soc_hours(soc, PWR, CAP)
        for soc in range(0, 101)
    ]
    assert all(b >= a for a, b in zip(times, times[1:]))


def test_curve_is_strictly_increasing_away_from_zero_power():
    times = [time_from_0_to_soc_hours(soc, PWR, CAP) for soc in range(0, 101)]
    assert all(b > a for a, b in zip(times, times[1:]))


def test_cv_taper_is_slower_than_cc():
    """The last 20% (80->100) must be slower per-percent than the first 80%."""
    # per-percent time in CC regime:
    cc_per_pct = time_to_charge(0, 0, 80, PWR, CAP) / 80.0
    # per-percent time in CV regime:
    cv_per_pct = time_to_charge(0, 80, 100, PWR, CAP) / 20.0
    assert cv_per_pct > cc_per_pct


def test_higher_power_reduces_time():
    assert time_to_charge(0, 0, 100, 80.0, CAP) < time_to_charge(0, 0, 100, 40.0, CAP)


def test_target_below_current_is_zero():
    assert time_to_charge(0, 50, 40, PWR, CAP) == 0.0


def test_target_equal_current_is_zero():
    assert time_to_charge(0, 80, 80, PWR, CAP) == 0.0


def test_zero_effective_power_is_infinite():
    assert time_to_charge(0, 20, 100, 0.0, CAP) == math.inf
    assert time_to_charge(0, 20, 100, -1.0, CAP) == math.inf


def test_soc_is_clamped_to_full():
    # Requesting >100% should clamp to the full-charge time, not extrapolate.
    assert time_to_charge(0, 0, 120, PWR, CAP) == pytest.approx(1.5 * T_FULL_MIN)


def test_current_above_knee_uses_cv_only():
    # Charging 90 -> 100 lives entirely inside the CV taper and must be positive
    # but far less than the whole CV window.
    t = time_to_charge(0, 90, 100, PWR, CAP)
    assert t > 0
    assert t < time_to_charge(0, 80, 100, PWR, CAP)


def test_additivity_of_intervals():
    """time_to_charge should be additive over adjacent SOC intervals."""
    a = time_to_charge(0, 20, 60, PWR, CAP)
    b = time_to_charge(0, 60, 90, PWR, CAP)
    total = time_to_charge(0, 20, 90, PWR, CAP)
    assert a + b == pytest.approx(total)


def test_kwh_needed_parameter_is_consistent_with_soc_delta():
    """The kwh_needed argument must match the SOC/capacity definition."""
    kwh = CAP * (100 - 40) / 100.0
    assert time_to_charge(kwh, 40, 100, PWR, CAP) == pytest.approx(
        time_to_charge(0, 40, 100, PWR, CAP)
    )

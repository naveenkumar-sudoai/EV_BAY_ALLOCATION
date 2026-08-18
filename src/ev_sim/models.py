"""Core data models for the EV bay allocation simulation.

Two plain dataclasses — :class:`Car` and :class:`Bay` — plus the sampling
helpers used to draw a realistic mix of vehicles from the Indian EV market.

The battery-capacity mix is a three-tier approximation:

* **small** hatchback tier  (~19–24 kWh)  — e.g. Tiago EV, eC3          — 35%
* **mid**   compact-SUV tier (~30–40 kWh)  — e.g. Nexon EV, Punch EV    — 50%
* **premium** SUV tier      (~65–75 kWh)  — e.g. eC9, XUV400 long range — 15%

The mid tier is most common, small second, premium rarest (50/35/15), matching
the request.  ``max_accept_kw`` is drawn per tier so that low-acceptance
vehicles cannot use a high-power bay's full rating — this is what makes
*capacity matching* (matching a car to a bay it can actually exploit) a
meaningful part of the allocation problem.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Vehicle mix
# ---------------------------------------------------------------------------
# Each tier: (name, weight, cap_min_kwh, cap_max_kwh, accept_min_kw, accept_max_kw)
BATTERY_TIERS: tuple[tuple[str, float, float, float, float, float], ...] = (
    ("small", 0.35, 19.0, 24.0, 7.0, 22.0),
    ("mid", 0.50, 30.0, 40.0, 30.0, 80.0),
    ("premium", 0.15, 65.0, 75.0, 80.0, 150.0),
)

_TIER_NAMES = [t[0] for t in BATTERY_TIERS]
_TIER_WEIGHTS = np.array([t[1] for t in BATTERY_TIERS], dtype=float)
_TIER_WEIGHTS /= _TIER_WEIGHTS.sum()


def sample_tier_index(rng: np.random.Generator) -> int:
    """Draw a tier index proportional to the configured market weights."""
    return int(rng.choice(len(BATTERY_TIERS), p=_TIER_WEIGHTS))


def sample_battery_capacity(rng: np.random.Generator) -> tuple[float, str]:
    """Sample ``(capacity_kwh, tier_name)`` from the market mix."""
    i = sample_tier_index(rng)
    _, _, lo, hi, _, _ = BATTERY_TIERS[i]
    return float(rng.uniform(lo, hi)), _TIER_NAMES[i]


def sample_max_accept_kw(rng: np.random.Generator, tier_name: str) -> float:
    """Sample the vehicle's maximum accepted charging power (kW) for a tier."""
    for name, _, _, _, lo, hi in BATTERY_TIERS:
        if name == tier_name:
            return float(rng.uniform(lo, hi))
    raise ValueError(f"unknown tier: {tier_name!r}")


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------
@dataclass
class Car:
    """A vehicle arriving at the station.

    Attributes describing the vehicle are set at arrival and treated as
    immutable.  ``service_start_time`` / ``service_end_time`` are filled in by
    the simulation engine once the car is (and finishes being) charged.
    """

    car_id: int
    arrival_time: float
    battery_capacity_kwh: float
    initial_soc_pct: float
    target_soc_pct: float
    max_accept_kw: float
    tier: str = "mid"
    #: Swap-out buffer (minutes), sampled at arrival so it stays synchronized
    #: across strategies under common random numbers.
    changeover: float = 0.0

    service_start_time: float | None = None
    service_end_time: float | None = None

    @property
    def kwh_needed(self) -> float:
        """Energy required to reach the target SOC (kWh)."""
        return max(
            0.0,
            self.battery_capacity_kwh * (self.target_soc_pct - self.initial_soc_pct) / 100.0,
        )

    @property
    def wait_time(self) -> float | None:
        """Time from arrival to service start (minutes), or ``None`` if not served."""
        if self.service_start_time is None:
            return None
        return self.service_start_time - self.arrival_time

    @property
    def charge_time(self) -> float | None:
        """Time spent actively charging (minutes), or ``None`` if not finished."""
        if self.service_start_time is None or self.service_end_time is None:
            return None
        return self.service_end_time - self.service_start_time

    @property
    def total_time_in_system(self) -> float | None:
        """Time from arrival to service completion (minutes)."""
        if self.service_end_time is None:
            return None
        return self.service_end_time - self.arrival_time

    @property
    def served(self) -> bool:
        return self.service_end_time is not None


@dataclass
class Bay:
    """A single charging bay with a fixed power rating.

    ``predicted_free_time`` is the simulation-clock time at which the bay is
    expected to be ready for the next car (current car finished plus changeover
    buffer).  For an idle bay this equals the current clock time.
    """

    bay_id: int
    rated_kw: float
    currently_occupied: bool = False
    current_car: Car | None = None
    predicted_free_time: float = 0.0

    def is_free(self, now: float) -> bool:
        """A bay is free only once its release has actually run.

        We require ``not currently_occupied`` *in addition to* the time check:
        the release (which clears the occupancy flag) happens in the changeover
        process after charging finishes, so a bay whose ``predicted_free_time``
        has been reached but whose release event has not yet fired must not be
        double-started.  ``predicted_free_time`` remains the *prediction* used
        for matching, not the occupancy gate.
        """
        return (not self.currently_occupied) and self.predicted_free_time <= now

    def occupy(self, car: Car, free_time: float) -> None:
        self.currently_occupied = True
        self.current_car = car
        self.predicted_free_time = free_time

    def release(self, now: float) -> None:
        self.currently_occupied = False
        self.current_car = None
        self.predicted_free_time = now

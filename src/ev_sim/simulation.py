"""SimPy discrete-event simulation engine.

The engine is strategy-agnostic.  Cars arrive according to a Poisson process,
join a station-level waiting pool, and are matched to bays by whichever
:class:`~ev_sim.allocators.Allocator` is injected.  The assignment is re-solved
from scratch on every state change (car arrival, or a bay becoming free after
charging + changeover).  A car matched to a *busy* bay holds a reservation for
it and starts when that bay frees; a reservation can migrate between bays as
the assignment is re-solved.

Changeover (physical swap) time is sampled per session from a uniform interval
and added to the bay's predicted free time, so the next car's wait naturally
includes the swap overhead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import simpy

from .allocators import Allocator, effective_power_kw
from .charge_curve import time_to_charge
from .metrics import bay_utilization_fraction
from .models import Bay, Car, sample_battery_capacity, sample_max_accept_kw


@dataclass
class SimulationConfig:
    """All tunable parameters of a single simulation run.

    Parameters
    ----------
    bay_power_kw:
        Rated power of each bay.  The number of bays is ``len(bay_power_kw)``.
    duration:
        Simulation horizon in minutes.
    warmup:
        Transient-discard period in minutes; cars arriving before this time are
        excluded from steady-state metrics.
    arrival_rate:
        Mean Poisson arrival rate in cars per hour.
    changeover_min / changeover_max:
        Uniform bounds (minutes) of the car-swap buffer added to a bay's free
        time after each session.
    initial_soc_a / initial_soc_b:
        Beta distribution shape parameters for initial SOC (fraction of 100%),
        scaled to ``[initial_soc_min, initial_soc_max]`` percent.  Defaults put
        the mode near ~35-40%, reflecting real low-to-moderate arrival SOC.
    partial_topup_prob / partial_target_min / partial_target_max:
        A fraction of cars only want a partial top-up, targeting a value in
        ``[partial_target_min, partial_target_max]`` instead of 100%.
    seed:
        Random seed for the fully reproducible run.
    """

    bay_power_kw: tuple[float, ...] = (60.0, 60.0, 22.0, 22.0)
    duration: float = 960.0
    warmup: float = 120.0
    arrival_rate: float = 4.0
    changeover_min: float = 2.0
    changeover_max: float = 5.0
    initial_soc_a: float = 2.2
    initial_soc_b: float = 3.0
    initial_soc_min: float = 2.0
    initial_soc_max: float = 90.0
    partial_topup_prob: float = 0.3
    partial_target_min: float = 80.0
    partial_target_max: float = 95.0
    seed: int = 0

    @property
    def num_bays(self) -> int:
        return len(self.bay_power_kw)

    def validate(self) -> None:
        if not self.bay_power_kw:
            raise ValueError("bay_power_kw must contain at least one bay")
        if any(kw <= 0.0 for kw in self.bay_power_kw):
            raise ValueError("bay_power_kw entries must be strictly positive")
        if self.arrival_rate <= 0:
            raise ValueError("arrival_rate must be positive")
        if self.duration <= 0 or self.warmup < 0 or self.warmup >= self.duration:
            raise ValueError("require 0 <= warmup < duration")


@dataclass
class SimulationResult:
    """Everything the engine measured during one run."""

    strategy: str
    seed: int
    config: SimulationConfig
    all_cars: list[Car] = field(default_factory=list)
    bay_busy_time: list[float] = field(default_factory=list)

    @property
    def completed_cars(self) -> list[Car]:
        """Cars whose charging finished (``service_end_time`` set)."""
        return [c for c in self.all_cars if c.served]

    @property
    def started_cars(self) -> list[Car]:
        """Cars that began charging (``service_start_time`` set) — their wait
        time is fully observed even if they are still charging at the horizon."""
        return [c for c in self.all_cars if c.service_start_time is not None]

    @property
    def never_started_cars(self) -> list[Car]:
        """Cars still queued at the horizon — wait time right-censored."""
        return [c for c in self.all_cars if c.service_start_time is None]

    @property
    def active_duration(self) -> float:
        return max(0.0, self.config.duration - self.config.warmup)

    @property
    def bay_utilization(self) -> list[float]:
        return bay_utilization_fraction(self.bay_busy_time, self.active_duration)

    def steady_state_cars(self) -> list[Car]:
        """Cars that began service after the warmup horizon.

        Wait time is determined at ``service_start_time``, so cars still
        charging at the horizon are included — excluding them (by requiring a
        ``service_end_time``) would censor exactly the longest-waiting cars
        under SRPT and bias the wait metrics.  Only *never-started* cars are
        right-censored; their count is reported separately.
        """
        return [c for c in self.started_cars if c.arrival_time >= self.config.warmup]

    def steady_state_wait_times(self) -> list[float]:
        return [c.wait_time for c in self.steady_state_cars()]


class Simulation:
    """Discrete-event simulation of a multi-bay EV charging station."""

    def __init__(self, config: SimulationConfig, allocator: Allocator):
        config.validate()
        self.config = config
        self.allocator = allocator

        # Common-random-numbers (CRN): derive three *independent* streams from a
        # single seed so that arrival times and car attributes are identical
        # across strategies for the same seed.  This is what makes the paired
        # strategy comparison a clean, variance-reduced design.
        children = np.random.SeedSequence(config.seed).spawn(3)
        self.arrival_rng = np.random.default_rng(children[0])
        self.car_rng = np.random.default_rng(children[1])
        self.changeover_rng = np.random.default_rng(children[2])

        self.env: simpy.Environment | None = None
        self.bays: list[Bay] = []
        self.waiting: list[Car] = []
        self.all_cars: list[Car] = []
        self.bay_busy_time: list[float] = []
        self._car_counter = 0

    # -- sampling -----------------------------------------------------------
    def sample_initial_soc(self) -> float:
        x = self.car_rng.beta(self.config.initial_soc_a, self.config.initial_soc_b)
        return float(np.clip(x * 100.0, self.config.initial_soc_min, self.config.initial_soc_max))

    def sample_target_soc(self, initial_soc: float) -> float:
        if self.car_rng.random() < self.config.partial_topup_prob:
            t = self.car_rng.uniform(self.config.partial_target_min, self.config.partial_target_max)
        else:
            t = 100.0
        return max(initial_soc, t)

    def make_car(self, arrival_time: float) -> Car:
        capacity, tier = sample_battery_capacity(self.car_rng)
        max_accept = sample_max_accept_kw(self.car_rng, tier)
        initial_soc = self.sample_initial_soc()
        target_soc = self.sample_target_soc(initial_soc)
        # Sample the swap-out buffer at arrival (in arrival order) so it becomes
        # a car attribute and stays synchronized across strategies under CRN.
        changeover = float(
            self.changeover_rng.uniform(self.config.changeover_min, self.config.changeover_max)
        )
        self._car_counter += 1
        return Car(self._car_counter, arrival_time, capacity, initial_soc,
                   target_soc, max_accept, tier, changeover=changeover)

    # -- event loop ---------------------------------------------------------
    def run(self) -> SimulationResult:
        self.env = simpy.Environment()
        self.bays = [Bay(i, kw, predicted_free_time=0.0)
                     for i, kw in enumerate(self.config.bay_power_kw)]
        self.bay_busy_time = [0.0] * len(self.bays)
        self.waiting = []
        self.all_cars = []
        self._car_counter = 0

        self.env.process(self._arrival_generator())
        self.env.run(until=self.config.duration)

        return SimulationResult(
            strategy=self.allocator.name,
            seed=self.config.seed,
            config=self.config,
            all_cars=self.all_cars,
            bay_busy_time=self.bay_busy_time,
        )

    def _arrival_generator(self) -> None:
        while True:
            interarrival = self.arrival_rng.exponential(60.0 / self.config.arrival_rate)
            yield self.env.timeout(interarrival)
            if self.env.now >= self.config.duration:
                return
            car = self.make_car(self.env.now)
            self.all_cars.append(car)
            self.waiting.append(car)
            self._schedule()

    def _schedule(self) -> None:
        """Re-solve the assignment and start any car whose matched bay is free."""
        if not self.waiting:
            return
        if not any(b.is_free(self.env.now) for b in self.bays):
            return

        assign = self.allocator.assign(list(self.waiting), self.bays, self.env.now)

        by_bay: dict[int, list[Car]] = {}
        for car in self.waiting:
            bid = assign.get(car.car_id)
            if bid is not None:
                by_bay.setdefault(bid, []).append(car)

        for bay in self.bays:
            if not bay.is_free(self.env.now):
                continue
            candidates = by_bay.get(bay.bay_id, [])
            if candidates:
                car = min(candidates, key=lambda c: (c.arrival_time, c.car_id))
                self._start_charge(car, bay)

    def _start_charge(self, car: Car, bay: Bay) -> None:
        self.waiting.remove(car)
        now = self.env.now
        car.service_start_time = now

        charge_min = time_to_charge(
            car.kwh_needed,
            car.initial_soc_pct,
            car.target_soc_pct,
            effective_power_kw(car, bay),
            car.battery_capacity_kwh,
        )
        if not math.isfinite(charge_min):
            # Defensive: an incompatible (zero-power) match must not freeze the
            # bay on an infinite timeout.  Config validation already rejects
            # non-positive bay power, so this is a fail-fast guard.
            raise ValueError(
                f"infinite charge time: car {car.car_id} cannot charge on bay {bay.bay_id}"
            )
        changeover = car.changeover
        bay.occupy(car, now + charge_min + changeover)

        # Accumulate charging busy-time over the steady-state window only.
        session_end = now + charge_min
        overlap = max(0.0, min(session_end, self.config.duration) - max(now, self.config.warmup))
        self.bay_busy_time[bay.bay_id] += overlap

        self.env.process(self._charge_process(car, bay, charge_min, changeover))

    def _charge_process(self, car: Car, bay: Bay, charge_min: float, changeover: float) -> None:
        yield self.env.timeout(charge_min)  # charging
        car.service_end_time = self.env.now
        yield self.env.timeout(changeover)  # physical swap-out
        bay.release(self.env.now)
        self._schedule()  # bay just became free -> re-solve

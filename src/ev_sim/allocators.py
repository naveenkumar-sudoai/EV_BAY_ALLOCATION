"""Bay-allocation strategies.

All strategies implement a single common interface::

    allocator.assign(cars: Sequence[Car], bays: Sequence[Bay], now: float) -> dict[int, int]

which returns a partial mapping ``car_id -> bay_id``.  The simulation engine is
strategy-agnostic: on every state change it calls ``assign`` over the currently
waiting cars and all bays, then starts any waiting car whose matched bay is
free.  Bays that are still busy act as *reservations* — the assigned car waits
for that specific bay — and the assignment is re-solved from scratch on every
event, so a car's reservation can migrate to a sooner-freeing bay.

Three policies are provided:

* :class:`FCFSAllocator` — arrival order to the soonest-free bay.
* :class:`SRPTAllocator` — minimise total predicted completion time via the
  Hungarian algorithm (``scipy.optimize.linear_sum_assignment``).  Incompatible
  pairings receive a heavy penalty rather than being hard-excluded, keeping the
  cost matrix a complete bipartite graph.
* :class:`SocAwareAgingAllocator` — the proposed method: completion-time cost
  blended with a SOC-aware priority score that *ages* with waiting time, plus a
  hard wait-time cap that force-schedules a car regardless of score.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .charge_curve import time_to_charge
from .models import Bay, Car

#: Penalty (minutes) applied to an *incompatible* (car, bay) pair so that the
#: Hungarian solver avoids it without a hard exclusion.  In the default model
#: every car can use every bay (effective power = min), so this never fires;
#: it is here for eligibility constraints (e.g. connector mismatch) and is
#: exercised directly by the unit tests.
INCOMPATIBLE_PENALTY = 1e9

#: Cost assigned to dummy rows/columns when padding a rectangular matrix to the
#: square shape required by ``linear_sum_assignment``.
_DUMMY_COST = 1e12


def effective_power_kw(car: Car, bay: Bay) -> float:
    """Power actually delivered, limited by the weaker of bay and vehicle."""
    return min(bay.rated_kw, car.max_accept_kw)


def charge_time_min(car: Car, bay: Bay) -> float:
    """Minutes to charge ``car`` to target on ``bay`` at the effective power."""
    return time_to_charge(
        car.kwh_needed,
        car.initial_soc_pct,
        car.target_soc_pct,
        effective_power_kw(car, bay),
        car.battery_capacity_kwh,
    )


def predicted_completion_time(car: Car, bay: Bay, now: float) -> float:
    """Predicted clock time (minutes) the car would finish if routed to ``bay``."""
    start = max(now, bay.predicted_free_time)
    return start + charge_time_min(car, bay)


def _is_incompatible(car: Car, bay: Bay) -> bool:
    """A pair is incompatible if the bay can deliver no positive power to it."""
    return effective_power_kw(car, bay) <= 0.0


class Allocator:
    """Base class / interface for all allocation strategies."""

    name: str = "base"

    def assign(self, cars: Sequence[Car], bays: Sequence[Bay], now: float) -> dict[int, int]:
        raise NotImplementedError


class FCFSAllocator(Allocator):
    """First-come-first-served: arrival order to the soonest-free bay."""

    name = "fcfs"

    def assign(self, cars: Sequence[Car], bays: Sequence[Bay], now: float) -> dict[int, int]:
        ordered_cars = sorted(cars, key=lambda c: (c.arrival_time, c.car_id))
        ordered_bays = sorted(bays, key=lambda b: (b.predicted_free_time, b.bay_id))
        return {c.car_id: b.bay_id for c, b in zip(ordered_cars, ordered_bays)}


class SRPTAllocator(Allocator):
    """Shortest-remaining-processing-time via min-total-completion matching.

    The cost of assigning car ``i`` to bay ``j`` is the predicted completion
    time; the Hungarian algorithm finds the assignment that minimises the *sum*
    of completion times over all waiting cars (with incompatible pairs heavily
    penalised).  This is the optimal non-preemptive min-sum-completion schedule
    and captures the SRPT intuition: short jobs go first, long jobs wait.
    """

    name = "srpt"

    def assign(self, cars: Sequence[Car], bays: Sequence[Bay], now: float) -> dict[int, int]:
        if not cars or not bays:
            return {}
        cars = list(cars)
        bays = list(bays)
        n, m = len(cars), len(bays)

        cost = np.zeros((n, m))
        for i, car in enumerate(cars):
            for j, bay in enumerate(bays):
                if _is_incompatible(car, bay):
                    # Heavy finite penalty, not a hard exclusion (an infinite
                    # completion time would make the matrix infeasible for the
                    # Hungarian solver).
                    cost[i, j] = INCOMPATIBLE_PENALTY
                else:
                    cost[i, j] = predicted_completion_time(car, bay, now)

        cost = _pad_to_square(cost, n, m)
        rows, cols = linear_sum_assignment(cost)

        assign: dict[int, int] = {}
        for i, j in zip(rows, cols):
            if i < n and j < m:  # skip dummy rows/cols
                assign[cars[i].car_id] = bays[j].bay_id
        return assign


class SocAwareAgingAllocator(Allocator):
    """Proposed SOC-aware + priority-aging policy.

    A car's priority score is::

        score(car) = charge_time_needed / (1 + k * wait_time_elapsed)

    where ``charge_time_needed`` is the car's *minimum* charge time (at its own
    maximum acceptance rate), a proxy for how much work it represents.  Lower
    score → served sooner.  As a car waits, its score shrinks, so it gradually
    outranks freshly arrived short jobs — this is the anti-starvation ageing.

    The Hungarian cost applies this score to the *work* component of the
    completion time, while leaving the bay-availability component untouched::

        cost[car, bay] = max(now, bay.free_time) + charge_time(car, bay) / (1 + k * wait)

    Equivalently, ``cost = bay_availability + priority_score * bay_slowdown``
    where ``bay_slowdown = charge_time(car,bay) / charge_time_needed(car) >= 1``
    captures how much slower ``bay`` is than the car's ideal.  This keeps the
    per-bay information needed for capacity matching, but *ages* the work so a
    long-waiting car's effective cost shrinks.

    At ``k = 0`` the cost reduces exactly to the predicted completion time —
    i.e. pure SRPT — so the ageing constant interpolates cleanly from
    efficiency-dominated (SRPT-like, starvation-prone) toward fairness-dominated
    (FCFS-like) as ``k`` grows.

    A hard wait-time cap overrides the scoring entirely: any car that has
    waited at least ``wait_cap`` minutes is force-assigned to its best
    (minimum-completion) bay regardless of score.
    """

    name = "soc_aware_aging"

    def __init__(self, k: float = 0.05, wait_cap: float = 45.0):
        self.k = float(k)
        self.wait_cap = float(wait_cap)

    def charge_time_needed(self, car: Car) -> float:
        """Intrinsic minimum charge time at the car's own max acceptance rate."""
        return time_to_charge(
            car.kwh_needed,
            car.initial_soc_pct,
            car.target_soc_pct,
            car.max_accept_kw,
            car.battery_capacity_kwh,
        )

    def priority_score(self, car: Car, now: float) -> float:
        wait = max(0.0, now - car.arrival_time)
        return self.charge_time_needed(car) / (1.0 + self.k * wait)

    def assign(self, cars: Sequence[Car], bays: Sequence[Bay], now: float) -> dict[int, int]:
        if not cars or not bays:
            return {}
        cars = list(cars)
        bays = list(bays)

        # 1. Hard wait-time cap: cars that have waited at least ``wait_cap``
        #    minutes are force-assigned regardless of score.  They are served in
        #    arrival order (longest-waiting first) to the soonest-free bay, which
        #    is the genuinely fair anti-starvation rule (FCFS among the capped).
        capped = sorted(
            (c for c in cars if now - c.arrival_time >= self.wait_cap),
            key=lambda c: (c.arrival_time, c.car_id),
        )
        assign: dict[int, int] = {}
        used_bays: set[int] = set()
        bay_pool = sorted(bays, key=lambda b: (b.predicted_free_time, b.bay_id))
        for car in capped:
            for bay in bay_pool:
                if bay.bay_id in used_bays or _is_incompatible(car, bay):
                    continue
                assign[car.car_id] = bay.bay_id
                used_bays.add(bay.bay_id)
                break

        # 2. Remaining (non-capped) cars matched by the aged completion-time
        #    cost via the Hungarian algorithm.
        remaining_cars = [c for c in cars if c.car_id not in assign]
        remaining_bays = [b for b in bays if b.bay_id not in used_bays]
        if remaining_cars and remaining_bays:
            assign.update(self._hungarian_match(remaining_cars, remaining_bays, now))

        return assign

    def _hungarian_match(
        self, cars: list[Car], bays: list[Bay], now: float
    ) -> dict[int, int]:
        n, m = len(cars), len(bays)
        cost = np.zeros((n, m))
        for i, car in enumerate(cars):
            wait = max(0.0, now - car.arrival_time)
            for j, bay in enumerate(bays):
                if _is_incompatible(car, bay):
                    cost[i, j] = INCOMPATIBLE_PENALTY
                else:
                    start = max(now, bay.predicted_free_time)
                    # Age only the work term of the completion time.
                    cost[i, j] = start + charge_time_min(car, bay) / (1.0 + self.k * wait)

        cost = _pad_to_square(cost, n, m)
        rows, cols = linear_sum_assignment(cost)

        out: dict[int, int] = {}
        for i, j in zip(rows, cols):
            if i < n and j < m:
                out[cars[i].car_id] = bays[j].bay_id
        return out


def _pad_to_square(cost: np.ndarray, n: int, m: int) -> np.ndarray:
    """Pad a rectangular ``(n, m)`` cost matrix to a square one.

    * More cars than bays (``n > m``): add ``n - m`` dummy bays at cost
      ``_DUMMY_COST`` so only ``m`` cars are matched to real bays; the rest are
      left unassigned (still waiting).
    * More bays than cars (``m > n``): add ``m - n`` dummy cars at cost 0 so
      every real car is matched and surplus bays simply stay idle.
    """
    if n == m:
        return cost
    if n > m:
        pad = np.full((n, n - m), _DUMMY_COST)
        return np.hstack([cost, pad])
    pad = np.full((m - n, m), 0.0)
    return np.vstack([cost, pad])

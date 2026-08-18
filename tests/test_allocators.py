"""Unit tests for the three allocation strategies, in isolation."""
from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from ev_sim.allocators import (
    INCOMPATIBLE_PENALTY,
    FCFSAllocator,
    SRPTAllocator,
    SocAwareAgingAllocator,
    effective_power_kw,
    predicted_completion_time,
)
from ev_sim.models import Bay, Car


# --- helpers ----------------------------------------------------------------
def make_car(car_id, arrival, cap=40.0, soc=20.0, target=100.0, accept=60.0):
    return Car(car_id, arrival, cap, soc, target, accept)


def make_bay(bay_id, rated=60.0, free=0.0):
    return Bay(bay_id, rated, predicted_free_time=free)


def assert_valid_assignment(assign, cars, bays):
    """A returned assignment must be injective in both cars and bays."""
    car_ids = [c.car_id for c in cars]
    bay_ids = [b.bay_id for b in bays]
    assigned_cars = list(assign.keys())
    assigned_bays = list(assign.values())
    assert all(cid in car_ids for cid in assigned_cars)
    assert all(bid in bay_ids for bid in assigned_bays)
    assert len(set(assigned_cars)) == len(assigned_cars)  # no car twice
    assert len(set(assigned_bays)) == len(assigned_bays)  # no bay twice


# --- FCFS -------------------------------------------------------------------
def test_fcfs_preserves_arrival_order_to_soonest_bay():
    bays = [make_bay(0, free=10.0), make_bay(1, free=0.0)]
    cars = [make_car(1, arrival=5.0), make_car(2, arrival=0.0)]
    assign = FCFSAllocator().assign(cars, bays, now=0.0)
    # earliest arrival (car 2) -> soonest free bay (bay 1)
    assert assign[2] == 1
    assert assign[1] == 0


def test_fcfs_more_cars_than_bays_leaves_excess_unassigned():
    bays = [make_bay(0), make_bay(1)]
    cars = [make_car(i, arrival=float(i)) for i in range(5)]
    assign = FCFSAllocator().assign(cars, bays, now=0.0)
    assert len(assign) == 2
    # the two earliest arrivals are the ones assigned
    assert set(assign) == {0, 1}


def test_fcfs_fewer_cars_than_bays_assigns_every_car():
    bays = [make_bay(i) for i in range(4)]
    cars = [make_car(0, arrival=0.0), make_car(1, arrival=1.0)]
    assign = FCFSAllocator().assign(cars, bays, now=0.0)
    assert len(assign) == 2


# --- SRPT -------------------------------------------------------------------
def test_effective_power_is_min_of_bay_and_car():
    car = make_car(0, 0.0, accept=30.0)
    assert effective_power_kw(car, make_bay(0, rated=60.0)) == 30.0
    assert effective_power_kw(car, make_bay(0, rated=22.0)) == 22.0


def test_srpt_shortest_job_gets_soonest_bay():
    # One bay free now, one free later; short job must go to the free-now bay.
    bays = [make_bay(0, free=0.0), make_bay(1, free=60.0)]
    short = Car(10, 0.0, 40.0, 80.0, 100.0, 60.0)     # only 20% -> short
    long = Car(11, 0.0, 40.0, 0.0, 100.0, 60.0)       # 0 -> 100% -> long
    assign = SRPTAllocator().assign([short, long], bays, now=0.0)
    assert assign[short.car_id] == 0
    assert assign[long.car_id] == 1


def test_srpt_matches_bruteforce_min_sum():
    rng = np.random.default_rng(7)
    bays = [make_bay(i, rated=float(rng.choice([22, 60])), free=float(rng.uniform(0, 40)))
            for i in range(3)]
    cars = [Car(i, 0.0, float(rng.uniform(20, 75)), float(rng.uniform(10, 50)),
                100.0, float(rng.uniform(7, 150))) for i in range(3)]
    assign = SRPTAllocator().assign(cars, bays, now=5.0)

    got = sum(predicted_completion_time(c, b, 5.0)
              for c in cars for b in bays if assign.get(c.car_id) == b.bay_id)

    best = min(
        sum(predicted_completion_time(cars[i], bays[perm[i]], 5.0) for i in range(3))
        for perm in permutations(range(3))
    )
    assert got == pytest.approx(best)


def test_srpt_penalises_incompatible_pairing():
    # Bay 0 cannot deliver power (rated 0) -> car routed to compatible bays only.
    bays = [make_bay(0, rated=0.0, free=0.0), make_bay(1, rated=60.0, free=0.0)]
    car = make_car(5, 0.0, accept=60.0)
    assign = SRPTAllocator().assign([car], bays, now=0.0)
    assert assign[car.car_id] == 1  # incompatible bay avoided via penalty


def test_srpt_incompatible_only_option_still_assigns():
    # Only a zero-rated bay exists: no hard exclusion means we still return a
    # mapping (penalty, not a crash or an empty result).
    bays = [make_bay(0, rated=0.0)]
    car = make_car(5, 0.0)
    assign = SRPTAllocator().assign([car], bays, now=0.0)
    assert assign == {car.car_id: 0}


def test_srpt_more_cars_than_bays_only_assigns_n_bays():
    bays = [make_bay(i) for i in range(3)]
    cars = [make_car(i, arrival=float(i)) for i in range(8)]
    assign = SRPTAllocator().assign(cars, bays, now=0.0)
    assert len(assign) == 3
    assert_valid_assignment(assign, cars, bays)


# --- SOC-aware + aging ------------------------------------------------------
def test_priority_score_formula():
    alloc = SocAwareAgingAllocator(k=0.05)
    car = make_car(0, 0.0)
    # charge_time_needed uses the car's own max accept rate (60 kW here)
    ct = alloc.charge_time_needed(car)
    assert alloc.priority_score(car, now=car.arrival_time) == pytest.approx(ct)
    # after 20 min wait with k=0.05 -> denom = 1 + 0.05*20 = 2
    assert alloc.priority_score(car, now=car.arrival_time + 20.0) == pytest.approx(ct / 2.0)


def test_aging_reduces_score_with_wait():
    alloc = SocAwareAgingAllocator(k=0.05)
    a = make_car(0, 0.0)
    b = make_car(1, 0.0)
    score_fresh = alloc.priority_score(a, now=0.0)
    score_waited = alloc.priority_score(b, now=100.0)
    assert score_waited < score_fresh


def test_k_zero_gives_pure_charge_time_score():
    alloc = SocAwareAgingAllocator(k=0.0)
    car = make_car(0, 0.0)
    assert alloc.priority_score(car, now=1000.0) == pytest.approx(alloc.charge_time_needed(car))


def test_wait_cap_forces_big_car_ahead_of_small_car():
    # Single bay, a huge-need car that has waited past the cap vs a fresh small car.
    bay = make_bay(0, free=0.0)
    big = Car(1, arrival_time=-50.0, battery_capacity_kwh=75.0, initial_soc_pct=0.0,
              target_soc_pct=100.0, max_accept_kw=60.0)   # waited 50 min, big charge
    small = make_car(2, 0.0, cap=40.0, soc=90.0)           # fresh, small top-up

    # Without a cap (huge threshold) the small car wins, like SRPT.
    no_cap = SocAwareAgingAllocator(k=0.05, wait_cap=1e9)
    assert no_cap.assign([big, small], [bay], now=0.0)[small.car_id] == 0

    # With the cap (45 min) the big car is force-assigned regardless of score.
    with_cap = SocAwareAgingAllocator(k=0.05, wait_cap=45.0)
    assign = with_cap.assign([big, small], [bay], now=0.0)
    assert assign[big.car_id] == 0


def test_cap_override_drops_score_term():
    alloc = SocAwareAgingAllocator(k=0.05, wait_cap=45.0)
    bay = make_bay(0, free=0.0)
    big = Car(1, arrival_time=-100.0, battery_capacity_kwh=75.0, initial_soc_pct=0.0,
              target_soc_pct=100.0, max_accept_kw=60.0)
    small = make_car(2, 0.0, cap=40.0, soc=90.0)
    # With cap override active, big car is assigned to the only bay despite being
    # the "worse" SRPT choice.
    assert alloc.assign([small, big], [bay], now=0.0)[big.car_id] == 0


def test_incompatible_penalty_dominates_any_real_cost():
    # Guardrail: the penalty must dwarf any realistic completion time so the
    # solver always avoids an incompatible pairing.
    assert INCOMPATIBLE_PENALTY > 1e6


def test_k_zero_matches_srpt():
    """At k=0 with no cap, the aged cost reduces exactly to completion time,
    so the proposed method must produce the same assignment as pure SRPT."""
    rng = np.random.default_rng(9)
    bays = [make_bay(i, rated=float(rng.choice([22, 60])), free=float(rng.uniform(0, 50)))
            for i in range(3)]
    cars = [Car(i, float(rng.uniform(0, 40)), float(rng.uniform(20, 75)),
                float(rng.uniform(5, 50)), 100.0, float(rng.uniform(7, 150)))
            for i in range(5)]
    srpt = SRPTAllocator().assign(cars, bays, now=30.0)
    aged = SocAwareAgingAllocator(k=0.0, wait_cap=1e9).assign(cars, bays, now=30.0)
    assert aged == srpt


@pytest.mark.parametrize("allocator", [
    FCFSAllocator(),
    SRPTAllocator(),
    SocAwareAgingAllocator(),
])
def test_all_allocators_return_valid_injective_assignments(allocator):
    rng = np.random.default_rng(3)
    bays = [make_bay(i, rated=float(rng.choice([22, 60])),
                     free=float(rng.uniform(0, 30))) for i in range(4)]
    cars = [Car(i, float(rng.uniform(0, 100)), float(rng.uniform(20, 75)),
                float(rng.uniform(5, 50)), 100.0, float(rng.uniform(7, 150)))
            for i in range(6)]
    assign = allocator.assign(cars, bays, now=50.0)
    assert_valid_assignment(assign, cars, bays)
    assert len(assign) == min(len(cars), len(bays))


@pytest.mark.parametrize("allocator", [
    FCFSAllocator(),
    SRPTAllocator(),
    SocAwareAgingAllocator(),
])
def test_empty_inputs_return_empty_assignment(allocator):
    assert allocator.assign([], [], 0.0) == {}
    assert allocator.assign([make_car(0, 0.0)], [], 0.0) == {}
    assert allocator.assign([], [make_bay(0)], 0.0) == {}

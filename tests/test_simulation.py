"""End-to-end and behavioural tests for the simulation engine."""
from __future__ import annotations

import numpy as np
import pytest

from ev_sim.allocators import FCFSAllocator, SRPTAllocator, SocAwareAgingAllocator
from ev_sim.metrics import summarize
from ev_sim.simulation import Simulation, SimulationConfig


def config(**kwargs) -> SimulationConfig:
    base = dict(
        bay_power_kw=(60.0, 60.0, 22.0, 22.0),
        duration=480.0,
        warmup=60.0,
        arrival_rate=4.0,
        seed=0,
    )
    base.update(kwargs)
    return SimulationConfig(**base)


@pytest.mark.parametrize("allocator", [
    FCFSAllocator(),
    SRPTAllocator(),
    SocAwareAgingAllocator(),
])
def test_run_completes_and_sane_metrics(allocator):
    result = Simulation(config(), allocator).run()
    wt = result.steady_state_wait_times()
    assert len(result.completed_cars) > 0
    assert len(wt) > 0
    # every served car: non-negative wait, positive charge time, correct totals
    for car in result.completed_cars:
        assert car.wait_time >= -1e-9
        assert car.charge_time > 0
        assert car.total_time_in_system == pytest.approx(car.wait_time + car.charge_time)
        assert car.service_end_time >= car.service_start_time
    # utilization bounded in [0, 1]
    assert all(0.0 <= u <= 1.0 + 1e-9 for u in result.bay_utilization)


def test_reproducibility_same_seed():
    a = Simulation(config(seed=123), SRPTAllocator()).run()
    b = Simulation(config(seed=123), SRPTAllocator()).run()
    assert a.steady_state_wait_times() == b.steady_state_wait_times()


def test_different_seeds_differ():
    a = Simulation(config(seed=1), FCFSAllocator()).run()
    b = Simulation(config(seed=2), FCFSAllocator()).run()
    assert a.steady_state_wait_times() != b.steady_state_wait_times()


def test_strategies_share_identical_car_stream():
    """Same seed -> identical arrivals and car attributes across strategies."""
    fcfs = Simulation(config(seed=42), FCFSAllocator()).run()
    srpt = Simulation(config(seed=42), SRPTAllocator()).run()
    fcfs_cars = [(c.arrival_time, c.battery_capacity_kwh, c.initial_soc_pct, c.target_soc_pct)
                 for c in fcfs.all_cars]
    srpt_cars = [(c.arrival_time, c.battery_capacity_kwh, c.initial_soc_pct, c.target_soc_pct)
                 for c in srpt.all_cars]
    assert fcfs_cars == srpt_cars


def test_arrival_count_is_plausible_for_poisson():
    # 480 min at 4 cars/h -> ~32 expected.  A wide bound catches gross bugs only.
    result = Simulation(config(seed=0), FCFSAllocator()).run()
    assert 10 < len(result.all_cars) < 60


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        Simulation(config(arrival_rate=0.0), FCFSAllocator()).run()
    with pytest.raises(ValueError):
        Simulation(config(bay_power_kw=()), FCFSAllocator()).run()
    with pytest.raises(ValueError):
        Simulation(config(bay_power_kw=(60.0, 0.0, 22.0, 22.0)), FCFSAllocator()).run()


def test_warmup_excludes_early_arrivals():
    result = Simulation(config(duration=300.0, warmup=100.0, seed=5), FCFSAllocator()).run()
    assert all(c.arrival_time >= 100.0 for c in result.steady_state_cars())


# --- behavioural (multi-seed, so robust to stochastic noise) ----------------
def _run_many(allocator, cfg, n_seeds, metric):
    values = []
    for seed in range(n_seeds):
        r = Simulation(config(**{**cfg, "seed": seed}), allocator).run()
        s = summarize(r.steady_state_wait_times(), r.bay_utilization, 30.0,
                      n_censored=len(r.never_started_cars))
        values.append(s[metric])
    return float(np.mean(values))


def test_srpt_is_efficient_but_unfair_versus_fcfs():
    """SRPT minimises mean wait (efficiency) yet is less fair than FCFS."""
    cfg = dict(arrival_rate=3.0, duration=1200.0, warmup=150.0)
    srpt_mean = _run_many(SRPTAllocator(), cfg, 10, "mean_wait_min")
    fcfs_mean = _run_many(FCFSAllocator(), cfg, 10, "mean_wait_min")
    srpt_jain = _run_many(SRPTAllocator(), cfg, 10, "jain_fairness")
    fcfs_jain = _run_many(FCFSAllocator(), cfg, 10, "jain_fairness")
    assert srpt_mean < fcfs_mean
    assert srpt_jain < fcfs_jain


def test_aging_bounds_max_wait_relative_to_srpt_under_load():
    """At high load the ageing cap bounds the worst-case wait that pure SRPT
    lets blow up through starvation."""
    cfg = dict(arrival_rate=4.5, duration=1500.0, warmup=200.0)
    srpt_max = _run_many(SRPTAllocator(), cfg, 8, "max_wait_min")
    aged_max = _run_many(SocAwareAgingAllocator(k=0.05, wait_cap=45.0), cfg, 8, "max_wait_min")
    assert aged_max < srpt_max

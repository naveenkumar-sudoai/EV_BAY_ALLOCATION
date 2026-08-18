#!/usr/bin/env python3
"""Run a single simulation and *see* it as a bay-occupancy timeline.

Prints a readable per-car trace and saves a Gantt chart showing each car's
arrival (dot) and charging interval (bar) on the bay it was assigned to.

Usage::

    .venv/bin/python visualize_run.py --strategy aging --rate 2.0 --seed 0
    .venv/bin/python visualize_run.py --strategy srpt --duration 720
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ev_sim.allocators import FCFSAllocator, SRPTAllocator, SocAwareAgingAllocator
from ev_sim.metrics import summarize
from ev_sim.simulation import Simulation, SimulationConfig

ALLOCATORS = {
    "fcfs": FCFSAllocator,
    "srpt": SRPTAllocator,
    "aging": SocAwareAgingAllocator,
}

TIER_COLORS = {"small": "#8c8c8c", "mid": "#4c78a8", "premium": "#e45756"}


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize one simulation run")
    p.add_argument("--strategy", choices=list(ALLOCATORS), default="aging")
    p.add_argument("--rate", type=float, default=2.0, help="cars per hour")
    p.add_argument("--duration", type=float, default=720.0, help="minutes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/simulation_timeline.png")
    args = p.parse_args()

    cfg = SimulationConfig(arrival_rate=args.rate, duration=args.duration,
                           warmup=0.0, seed=args.seed)
    result = Simulation(cfg, ALLOCATORS[args.strategy]()).run()
    s = summarize(result.steady_state_wait_times(), result.bay_utilization, 30.0,
                  n_censored=len(result.never_started_cars))

    # --- text trace --------------------------------------------------------
    print(f"\n{'strategy':>10} {args.strategy}    rate {args.rate}/h    "
          f"duration {args.duration:.0f} min    seed {args.seed}")
    print(f"{'':>10} {len(result.all_cars)} cars arrived, "
          f"{len(result.completed_cars)} served, {s['n_censored']} still queued at end")
    print(f"{'':>10} mean wait {s['mean_wait_min']:.1f} min  "
          f"max wait {s['max_wait_min']:.1f} min  Jain {s['jain_fairness']:.2f}")
    print(f"{'':>10} bay utilization {[f'{u:.2f}' for u in result.bay_utilization]}")
    print()
    print(f"{'car':>4} {'tier':>8} {'arrive':>7} {'wait':>6} {'bay':>4} "
          f"{'charge':>7} {'kWh':>5}")
    for car in result.started_cars[:20]:
        ct = f"{car.charge_time:.1f}" if car.charge_time is not None else "…"
        print(f"{car.car_id:>4} {car.tier:>8} {car.arrival_time:>7.1f} "
              f"{car.wait_time:>6.1f} {car.served_bay:>4} "
              f"{ct:>7} {car.kwh_needed:>5.1f}")
    if len(result.started_cars) > 20:
        print(f"  ... and {len(result.started_cars) - 20} more")

    # --- Gantt chart -------------------------------------------------------
    n_bays = cfg.num_bays
    fig, ax = plt.subplots(figsize=(12, 1.6 + 0.7 * n_bays))
    for car in result.started_cars:
        if car.served_bay is None:
            continue
        y = car.served_bay
        end = car.service_end_time if car.service_end_time is not None else args.duration
        start = car.service_start_time
        ax.broken_barh([(start, end - start)], (y - 0.38, 0.76),
                       facecolors=TIER_COLORS.get(car.tier, "#666"),
                       edgecolor="black", linewidth=0.5, alpha=0.9)
        ax.plot(car.arrival_time, y, marker="o", ms=3, color="black", zorder=5)

    ax.set_yticks(range(n_bays))
    ax.set_yticklabels([f"Bay {i} ({cfg.bay_power_kw[i]:.0f} kW)" for i in range(n_bays)])
    ax.set_xlabel("Time (minutes)")
    ax.set_title(f"Bay occupancy timeline — {args.strategy.upper()} "
                 f"(rate {args.rate}/h, seed {args.seed})")
    ax.set_xlim(0, args.duration)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved timeline chart -> {args.out}")


if __name__ == "__main__":
    main()

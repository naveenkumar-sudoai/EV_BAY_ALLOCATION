#!/usr/bin/env python3
"""Render a real-time animation of one simulation run.

Produces an MP4 (or GIF) that plays the discrete-event timeline forward:
cars arrive on the *queue* lane, then move into a *bay* lane where their
charging bar fills, then leave.  A red "now" line sweeps across, and the header
shows live counters.

Usage::

    .venv/bin/python animate_run.py --strategy srpt --rate 3.0 --duration 720
    .venv/bin/python animate_run.py --strategy aging --out results/sim.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from ev_sim.allocators import FCFSAllocator, SRPTAllocator, SocAwareAgingAllocator
from ev_sim.simulation import Simulation, SimulationConfig

ALLOCATORS = {
    "fcfs": FCFSAllocator,
    "srpt": SRPTAllocator,
    "aging": SocAwareAgingAllocator,
}

TIER_COLORS = {"small": "#8c8c8c", "mid": "#4c78a8", "premium": "#e45756"}


def main() -> None:
    p = argparse.ArgumentParser(description="Animate one simulation run")
    p.add_argument("--strategy", choices=list(ALLOCATORS), default="aging")
    p.add_argument("--rate", type=float, default=2.0, help="cars per hour")
    p.add_argument("--duration", type=float, default=480.0, help="minutes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--step", type=float, default=None, help="minutes per frame")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--out", default="results/simulation_animation.mp4")
    args = p.parse_args()

    cfg = SimulationConfig(arrival_rate=args.rate, duration=args.duration,
                           warmup=0.0, seed=args.seed)
    result = Simulation(cfg, ALLOCATORS[args.strategy]()).run()
    cars = result.all_cars
    n_bays = cfg.num_bays

    # --- animation state ---------------------------------------------------
    step = args.step or max(args.duration / 400.0, 0.5)
    times = list(__import__("numpy").arange(0, args.duration + step, step))

    fig, ax = plt.subplots(figsize=(12, 2.2 + 0.55 * n_bays))
    fig.subplots_adjust(left=0.16, right=0.97, top=0.80, bottom=0.10)

    queue_y = n_bays + 0.5   # waiting lane sits above the bays

    def draw(t):
        ax.clear()
        ax.set_xlim(0, args.duration)
        ax.set_ylim(0, n_bays + 1)

        # Bay lanes (light background + label)
        for i in range(n_bays):
            ax.axhspan(i, i + 1, color="#f0f2f5", zorder=0)
            ax.text(args.duration * 0.005, i + 0.5,
                    f"Bay {i}  ({cfg.bay_power_kw[i]:.0f} kW)",
                    va="center", ha="left", fontsize=9, color="#555")
        ax.axhspan(n_bays, n_bays + 1, color="#fdf2f2", zorder=0)
        ax.text(args.duration * 0.005, queue_y, "WAITING QUEUE",
                va="center", ha="left", fontsize=9, color="#a33", style="italic")

        # Cars
        waiting = served = charging = 0
        for car in cars:
            a = car.arrival_time
            s = car.service_start_time
            e = car.service_end_time
            color = TIER_COLORS.get(car.tier, "#666")

            if a > t:
                continue  # not arrived yet

            if s is None:
                # arrived but never started by end -> sits in the queue
                if a <= t:
                    ax.plot(a, queue_y, marker="o", ms=6, color=color,
                            mec="black", mew=0.5, zorder=5)
                    waiting += 1
                continue

            if s <= t and (e is None or e > t):
                # currently charging -> bar fills up to now
                charging += 1
                end = min(e if e is not None else args.duration, t)
                ax.add_patch(Rectangle((s, car.served_bay + 0.15), end - s, 0.7,
                                       facecolor=color, edgecolor="black",
                                       linewidth=0.6, alpha=0.95, zorder=3))
                ax.text(s + max(end - s, 1) * 0.5, car.served_bay + 0.5,
                        f"#{car.car_id}", ha="center", va="center",
                        fontsize=7, color="white", zorder=4)
            elif e is not None and e <= t:
                # finished -> full faded bar
                served += 1
                ax.add_patch(Rectangle((s, car.served_bay + 0.2), e - s, 0.6,
                                       facecolor=color, edgecolor="none",
                                       alpha=0.35, zorder=2))
            elif a <= t < s:
                # waiting (not yet started)
                ax.plot(a, queue_y, marker="o", ms=6, color=color,
                        mec="black", mew=0.5, zorder=5)
                waiting += 1

        # Now-line
        ax.axvline(t, color="#d62728", lw=2, zorder=6, alpha=0.9)

        # Header stats
        ax.set_title(
            f"{args.strategy.upper()}  ·  rate {args.rate}/h  ·  t = {t:5.1f} min   "
            f"|  arrived {sum(c.arrival_time <= t for c in cars):3d}   "
            f"waiting {waiting:2d}   charging {charging:2d}   served {served:3d}",
            fontsize=11, loc="left", pad=12)
        ax.set_xlabel("Time (minutes)")
        ax.set_yticks([])

    anim = FuncAnimation(fig, draw, frames=times, interval=1000 / args.fps,
                         repeat=False)

    out = args.out
    if out.endswith(".gif"):
        writer = PillowWriter(fps=args.fps)
    else:
        writer = FFMpegWriter(fps=args.fps, bitrate=1800)
    anim.save(out, writer=writer)
    plt.close(fig)
    print(f"Saved animation ({len(times)} frames) -> {out}")
    print("Open it with:  xdg-open " + out)


if __name__ == "__main__":
    main()

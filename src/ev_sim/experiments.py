"""Experiment runner.

Runs the four experiments that back the paper and writes CSV summary tables and
publication-style figures to ``results/``:

* **A — Baseline**       3 strategies x 30 seeds at a fixed moderate load.
* **B — Load sensitivity** 5 arrival rates x 3 strategies x 15 seeds.
* **C — k sensitivity**  sweep the ageing constant (aging strategy only).
* **D — Bay heterogeneity** homogeneous (4x40 kW) vs heterogeneous (60/60/22/22).

Every experiment logs its seed and configuration so the whole suite is
reproducible from scratch.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .allocators import FCFSAllocator, SRPTAllocator, SocAwareAgingAllocator
from .metrics import summarize
from .simulation import Simulation, SimulationConfig

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"

# ---------------------------------------------------------------------------
# Strategy registry & presentation
# ---------------------------------------------------------------------------
STRATEGY_ORDER = ["FCFS", "SRPT", "SOC-Aware+Aging"]
STRATEGY_COLORS = {
    "FCFS": "#9a9a9a",
    "SRPT": "#e45756",
    "SOC-Aware+Aging": "#4c78a8",
}
STRATEGY_LABELS = {
    "FCFS": "FCFS",
    "SRPT": "Pure SRPT",
    "SOC-Aware+Aging": "SOC-aware + aging (proposed)",
}


def make_allocator(strategy: str):
    if strategy == "FCFS":
        return FCFSAllocator()
    if strategy == "SRPT":
        return SRPTAllocator()
    if strategy == "SOC-Aware+Aging":
        return SocAwareAgingAllocator(k=0.05, wait_cap=45.0)
    raise ValueError(f"unknown strategy: {strategy}")


METRIC_COLUMNS = [
    "mean_wait_min",
    "median_wait_min",
    "max_wait_min",
    "jain_fairness",
    "starvation_count",
    "mean_bay_utilization",
    "n_served",
    "n_censored",
]


def _set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.autolayout": True,
        }
    )


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def run_single(allocator, cfg: SimulationConfig) -> tuple[dict, list[float]]:
    """Run one replicate and return (metrics dict, steady-state wait times)."""
    result = Simulation(cfg, allocator).run()
    metrics = summarize(
        result.steady_state_wait_times(),
        result.bay_utilization,
        starvation_threshold=30.0,
        n_censored=len(result.never_started_cars),
    )
    return metrics, result.steady_state_wait_times()


def run_replicates(allocator_factory, cfg: SimulationConfig, seeds: list[int]):
    """Run ``len(seeds)`` replicates and return (metric records, pooled waits)."""
    records: list[dict] = []
    pooled_waits: list[float] = []
    for seed in seeds:
        cfg_seed = replace(cfg, seed=seed)
        metrics, waits = run_single(allocator_factory(), cfg_seed)
        metrics = dict(metrics)
        metrics["seed"] = seed
        records.append(metrics)
        pooled_waits.extend(waits)
    return records, pooled_waits


def mean_ci(values, confidence: float = 0.95) -> tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)`` via a Student-t interval."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = vals.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(vals.mean())
    if n < 2:
        return mean, mean, mean
    sem = vals.std(ddof=1) / math.sqrt(n)
    lo, hi = stats.t.interval(confidence, n - 1, loc=mean, scale=sem)
    return mean, float(lo), float(hi)


def _format_ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.1f} ± {mean - lo:.1f}"


def measure_capacity(cfg: SimulationConfig, rate: float = 0.8, seeds: int = 5) -> tuple[float, float]:
    """Estimate station capacity by simulating a near-empty system.

    Returns ``(mean_service_time_min, capacity_cars_per_hour)``.  At a very
    light arrival rate there is no queueing, so every served car's charge time
    equals its service time and, by Little's law,
    ``capacity = num_bays * 60 / mean_service_time``.
    """
    light = replace(cfg, arrival_rate=rate)
    service_times: list[float] = []
    for seed in range(seeds):
        r = Simulation(replace(light, seed=seed), FCFSAllocator()).run()
        service_times.extend(
            c.charge_time for c in r.completed_cars if c.arrival_time >= light.warmup
        )
    mean_service = float(np.mean(service_times))
    capacity = light.num_bays * 60.0 / mean_service
    return mean_service, capacity


def summary_table(strategy_records: dict[str, list[dict]]) -> pd.DataFrame:
    """Long-format summary table: strategy x metric -> (mean, ci_low, ci_high)."""
    rows = []
    for strategy in STRATEGY_ORDER:
        records = strategy_records[strategy]
        for metric in METRIC_COLUMNS:
            m, lo, hi = mean_ci([r[metric] for r in records])
            rows.append(
                {
                    "strategy": strategy,
                    "metric": metric,
                    "mean": m,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_seeds": len(records),
                }
            )
    return pd.DataFrame(rows)


def print_summary(strategy_records: dict[str, list[dict]]) -> None:
    df = summary_table(strategy_records)
    display = ["mean_wait_min", "median_wait_min", "max_wait_min",
               "jain_fairness", "starvation_count", "mean_bay_utilization"]
    labels = [STRATEGY_LABELS[s] for s in STRATEGY_ORDER]
    width = max(22, max(len(l) for l in labels)) + 2
    print()
    print(" " * 22 + "".join(f"{l:>{width}}" for l in labels))
    for metric in display:
        line = f"{metric:>22}"
        for strategy in STRATEGY_ORDER:
            row = df[(df.strategy == strategy) & (df.metric == metric)].iloc[0]
            line += f"{_format_ci(row['mean'], row['ci_low'], row['ci_high']):>{width}}"
        print(line)
    print()


# ---------------------------------------------------------------------------
# Experiment A — baseline
# ---------------------------------------------------------------------------
def experiment_a(seeds: int = 30, rate: float = 2.0, duration: float = 7200.0,
                 warmup: float = 720.0) -> None:
    _set_style()
    cfg = SimulationConfig(arrival_rate=rate, duration=duration, warmup=warmup)
    seeds_list = list(range(seeds))

    records, waits = {}, {}
    for strategy in STRATEGY_ORDER:
        records[strategy], waits[strategy] = run_replicates(
            lambda s=strategy: make_allocator(s), cfg, seeds_list
        )

    summary_table(records).to_csv(RESULTS_DIR / "experiment_a_baseline.csv", index=False)
    pd.DataFrame(
        [
            {**r, "strategy": s}
            for s in STRATEGY_ORDER
            for r in records[s]
        ]
    ).to_csv(RESULTS_DIR / "experiment_a_raw.csv", index=False)

    # Box plot of the wait-time distributions.
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    data = pd.DataFrame(
        {
            "strategy": [s for s in STRATEGY_ORDER for _ in waits[s]],
            "wait_time_min": [w for s in STRATEGY_ORDER for w in waits[s]],
        }
    )
    sns.boxplot(
        data=data, x="strategy", y="wait_time_min",
        order=STRATEGY_ORDER, hue="strategy", palette=STRATEGY_COLORS,
        legend=False, width=0.55, linewidth=1.0, fliersize=2.0, ax=ax,
    )
    ax.set_ylabel("Wait time (minutes)")
    ax.set_xlabel("")
    ax.set_xticks(range(len(STRATEGY_ORDER)))
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in STRATEGY_ORDER])
    ax.set_title(f"Wait-time distribution by strategy (load ≈ {rate:.1f} cars/h, {seeds} seeds)")
    fig.savefig(RESULTS_DIR / "experiment_a_boxplot.png", bbox_inches="tight")
    plt.close(fig)

    print(f"[Experiment A] saved {RESULTS_DIR/'experiment_a_baseline.csv'}, "
          f"{RESULTS_DIR/'experiment_a_raw.csv'}, {RESULTS_DIR/'experiment_a_boxplot.png'}")
    print_summary(records)


# ---------------------------------------------------------------------------
# Experiment B — load sensitivity
# ---------------------------------------------------------------------------
def experiment_b(rates=(1.0, 1.5, 2.0, 2.3, 2.6), seeds: int = 15,
                 duration: float = 7200.0, warmup: float = 720.0) -> None:
    _set_style()
    seeds_list = list(range(seeds))

    # Calibrate capacity so each arrival rate can be reported as an offered load.
    base_cfg = SimulationConfig(arrival_rate=min(rates), duration=duration, warmup=warmup)
    mean_service, capacity = measure_capacity(base_cfg)
    print(f"[Experiment B] measured mean service time = {mean_service:.1f} min; "
          f"capacity = {capacity:.2f} cars/h")

    rows = []
    agg = {s: {m: [] for m in ("mean_wait_min", "max_wait_min", "jain_fairness")}
           for s in STRATEGY_ORDER}

    for rate in rates:
        load_factor = rate / capacity
        cfg = SimulationConfig(arrival_rate=rate, duration=duration, warmup=warmup)
        for strategy in STRATEGY_ORDER:
            records, _ = run_replicates(
                lambda s=strategy: make_allocator(s), cfg, seeds_list)
            for metric in ("mean_wait_min", "max_wait_min", "jain_fairness"):
                m, lo, hi = mean_ci([r[metric] for r in records])
                agg[strategy][metric].append((m, lo, hi))
            for r in records:
                rows.append({**r, "arrival_rate": rate, "load_factor": load_factor,
                             "strategy": strategy})

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "experiment_b_load_sensitivity.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for ax, (metric, ylabel, title) in zip(
        axes,
        [("mean_wait_min", "Mean wait time (min)", "Mean wait time"),
         ("max_wait_min", "Max wait time (min)", "Worst-case wait (starvation)")],
    ):
        for strategy in STRATEGY_ORDER:
            means = [agg[strategy][metric][i][0] for i in range(len(rates))]
            los = [agg[strategy][metric][i][1] for i in range(len(rates))]
            his = [agg[strategy][metric][i][2] for i in range(len(rates))]
            ax.plot(rates, means, marker="o", ms=4, lw=1.8,
                    color=STRATEGY_COLORS[strategy], label=STRATEGY_LABELS[strategy])
            ax.fill_between(rates, los, his, alpha=0.18, color=STRATEGY_COLORS[strategy])
        ax.set_xlabel("Arrival rate (cars / hour)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    axes[0].legend(frameon=True, loc="upper left")
    fig.suptitle(f"Load sensitivity of wait time across strategies "
                 f"(capacity ≈ {capacity:.1f} cars/h)")
    fig.savefig(RESULTS_DIR / "experiment_b_load_sensitivity.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Experiment B] saved {RESULTS_DIR/'experiment_b_load_sensitivity.csv'} and .png")


# ---------------------------------------------------------------------------
# Experiment C — ageing constant k sensitivity
# ---------------------------------------------------------------------------
def experiment_c(k_values=None, seeds: int = 15, rate: float = 2.9,
                 duration: float = 7200.0, warmup: float = 720.0,
                 wait_cap: float = float("inf")) -> None:
    _set_style()
    if k_values is None:
        k_values = list(np.logspace(-3, 0, 12))  # 0.001 .. 1.0
    seeds_list = list(range(seeds))
    cfg = SimulationConfig(arrival_rate=rate, duration=duration, warmup=warmup)

    # NOTE: the hard wait-time cap is disabled here (wait_cap = inf) to *isolate*
    # the effect of the ageing constant k.  With the cap active it dominates the
    # fairness behaviour and flattens the curve; the cap's own contribution is
    # quantified separately in Experiments A/B/D.  The load is deliberately near
    # saturation (rho ~ 1.0) because the ageing discount only becomes observable
    # once cars wait long enough for k to matter; the within-strategy comparison
    # is still valid since CRN gives every k the identical arrival stream.
    rows = []
    mean_wait, lo_wait, hi_wait = [], [], []
    mean_jain, lo_jain, hi_jain = [], [], []
    for k in k_values:
        alloc = SocAwareAgingAllocator(k=k, wait_cap=wait_cap)
        records = []
        for seed in seeds_list:
            m, _ = run_single(alloc, replace(cfg, seed=seed))
            m = dict(m); m["seed"] = seed; m["k"] = k
            records.append(m)
            rows.append(m)
        mw, lw, hw = mean_ci([r["mean_wait_min"] for r in records])
        mj, lj, hj = mean_ci([r["jain_fairness"] for r in records])
        mean_wait.append(mw); lo_wait.append(lw); hi_wait.append(hw)
        mean_jain.append(mj); lo_jain.append(lj); hi_jain.append(hj)

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "experiment_c_k_sensitivity.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(6.8, 4.6))
    ax1.plot(k_values, mean_wait, marker="o", ms=4, lw=1.8, color="#4c78a8",
             label="Mean wait time")
    ax1.fill_between(k_values, lo_wait, hi_wait, alpha=0.18, color="#4c78a8")
    ax1.set_xscale("log")
    ax1.set_xlabel("Ageing constant $k$")
    ax1.set_ylabel("Mean wait time (min)", color="#4c78a8")
    ax1.tick_params(axis="y", labelcolor="#4c78a8")

    ax2 = ax1.twinx()
    ax2.plot(k_values, mean_jain, marker="s", ms=4, lw=1.8, color="#e45756",
             label="Jain's fairness index")
    ax2.fill_between(k_values, lo_jain, hi_jain, alpha=0.15, color="#e45756")
    ax2.set_ylabel("Jain's fairness index", color="#e45756")
    ax2.tick_params(axis="y", labelcolor="#e45756")
    ax2.set_ylim(0.0, 1.0)

    lines = [ax1.get_lines()[0], ax2.get_lines()[0]]
    ax1.legend(lines, [l.get_label() for l in lines], loc="center")
    ax1.set_title("Efficiency–fairness trade-off vs ageing constant $k$ (hard cap disabled)")
    fig.savefig(RESULTS_DIR / "experiment_c_k_sensitivity.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Experiment C] saved {RESULTS_DIR/'experiment_c_k_sensitivity.csv'} and .png")


# ---------------------------------------------------------------------------
# Experiment D — bay heterogeneity
# ---------------------------------------------------------------------------
def experiment_d(seeds: int = 15, rate: float = 2.0, duration: float = 7200.0,
                 warmup: float = 720.0) -> None:
    _set_style()
    seeds_list = list(range(seeds))
    configs = {
        "Homogeneous (4x40 kW)": (40.0, 40.0, 40.0, 40.0),
        "Heterogeneous (60/60/22/22 kW)": (60.0, 60.0, 22.0, 22.0),
    }

    rows = []
    summary_rows = []
    for cfg_name, bay_power in configs.items():
        cfg = SimulationConfig(bay_power_kw=bay_power, arrival_rate=rate,
                               duration=duration, warmup=warmup)
        for strategy in STRATEGY_ORDER:
            records, _ = run_replicates(
                lambda s=strategy: make_allocator(s), cfg, seeds_list)
            for r in records:
                rows.append({**r, "config": cfg_name, "strategy": strategy})
            for metric in ("mean_wait_min", "max_wait_min", "jain_fairness"):
                m, lo, hi = mean_ci([r[metric] for r in records])
                summary_rows.append({"config": cfg_name, "strategy": strategy,
                                     "metric": metric, "mean": m,
                                     "ci_low": lo, "ci_high": hi})

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "experiment_d_heterogeneity.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(
        RESULTS_DIR / "experiment_d_heterogeneity_summary.csv", index=False)

    # Grouped bar chart: mean wait (left) and Jain fairness (right).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    width = 0.28
    x = np.arange(len(configs))
    for ax, (metric, ylabel, title) in zip(
        axes,
        [("mean_wait_min", "Mean wait time (min)", "Mean wait time"),
         ("jain_fairness", "Jain's fairness index", "Fairness (Jain's index)")],
    ):
        for offset, strategy in enumerate(STRATEGY_ORDER):
            vals = []
            errs = []
            for cfg_name in configs:
                row = next(s for s in summary_rows
                           if s["config"] == cfg_name and s["strategy"] == strategy
                           and s["metric"] == metric)
                vals.append(row["mean"])
                errs.append([[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]])
            ax.bar(x + (offset - 1) * width, vals, width,
                   yerr=np.array(errs).reshape(2, -1), capsize=3,
                   color=STRATEGY_COLORS[strategy], label=STRATEGY_LABELS[strategy],
                   alpha=0.9, error_kw=dict(lw=1.0))
        ax.set_xticks(x)
        ax.set_xticklabels(["Homogeneous\n(4x40 kW)", "Heterogeneous\n(60/60/22/22 kW)"])
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    axes[0].legend(frameon=True, loc="upper left")
    fig.suptitle("Effect of bay heterogeneity across strategies")
    fig.savefig(RESULTS_DIR / "experiment_d_heterogeneity.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[Experiment D] saved {RESULTS_DIR/'experiment_d_heterogeneity.csv'} and .png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run EV bay-allocation experiments")
    parser.add_argument("--experiment", choices=["a", "b", "c", "d", "all"],
                        default="all", help="which experiment(s) to run")
    parser.add_argument("--seeds", type=int, default=None,
                        help="override the number of seeds")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.experiment in ("a", "all"):
        experiment_a(**( {"seeds": args.seeds} if args.seeds else {}))
    if args.experiment in ("b", "all"):
        experiment_b(**( {"seeds": args.seeds} if args.seeds else {}))
    if args.experiment in ("c", "all"):
        experiment_c(**( {"seeds": args.seeds} if args.seeds else {}))
    if args.experiment in ("d", "all"):
        experiment_d(**( {"seeds": args.seeds} if args.seeds else {}))


if __name__ == "__main__":
    main()

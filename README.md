# EV Bay Allocation — Discrete-Event Simulation

A SimPy-based discrete-event simulation of a **multi-bay EV charging station**
that compares three bay-allocation strategies on **wait time** and **fairness**:

| Strategy | Basis | Failure mode |
|---|---|---|
| **FCFS** | Arrival order to the soonest-free bay | Ignores charge time & power matching → high mean wait |
| **Pure SRPT** | Min total completion time (Hungarian) | Starves large/slow-charging cars under load |
| **SOC-aware + aging** *(proposed)* | Completion time + priority aging + hard cap | Requires tuning the aging constant `k` |

The proposed method blends the SOC-aware completion-time cost with a
**priority-aging score** and a **hard wait-time cap**, aiming to sit on the
efficiency–fairness frontier: near-SRPT efficiency with near-FCFS fairness.

---

## Installation

Requires Python ≥ 3.10.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Dependencies: `simpy`, `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`,
`pytest`.

---

## Project structure

```
src/ev_sim/
  models.py        Car / Bay dataclasses + Indian-market battery mix
  charge_curve.py  Piecewise CC/CV charging curve (time vs SOC)
  allocators.py    FCFS, SRPT, SOC-aware+aging (common assign() interface)
  simulation.py    SimPy engine + SimulationConfig / SimulationResult
  metrics.py       Jain's fairness, wait-time summaries, bay utilization
  experiments.py   Experiment runners A/B/C/D + publication plots
tests/             pytest unit tests (charge curve, allocators, metrics, engine)
results/           CSVs + figures produced by the experiments
run_experiments.py Reproducibility entry point
```

---

## Running the tests

```bash
.venv/bin/python -m pytest -q
```

The suite covers the charge-curve math (monotonicity, the "last 20% ≈ first
70%" calibration), each allocator in isolation (including the Hungarian
optimality and the incompatibility penalty), Jain's fairness against known
values, and simulation-level invariants plus multi-seed behavioural checks.

---

## Running the experiments

```bash
.venv/bin/python run_experiments.py --experiment all     # A + B + C + D
.venv/bin/python run_experiments.py --experiment a       # just the baseline
.venv/bin/python run_experiments.py --experiment a --seeds 50   # more seeds
```

Every experiment writes one or more CSVs and a publication-style figure to
`results/`. All runs are seeded and logged for reproducibility.

| Experiment | Command | Output |
|---|---|---|
| **A — Baseline** | `--experiment a` | `experiment_a_baseline.csv`, `experiment_a_raw.csv`, `experiment_a_boxplot.png` |
| **B — Load sensitivity** | `--experiment b` | `experiment_b_load_sensitivity.csv`, `.png` |
| **C — k sensitivity** | `--experiment c` | `experiment_c_k_sensitivity.csv`, `.png` |
| **D — Bay heterogeneity** | `--experiment d` | `experiment_d_heterogeneity.csv`, `experiment_d_heterogeneity_summary.csv`, `.png` |

Experiments **A/B/C** use the default 4-bay configuration
(`60/60/22/22 kW`); Experiment **D** explicitly contrasts this heterogeneous
configuration with a homogeneous `4×40 kW` one. Experiment **B** calibrates the
station capacity first and records the offered load `ρ = arrival_rate /
capacity` alongside each arrival rate.

---

## Methodology

### Charging curve (`charge_curve.py`)

Lithium-ion charging is modelled piecewise:

* **CC (constant current), 0–80% SOC** — linear in time.
* **CV (constant voltage), 80–100% SOC** — a concave taper
  `g(u) = 1 − (1 − u)²`, calibrated so the **last 20% takes as long as the
  first 70%** (a standard Li-ion rule of thumb). A full charge therefore takes
  `1.5 × T_full`, where `T_full = capacity / power`.

`time_to_charge(...)` returns minutes and is monotone by construction (it is a
difference of two evaluations of the monotone primitive
`time_from_0_to_soc`). Effective power is `min(bay.rated_kw, car.max_accept_kw)`
so a bay cannot charge faster than a car can accept.

### Vehicle mix (`models.py`)

Battery capacities are sampled from a three-tier Indian-market mix
(≈50/35/15):

| Tier | Capacity | `max_accept_kw` | Example |
|---|---|---|---|
| small | 19–24 kWh | 7–22 kW | Tiago EV, eC3 |
| mid | 30–40 kWh | 30–80 kW | Nexon EV, Punch EV |
| premium | 65–75 kWh | 80–150 kW | eC9, XUV400 LR |

### Allocation strategies (`allocators.py`)

All strategies expose one interface, `assign(cars, bays, now) -> {car_id: bay_id}`,
and the engine re-solves on **every state change** (car arrival, bay freed).

* **FCFS** — sort cars by arrival, bays by free time, zip.
* **SRPT** — cost = predicted completion time; the **Hungarian algorithm**
  (`scipy.optimize.linear_sum_assignment`) finds the min-total-completion
  matching. Incompatible pairings get a heavy finite penalty (not a hard
  exclusion) so the matrix stays a complete bipartite graph.
* **SOC-aware + aging** — completion time is split into a bay-availability term
  plus an *aged work* term:

  ```
  cost[car, bay] = max(now, bay.free) + charge_time(car, bay) / (1 + k · wait)
  ```

  This is `priority_score = charge_time_needed / (1 + k·wait)` applied as a
  weight on the work term. At `k = 0` it reduces *exactly* to SRPT. A **hard
  wait-time cap** (`wait_cap`, default 45 min) force-assigns cars that have
  waited past the ceiling to the soonest-free bay in arrival order, regardless
  of score.

### Simulation (`simulation.py`)

* Poisson arrivals (configurable mean rate, cars/hour).
* Initial SOC ~ Beta(2.2, 3.0) scaled to [2, 90]% (low–moderate arrival SOC);
  70% target 100%, 30% a partial top-up in [80, 95]%.
* Changeover (swap) buffer sampled uniformly in [2, 5] min, added to each bay's
  predicted free time.
* **Common random numbers (CRN)**: three independent RNG streams are derived
  from each seed via `SeedSequence.spawn(3)` (arrivals / car attributes /
  changeover). Arrival times, car attributes, and each car's swap-out buffer are
  all sampled in arrival order, so the same seed yields an *identical* car
  stream across strategies — a paired, variance-reduced design.
* A `warmup` period discards the initial transient. Wait-time metrics are
  computed over cars that **began** service after warmup (their wait is fully
  observed at service start); cars still queued at the horizon are
  **right-censored** and reported as `n_censored` rather than silently dropped,
  since under SRPT the censored cars are disproportionately the starved ones.

### Metrics (`metrics.py`)

* Mean / median / max wait time.
* **Jain's fairness index** `J = (Σx)² / (n Σx²)` over per-car wait times
  (`1` = perfectly equal). Note that a system where most cars wait ~0 and a few
  wait long has a low `J`; this is a comparative, not absolute, fairness measure.
* Starvation count (cars waiting ≥ 30 min).
* `n_censored` — cars still queued at the horizon (right-censored wait).
* Per-bay utilization over the steady-state window.

---

## What the results show

**Experiment A (baseline, `ρ ≈ 0.7`, 30 seeds)** — mean ± 95% CI:

| Metric | FCFS | Pure SRPT | SOC-aware + aging |
|---|---|---|---|
| Mean wait (min) | 17.7 | **13.4** | 15.5 |
| Max wait (min) | 130.1 | 142.8 | **112.6** |
| Jain's fairness | 0.30 | 0.20 | 0.30 |

SRPT is the most *efficient* (lowest mean wait) but the most *unfair* (lowest
Jain's, highest worst-case wait). The proposed method recovers most of SRPT's
efficiency while bounding the worst-case wait *below* even FCFS.

**Experiment B (load sweep)** — SRPT's max wait diverges upward as the offered
load `ρ` grows (starvation), while the proposed method's worst-case wait stays
below FCFS across the whole range. Capacity is calibrated first
(`≈ 2.9 cars/h`) and `ρ` is recorded per rate; the top rate reaches `ρ ≈ 0.9`.

**Experiment C (k sweep, hard cap disabled to isolate `k`)** — run near
saturation (`ρ ≈ 1.0`) where the ageing discount is observable: increasing `k`
trades efficiency for fairness, with mean wait rising and Jain's fairness rising
monotonically, tracing the efficiency–fairness frontier.

**Experiment D (bay heterogeneity)** — with unequal bays (60/60/22/22 kW),
capacity matching matters: FCFS (which ignores power matching) degrades far
more when bays are unequal than the completion-time-aware methods, so the
proposed method captures most of SRPT's matching benefit while staying fairer.

> Exact numbers depend on the seed count and defaults; regenerate with the
> commands above. Figures are saved at 300 dpi with publication-style labels.

---

## Reproducibility

* Every run is seeded; the seed is logged in the raw CSVs.
* CRN ensures the same seed produces identical arrival/car streams across
  strategies, so the strategy comparison is paired at the *point-estimate*
  level. Reported 95% CIs are per-strategy Student-t intervals across seeds
  (not paired-difference intervals).
* Re-run everything from scratch:

  ```bash
  .venv/bin/python -m pytest -q
  .venv/bin/python run_experiments.py --experiment all
  ```

---

## Author

**Naveen** — Project design & implementation.

## License

All rights reserved.

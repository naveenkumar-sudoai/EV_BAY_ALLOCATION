# EV Bay Allocation System

## Intelligent SOC-Aware Bay Allocation for Multi-Bay EV Charging Stations

A systems analysis and design architecture for **dynamic, capacity-matched, fairness-aware charging bay scheduling** — replacing simple first-come-first-served with intelligent bay-to-car matching that minimizes overall waiting time.

---

## The Problem

In a multi-bay EV charging station (e.g., 6 bays), cars are conventionally served **first-come-first-served** — with no regard to how much charge each car actually needs or how soon each bay will free up. This leads to:

- A car needing a 10-minute top-up sitting behind one that needs an 80% full charge
- High-power bays wasted on vehicles that can't accept the full rate
- No fairness guarantees — large-need cars can get starved indefinitely

## The Core Idea

Replace static queuing with **dynamic, SOC-aware bay allocation**:

1. Continuously track the remaining charge time of every occupied bay
2. When a new car joins the queue, match it to the bay that **minimizes overall waiting time** across the system
3. A car needing only a small top-up gets routed to the bay finishing soonest
4. Priority aging prevents starvation of large-need vehicles

**Structurally**, this is an **online scheduling / constrained assignment problem**, related to Shortest Remaining Processing Time (SRPT) scheduling and job-shop scheduling with machine-eligibility constraints.

---

## Why Naive Approaches Fail

| Issue | Description |
|---|---|
| **Non-linear charging curve** | Li-ion cells charge in CC/CV phases — the last 20% often takes as long as the first 70%. Raw percentage-remaining is a poor proxy for actual time-remaining. |
| **Battery capacity variance** | 10% on a 40 kWh car ≠ 10% on a 100 kWh car. kWh-remaining (SOC × capacity) is needed for comparable estimates. |
| **Bay power vs. vehicle acceptance** | Effective charging power = `min(bay_kW, vehicle_max_kW)`. Not all vehicles can accept a bay's full power. |
| **Starvation risk** | Pure SRPT (shortest-job-first) starves large-need cars under sustained queue load. |
| **Changeover overhead** | Physically swapping cars takes 2–5 minutes — treating transitions as instantaneous makes schedules systematically over-optimistic. |
| **Telemetry access** | Accurate SOC data requires vehicle-side integration (ISO 15118) and station-side protocol support (OCPP). |

---

## Proposed System Design

### 1. Constrained Optimal Assignment

Re-solve a **weighted bipartite matching** between waiting cars and available/soon-to-be-available bays on every state change (car arrives, bay frees up). Uses the **Hungarian algorithm** (or greedy approximation for embedded-scale queues).

### 2. Accurate Time Estimation

```
time_to_target(car, bay) = f(kWh_needed, charge_curve_model, min(bay_kW, vehicle_max_kW))
```

Uses a **piecewise CC/CV curve model** rather than linear percentage assumptions.

### 3. Fairness via Priority Aging

Borrowed from OS scheduler design (e.g., Linux CFS virtual-runtime fairness):

```
priority(car) = charge_time_needed / (1 + k × wait_time_elapsed)
```

- **Lower score → served sooner**
- As wait time grows, even an 80%-need car eventually outranks a freshly arrived 20%-need car
- `k` is a tunable aging constant
- **Hard wait-time cap** guarantees no car waits beyond a fixed ceiling
- **Periodic re-ranking window** avoids pathological reshuffling

### 4. Practical Guards

- **Changeover buffer** (2–5 min) added to every bay's predicted free time
- **SOC input** via OCPP/ISO 15118 where hardware allows; app-based self-report with plausibility checks as MVP fallback
- **Re-evaluate on every state change** (not a one-shot plan)

---

## Allocation Strategies Compared

| Strategy | Basis | Key Flaw |
|---|---|---|
| **First-Come-First-Served** | Arrival order only | Bays occupied far longer than necessary |
| **Pure SRPT** (shortest need first) | Charge time needed | Starves large-need cars under sustained load |
| **SOC-aware + aging** (proposed) | Charge time + wait time + capacity matching | Requires accurate telemetry and tuning of `k` |
| **Auction / mechanism design** | Reported user preferences | Complex; assumes truthful reporting |

---

## System Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Telemetry   │────▶│  Allocation       │────▶│  Bay Assignment  │
│  (SOC, kW)   │     │  Engine           │     │  Output          │
└─────────────┘     │  (Bipartite       │     └─────────────────┘
                    │   Matching +      │
┌─────────────┐     │   Priority Aging) │     ┌─────────────────┐
│  Queue State │────▶│                   │────▶│  User Display    │
│  (waiting)   │     └──────────────────┘     └─────────────────┘
└─────────────┘
```

---

## Key Research Foundations

- Shortest Remaining Processing Time (SRPT) scheduling
- Hungarian algorithm for bipartite matching
- Linux CFS virtual-runtime fairness (priority aging)
- ISO 15118 (vehicle-to-grid communication)
- OCPP (Open Charge Point Protocol)

---

## Project Status

**Phase**: Design & Architecture (Scoping Note)

This is a systems analysis and design document. The concept bridges an identified gap: while SOC-based prioritization, changeover handling, and online scheduling theory exist separately, a combined system doing **capacity-matched, fairness-aware, curve-accurate bay assignment** within a single station is not commonly deployed.

---

## Author

**Naveen** — Project Scoping & Systems Analysis

---

## License

All rights reserved. This design architecture is a project scoping document.

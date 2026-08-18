"""EV Bay Allocation Simulation.

Discrete-event simulation comparing bay-allocation strategies for a multi-bay
EV charging station: First-Come-First-Served (FCFS), Pure Shortest-Remaining-
Processing-Time (SRPT), and a proposed SOC-aware + priority-aging policy.

The experiment runner lives in :mod:`ev_sim.experiments` (imported explicitly,
since it pulls in matplotlib and selects a headless backend).
"""

from .allocators import (
    FCFSAllocator,
    SRPTAllocator,
    SocAwareAgingAllocator,
    charge_time_min,
    effective_power_kw,
)
from .charge_curve import time_to_charge
from .metrics import bay_utilization_fraction, jains_fairness, summarize
from .models import BATTERY_TIERS, Bay, Car, sample_battery_capacity, sample_max_accept_kw
from .simulation import Simulation, SimulationConfig, SimulationResult

__version__ = "0.1.0"

__all__ = [
    # models
    "Car",
    "Bay",
    "BATTERY_TIERS",
    "sample_battery_capacity",
    "sample_max_accept_kw",
    # charge curve
    "time_to_charge",
    # allocators
    "FCFSAllocator",
    "SRPTAllocator",
    "SocAwareAgingAllocator",
    "charge_time_min",
    "effective_power_kw",
    # simulation
    "Simulation",
    "SimulationConfig",
    "SimulationResult",
    # metrics
    "jains_fairness",
    "summarize",
    "bay_utilization_fraction",
]

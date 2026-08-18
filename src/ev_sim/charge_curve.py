"""Piecewise CC/CV battery charging curve model.

Lithium-ion cells charge in two regimes:

* **CC (constant current)** — from ~0% to ~80% SOC the charger holds current
  constant and SOC rises (nearly) linearly with time.
* **CV (constant voltage)** — above the knee (~80% SOC) the current tapers off
  and the remaining SOC is delivered much more slowly.

We model this with a *piecewise* time-vs-SOC curve:

* ``0% -> 80%``: linear in SOC (CC regime).
* ``80% -> 100%``: a concave taper ``g(u) = 1 - (1 - u)^2`` over the CV
  fraction ``u``, which reaches 100% in finite time.

The taper is calibrated so that the **last 20% of SOC takes exactly as long as
the first 70%** — a standard Li-ion rule of thumb.  Concretely the CV window is
``TAIL_FACTOR * T_full`` where ``T_full = capacity / power`` is the nominal
time to add 100% SOC at the effective power.  This makes a full charge take
``1.5 * T_full`` in total, which is realistic for the 80% knee model.

All times are returned in **minutes**.
"""
from __future__ import annotations

import math

#: SOC (%) at which the CC -> CV knee occurs.
CC_SOC_PCT = 80.0

#: The CV window (last 20%) lasts ``TAIL_FACTOR`` times the nominal full-linear
#: time ``T_full``.  With ``TAIL_FACTOR = 0.7`` the last 20% takes as long as
#: the first 70%, which is the reference property asserted by the unit tests.
TAIL_FACTOR = 0.7

#: Upper SOC bound (percent).  SOC is always clamped to [0, FULL_SOC_PCT].
FULL_SOC_PCT = 100.0


def _nominal_full_time_hours(battery_capacity_kwh: float, effective_kw: float) -> float:
    """Nominal time (hours) to deliver 100% SOC at constant effective power."""
    return battery_capacity_kwh / effective_kw


def time_from_0_to_soc_hours(
    soc_pct: float, effective_kw: float, battery_capacity_kwh: float
) -> float:
    """Time in hours to charge a battery from 0% to ``soc_pct``.

    This is the monotone primitive of the charge curve; the time to go between
    two arbitrary SOC levels is a difference of two evaluations (see
    :func:`time_to_charge`), which guarantees monotonicity for free.
    """
    if effective_kw <= 0.0:
        return math.inf

    soc = min(max(soc_pct, 0.0), FULL_SOC_PCT)
    t_full = _nominal_full_time_hours(battery_capacity_kwh, effective_kw)

    if soc <= CC_SOC_PCT:
        # CC regime: linear in SOC.
        return t_full * (soc / FULL_SOC_PCT)

    # CV regime: concave taper over the last (100 - CC_SOC_PCT) percent.
    u = (soc - CC_SOC_PCT) / (FULL_SOC_PCT - CC_SOC_PCT)
    t_cv = TAIL_FACTOR * t_full
    taper = 1.0 - (1.0 - u) ** 2
    return t_full * (CC_SOC_PCT / FULL_SOC_PCT) + t_cv * taper


def time_to_charge(
    kwh_needed: float,
    current_soc_pct: float,
    target_soc_pct: float,
    effective_kw: float,
    battery_capacity_kwh: float,
) -> float:
    """Minutes to charge from ``current_soc_pct`` to ``target_soc_pct``.

    Parameters
    ----------
    kwh_needed:
        Energy required (kWh).  Kept for API clarity and downstream callers;
        the authoritative computation uses the SOC positions, which is what
        makes the CC/CV nonlinearity correct.  ``kwh_needed`` should equal
        ``battery_capacity_kwh * (target - current) / 100``.
    current_soc_pct, target_soc_pct:
        Starting / ending state of charge (percent).
    effective_kw:
        Effective charging power = ``min(bay.rated_kw, car.max_accept_kw)``.
    battery_capacity_kwh:
        Usable battery capacity in kWh.
    """
    if effective_kw <= 0.0:
        return math.inf

    t_current = time_from_0_to_soc_hours(current_soc_pct, effective_kw, battery_capacity_kwh)
    t_target = time_from_0_to_soc_hours(target_soc_pct, effective_kw, battery_capacity_kwh)
    return max(0.0, (t_target - t_current) * 60.0)

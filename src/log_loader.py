"""
log_loader.py
=============
Loads and parses the Daksh Soil Sensor Reading Log (hand-recorded pot
experiment, 22–26 May 2026) into structured (s, a, r, s') transitions
suitable for simulator calibration and offline RL warm-starting.

Experimental setup
------------------
5 pots, sampled roughly every hour from 08:45–17:15 local time:
  Pot 1 — Sandy Soil     (control; different physics, NOT the target soil)
  Pot 2 — Net House Soil (experimental soil, same as the planned T0/T1/T2
  Pot 3 — Net House Soil  plots in the greenhouse trial)
  Pot 4 — Net House Soil
  Pot 5 — Net House Soil

Irrigation protocol (as logged in Remarks):
  - Morning pulse at ~08:30 local, 1 L per pot, daily
  - Some days a second pulse at ~12:50 PM, also 1 L
  - Action is recorded at the NEXT reading after irrigation (e.g. 8:30
    irrigation shows up as the action label on the 8:45 reading)

Known data quality issues (flagged in soil_log.csv):
  - 22 May: only afternoon data; no pre-irrigation morning state
  - 25 May 15:15: Pot 5 M/T appear transposed (38%/95°C → excluded)
  - No pre-irrigation reading (8:00 AM) exists yet; the 8:45 reading is
    already post-irrigation

Volume to mm conversion
-----------------------
1 L applied to a pot of unknown area. We store actions in mL (0 or 1000)
and expose a calibrated `mL_per_pct` parameter:
    ΔM_pct ≈ volume_mL / mL_per_pct

`mL_per_pct` is estimated from the data:
  The net house soil shows a post-irrigation reading of ~95% (sensor FC).
  Without a pre-irrigation observation, we bound the response: on May 23,
  Pot 2 was at 65% at 17:15 the previous evening, suggesting the next
  morning the soil had drained to perhaps 50-60% before the 8:30 pulse.
  1000 mL raising ~35-45% points → mL_per_pct ≈ 22–29.
  We use 25 mL/% as a conservative midpoint, pending confirmation of
  pot dimensions or a pre-irrigation sensor reading.

Drying dynamics (net house soil, from no-irrigation rows on 23 May)
--------------------------------------------------------------------
  08:45 → 17:15 (8.5 h), no second irrigation:
    Pot 2: 95% → 65%  = −30% in 8.5 h ≈ −3.5 %/h
    Pot 4: 95% → 94%  (nearly saturated all day; ≥3 h lag before drain)
  Implied tau_drain (surface) ≈ 14–20 h for net house soil.
  Compare: sandy soil drains ~40%/h → tau ≈ 1.5–2 h.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Pots to use as the "real experimental soil" calibration signal.
# Pot 1 (sandy) has very different physics and is used only for sandy-soil
# τ_drain estimation, NOT for the net house soil parameters.
_NET_HOUSE_POTS = ["p2", "p3", "p4", "p5"]
_SANDY_POT      = "p1"   # alias for "sandy" columns

# Sensor apparent saturation (sensor reports ≥95% as effectively FC)
SENSOR_FC = 95.0
SENSOR_WP = 15.0    # below this the sensor is unreliable; flag as bad data


@dataclass
class LogReading:
    """One timestamped, per-pot reading."""
    dt: pd.Timestamp
    pot_id: str          # "sandy" | "p2" | "p3" | "p4" | "p5"
    M: float             # soil moisture %
    T: float             # soil temperature °C
    action_mL: float     # irrigation applied just BEFORE this reading (mL)
    rain: bool           # True if remark mentions rain at this timestep


@dataclass
class LogTransition:
    """
    A (s, a, s') tuple extracted from the log, where:
      s  = (M, T) at time t
      a  = irrigation_mL applied between t and t+1
      s' = (M, T) at t+1
    Only created between consecutive readings of the same pot on the same day.
    """
    pot_id: str
    dt: pd.Timestamp
    M: float
    T: float
    action_mL: float
    M_next: float
    T_next: float
    rain: bool           # if True, transition is flagged; exclude from drying-calibration


def load_log_csv(path: str | Path) -> pd.DataFrame:
    """
    Read the soil_log.csv into a tidy DataFrame.
    Returns one row per (datetime, pot) with columns:
      dt, pot_id, M, T, action_mL, rain
    Excludes rows flagged DATA_QUALITY or with missing M/T values.
    """
    raw = pd.read_csv(path, dtype=str)
    raw["dt"] = pd.to_datetime(
        raw["datetime"].str.strip() + " " + raw["time_str"].str.strip()
    )

    pot_cols = {
        "sandy": ("sandy_M", "sandy_T"),
        "p2":    ("p2_M",    "p2_T"),
        "p3":    ("p3_M",    "p3_T"),
        "p4":    ("p4_M",    "p4_T"),
        "p5":    ("p5_M",    "p5_T"),
    }

    rows: list[dict] = []
    for _, row in raw.iterrows():
        remark = str(row.get("remark", "")).lower()
        # Skip rows with known data quality issues
        if "data_quality" in remark:
            continue
        action_mL = float(row["action_mL"]) if pd.notna(row["action_mL"]) else 0.0
        rain = "rain" in remark

        for pot_id, (mc, tc) in pot_cols.items():
            m_raw = row.get(mc, "")
            t_raw = row.get(tc, "")
            if pd.isna(m_raw) or str(m_raw).strip() == "":
                continue
            try:
                M = float(m_raw)
                T = float(t_raw)
            except (ValueError, TypeError):
                continue
            # Sanity bounds
            if not (0.0 <= M <= 110.0) or not (10.0 <= T <= 70.0):
                continue
            rows.append({
                "dt": row["dt"], "pot_id": pot_id,
                "M": M, "T": T,
                "action_mL": action_mL, "rain": rain,
            })

    return pd.DataFrame(rows).sort_values("dt").reset_index(drop=True)


def build_log_transitions(df: pd.DataFrame) -> List[LogTransition]:
    """
    From the tidy DataFrame, build consecutive (s, a, s') transitions
    within each (pot, day) group.

    The action label is the action_mL recorded at the CURRENT row, meaning
    the irrigation that happened just before this reading. The transition is:
      state  = (M, T) at current row  (post-irrigation if action > 0)
      action = action_mL at NEXT row
      next_state = (M, T) at next row
    This matches the intended semantics: state → action → next_state.
    """
    transitions: list[LogTransition] = []
    df["date"] = df["dt"].dt.date

    for (pot_id, date), grp in df.groupby(["pot_id", "date"]):
        grp = grp.sort_values("dt").reset_index(drop=True)
        for i in range(len(grp) - 1):
            cur  = grp.iloc[i]
            nxt  = grp.iloc[i + 1]
            # Only connect rows that are at most 2 h apart (data gaps > 2h
            # produce unreliable ΔM estimates)
            dt_gap = (nxt["dt"] - cur["dt"]).total_seconds() / 3600.0
            if dt_gap > 2.1:
                continue
            transitions.append(LogTransition(
                pot_id=pot_id,
                dt=cur["dt"],
                M=float(cur["M"]),
                T=float(cur["T"]),
                action_mL=float(nxt["action_mL"]),  # action BETWEEN cur→nxt
                M_next=float(nxt["M"]),
                T_next=float(nxt["T"]),
                rain=bool(cur["rain"] or nxt["rain"]),
            ))

    return transitions


def split_by_soil(
    transitions: List[LogTransition],
) -> Tuple[List[LogTransition], List[LogTransition]]:
    """
    Split transitions into:
      sandy_trans   — Pot 1, used only for sandy-soil τ calibration
      net_trans     — Pots 2-5, used for net house soil calibration and
                      for building the RL offline buffer
    """
    sandy = [t for t in transitions if t.pot_id == "sandy"]
    net   = [t for t in transitions if t.pot_id in _NET_HOUSE_POTS]
    return sandy, net


def net_soil_drying_rate(net_trans: List[LogTransition]) -> float:
    """
    Estimate mean drying rate (%/h) for net house soil under zero
    irrigation and no rain.

    We EXCLUDE transitions where the starting moisture is at or near
    sensor FC (>= 88%). When soil is saturated, the sensor rounds to
    95% for several hours before registering a drop — including these
    transitions conflates sensor saturation artefacts with real drying.
    Only transitions well below FC are included in the rate estimate.

    Returns mean drying rate (positive %/h), or nan if insufficient data.
    """
    rates = []
    for t in net_trans:
        if t.action_mL > 0 or t.rain:
            continue
        # Skip if either endpoint is at sensor floor (unreliable)
        if t.M <= SENSOR_WP or t.M_next <= SENSOR_WP:
            continue
        # Skip saturated plateau — sensor caps at 95%, transitions in this
        # zone do not reflect true drying rates (both are rounded to FC).
        if t.M >= 88.0:
            continue
        dM = t.M_next - t.M   # negative = drying
        if dM < 0:             # only count actual drying, not noise upswings
            rates.append(-dM)
    return float(np.mean(rates)) if rates else float("nan")


# ---------------------------------------------------------------------------
# Convenience summary
# ---------------------------------------------------------------------------

def summarise_log(path: str | Path) -> None:
    """Print a human-readable summary of the log — call before training."""
    df = load_log_csv(path)
    trans = build_log_transitions(df)
    sandy_trans, net_trans = split_by_soil(trans)

    irrig_events = [t for t in net_trans if t.action_mL > 0]
    drying_rate  = net_soil_drying_rate(net_trans)

    print("=" * 60)
    print("Soil Sensor Reading Log — summary")
    print("=" * 60)
    print(f"Date range   : {df['dt'].min().date()} → {df['dt'].max().date()}")
    print(f"Days with data: {df['dt'].dt.date.nunique()}")
    print(f"Total readings (all pots): {len(df)}")
    print()
    print(f"Net house soil transitions: {len(net_trans)}")
    print(f"  of which with irrigation: {len(irrig_events)}")
    print(f"  of which without irrigat: {len(net_trans) - len(irrig_events)}")
    print(f"  rain-flagged transitions: {sum(t.rain for t in net_trans)}")
    print()
    print(f"Sandy soil transitions: {len(sandy_trans)}")
    print()
    print(f"Net house soil mean drying rate: {drying_rate:.2f} %/h")
    implied_tau = (SENSOR_FC - SENSOR_WP) / max(drying_rate, 0.1)
    print(f"Implied τ_drain (net house)   : {implied_tau:.1f} h")
    print()
    print("Irrigation events:")
    for t in irrig_events:
        print(f"  {t.dt}  pot={t.pot_id}  {t.action_mL:.0f} mL")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "./data/soil_log.csv"
    summarise_log(path)

"""
field_loader.py
===============
Loads the real shade-house field sensor logs (Agri-Monitor v5 / ESP32 nodes,
files TST1234_*.csv) and reconciles them with the hand-kept irrigation/rain
event log into labelled (state, action, next_state) transitions for simulator
calibration.

FIELD SETUP (confirmed July 2026)
---------------------------------
  - Study TST1234, English cucumber, planting 2026-06-08
  - Treatment assignment (authoritative, per researcher, July 2026):
        P1 = T2 (IQL agent plot)
        P2 = T0 (farmer's traditional schedule)
        P3 = T1 (fixed 12-min alternate-day control)
    WARNING: the node firmware's `treatment` CSV column does NOT match this
    (it tags P1=0, P2=1, P3=1). Use PLOT_TREATMENT below, never the raw
    column, for analysis. The raw column is kept as `treatment_fw` for
    provenance.
    NOTE: through late June ALL THREE plots received the SAME irrigation,
    so that window is three replicates of one regime, not a contrast.
  - Sensor depth: 20 cm only (single layer)
  - Sample cadence: 5 minutes (300 s)
  - Drip: 1 L/h per emitter, emitters every 30 cm, ~1 emitter per plant
    => irrigation minutes map to litres-per-plant at 1 L/h.
  - All timestamps are epoch seconds (UTC); analysis is done in IST
    (Asia/Kolkata). `dt` in the returned frame is tz-aware IST.

EVENT LOG (per plant, all plots same)
-------------------------------------
  Irrigation:
    2026-06-10  30 min  -> 0.50 L/plant
    2026-06-13  15 min  -> 0.25 L/plant
  Rain (uncontrolled input; transitions flagged, excluded from drying fit):
    2026-06-08  1.5 h
    2026-06-15  1.0 h

Column decoding (raw CSV -> physical):
  temp_c      = temp_c_x10 / 10
  moist_pct   = moist_pct_x10 / 10     (already calibration-corrected: calib==raw)
  dt          = pandas.to_datetime(epoch_unix, unit='s')
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


DRIP_RATE_L_PER_H = 1.0   # per emitter ~ per plant

TZ = "Asia/Kolkata"       # all field operations and event clock times are IST

# Authoritative plot -> treatment mapping (July 2026). The firmware CSV
# `treatment` column is WRONG for P1/P2 — do not use it for analysis.
PLOT_TREATMENT = {"P1": "T2", "P2": "T0", "P3": "T1"}

# Event log — single source of truth. Times are local IST; if a specific
# clock time is unknown we assume the morning window (08:30) for irrigation.
IRRIGATION_EVENTS = [
    # (date, clock_time, minutes)
    ("2026-06-10", "08:30", 30),   # 0.50 L/plant
    ("2026-06-13", "08:30", 15),   # 0.25 L/plant
]
RAIN_EVENTS = [
    # (date, clock_time, hours)
    ("2026-06-08", "08:30", 1.5),
    ("2026-06-15", "08:30", 1.0),
]
RAIN_WINDOW_H = 3.0   # flag transitions within +/- this many hours of a rain event


@dataclass
class FieldTransition:
    plot_id: str
    dt: pd.Timestamp
    M: float          # moisture % at 20 cm
    T: float          # soil temp C
    action_L: float   # litres/plant applied between this row and the next
    M_next: float
    T_next: float
    rain: bool        # True if a rain event overlaps this transition


def load_field_csvs(paths: List[str | Path]) -> pd.DataFrame:
    """Read and concatenate the TST CSVs into a tidy per-(dt, plot) frame.

    `dt` is tz-aware IST (epoch seconds are UTC; the previous naive
    conversion silently reported times 5 h 30 min early).
    `treatment` is the corrected label from PLOT_TREATMENT; the raw firmware
    column is preserved as `treatment_fw`.
    """
    frames = []
    for p in paths:
        df = pd.read_csv(p, comment="#")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["dt"] = (pd.to_datetime(df["epoch_unix"], unit="s", utc=True)
                  .dt.tz_convert(TZ))
    df["plot_id"] = df["plot_id"].str.upper()
    df["M"] = df["moist_pct_x10"] / 10.0
    df["T"] = df["temp_c_x10"] / 10.0
    df["treatment_fw"] = df["treatment"]                 # raw firmware tag
    df["treatment"] = df["plot_id"].map(PLOT_TREATMENT)  # corrected label
    df = (df.drop_duplicates(subset=["epoch_unix", "plot_id"])
            .sort_values("dt")
            .reset_index(drop=True))
    return df[["dt", "plot_id", "M", "T", "treatment", "treatment_fw",
               "depth_cm"]]


def _irrigation_litres_at(ts: pd.Timestamp) -> float:
    """Return litres/plant if an irrigation event falls in the hour before ts."""
    for date, clock, minutes in IRRIGATION_EVENTS:
        ev = pd.Timestamp(f"{date} {clock}", tz=TZ)
        # attribute the event to the first sensor row within 1 h after it
        if ev <= ts < ev + pd.Timedelta(hours=1):
            return DRIP_RATE_L_PER_H * minutes / 60.0
    return 0.0


def _is_rain(ts: pd.Timestamp) -> bool:
    for date, clock, _hours in RAIN_EVENTS:
        ev = pd.Timestamp(f"{date} {clock}", tz=TZ)
        if abs((ts - ev).total_seconds()) <= RAIN_WINDOW_H * 3600:
            return True
    return False


def build_field_transitions(df: pd.DataFrame,
                            resample_minutes: int = 60) -> List[FieldTransition]:
    """
    Resample each plot to a fixed cadence and build consecutive transitions.
    Resampling to hourly (default) reduces 5-min sensor noise and matches the
    simulator's hourly step.
    """
    out: List[FieldTransition] = []
    for plot, g in df.groupby("plot_id"):
        s = (g.set_index("dt")[["M", "T"]]
               .resample(f"{resample_minutes}min").mean().dropna())
        idx = s.index
        for i in range(len(s) - 1):
            ts, ts_next = idx[i], idx[i + 1]
            gap_h = (ts_next - ts).total_seconds() / 3600.0
            if gap_h > resample_minutes / 60.0 * 1.5:
                continue   # data gap; skip
            out.append(FieldTransition(
                plot_id=plot, dt=ts,
                M=float(s["M"].iloc[i]), T=float(s["T"].iloc[i]),
                action_L=_irrigation_litres_at(ts_next),
                M_next=float(s["M"].iloc[i + 1]), T_next=float(s["T"].iloc[i + 1]),
                rain=_is_rain(ts) or _is_rain(ts_next),
            ))
    return out


def field_drying_rate(trans: List[FieldTransition]) -> float:
    """Mean drying rate (%/h) under no irrigation, no rain. Positive = drying."""
    rates = []
    for t in trans:
        if t.action_L > 0 or t.rain:
            continue
        dM = t.M_next - t.M
        if dM < 0:
            rates.append(-dM)
    return float(np.mean(rates)) if rates else float("nan")


def summarise_field(paths: List[str | Path]) -> Dict:
    df = load_field_csvs(paths)
    trans = build_field_transitions(df)
    irr = [t for t in trans if t.action_L > 0]
    rain = [t for t in trans if t.rain]
    dry = field_drying_rate(trans)

    print("=" * 64)
    print("Field sensor log (TST1234) — summary")
    print("=" * 64)
    print(f"Span: {df.dt.min()} -> {df.dt.max()} "
          f"({(df.dt.max()-df.dt.min()).days} days)")
    print(f"Plots: {sorted(df.plot_id.unique())}")
    print(f"Depth: {sorted(df.depth_cm.unique())} cm (single layer)")
    print(f"Hourly transitions: {len(trans)}")
    print(f"  irrigation-labelled: {len(irr)}  (events: "
          f"{[round(t.action_L,2) for t in irr]})")
    print(f"  rain-flagged: {len(rain)}")
    print(f"  clean drying transitions: "
          f"{sum(1 for t in trans if t.action_L==0 and not t.rain and t.M_next<t.M)}")
    print(f"Field drying rate (20cm): {dry:.3f} %/h")
    if not np.isnan(dry) and dry > 0:
        print(f"Implied tau_drain (field): {70.0/dry:.1f} h")
    print("=" * 64)
    return {"df": df, "transitions": trans, "drying_rate": dry}


if __name__ == "__main__":
    import sys
    paths = sys.argv[1:] or ["./data/TST1234_001.csv", "./data/TST1234_002.csv"]
    summarise_field(paths)

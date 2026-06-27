"""
data_loader.py
==============
Parse the Fyllo Excel export into a clean per-plot hourly DataFrame
and (s, s') transition tuples for simulator calibration.

The Fyllo file (Feb 28 – Apr 27 2026) has 5 sheets:
  - summary: 4 shade-treatment columns merged
  - plot1:   the rich sheet (only Soil T, Moisture 1/2, Stage are populated)
  - plot2/3, Plot 4: minimal (Soil T, Moisture 1/2)

This module returns a consistent schema:
    [dt, plot_id, T_soil, M_surface, M_deep, stage]

stage is filled by forward-fill from plot1 (only plot with labels).
Other columns (air T/H, VPD, ET, light) are absent from the export
and so are returned as NaN — the simulator must handle their absence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# Sensor channel mapping per sheet
_PLOT_SHEETS = {
    "T0_plot1": "plot1",      # rich plot, has stage labels
    "T1_plot2": "plot2",
    "T2_plot3": "plot3",
    "T3_plot4": "Plot 4",
}


def _parse_fyllo_date(s: str) -> pd.Timestamp:
    """Fyllo writes dates as 'February 28th 2026, 6:30 PM'."""
    if not isinstance(s, str):
        return pd.NaT
    clean = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", s)
    return pd.to_datetime(clean, format="%B %d %Y, %I:%M %p", errors="coerce")


@dataclass
class PlotTimeseries:
    """Single plot, hourly cadence."""
    plot_id: str
    df: pd.DataFrame   # cols: dt, T_soil, M_surface, M_deep, stage

    @property
    def n(self) -> int:
        return len(self.df)


def load_fyllo_excel(path: str | Path) -> Dict[str, PlotTimeseries]:
    """Return {plot_id: PlotTimeseries} for the four plots."""
    path = Path(path)
    out: Dict[str, PlotTimeseries] = {}

    # plot1 carries the stage labels; load it first
    p1_raw = pd.read_excel(path, sheet_name="plot1")
    p1_raw["dt"] = p1_raw["Date"].apply(_parse_fyllo_date)
    stage_series = p1_raw.set_index("dt")["Stage"]

    for plot_id, sheet in _PLOT_SHEETS.items():
        raw = pd.read_excel(path, sheet_name=sheet)
        raw["dt"] = raw["Date"].apply(_parse_fyllo_date)
        df = pd.DataFrame({
            "dt": raw["dt"],
            "T_soil": pd.to_numeric(raw["Soil Temperature(℃)"], errors="coerce"),
            "M_surface": pd.to_numeric(raw["Moisture 1(%)"], errors="coerce"),
            "M_deep": pd.to_numeric(raw["Moisture 2(%)"], errors="coerce"),
        })
        df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)

        # Attach stage by nearest-time merge (plot1's stage broadcast to all plots)
        df = pd.merge_asof(
            df, stage_series.reset_index().rename(columns={"Stage": "stage"}),
            on="dt", direction="nearest", tolerance=pd.Timedelta("2h"),
        )
        df["stage"] = df["stage"].ffill().bfill()
        out[plot_id] = PlotTimeseries(plot_id=plot_id, df=df)

    return out


def build_observed_transitions(
    plots: Dict[str, PlotTimeseries],
) -> pd.DataFrame:
    """
    Build (s, s') transitions for simulator calibration.

    A transition is one hourly step. Includes columns:
      plot_id, dt, T_soil, M_surface, M_deep,
      M_surface_next, M_deep_next, T_soil_next, dM_surface, dM_deep, stage

    Note: actions are NOT in the dataset; this is for fitting the
    *uncontrolled* dynamics (drying rate as a function of state).
    """
    frames: List[pd.DataFrame] = []
    for pid, ts in plots.items():
        d = ts.df.copy()
        for col in ["T_soil", "M_surface", "M_deep"]:
            d[col + "_next"] = d[col].shift(-1)
        d["dM_surface"] = d["M_surface_next"] - d["M_surface"]
        d["dM_deep"]    = d["M_deep_next"]    - d["M_deep"]
        d["dt_next"]    = d["dt"].shift(-1)
        d["dt_h"]       = (d["dt_next"] - d["dt"]).dt.total_seconds() / 3600.0
        d["plot_id"]    = pid
        # Keep only ~hourly steps and drop the last row of each plot
        d = d[(d["dt_h"] > 0.5) & (d["dt_h"] < 2.0)].copy()
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def hour_of_day_features(dt: pd.Series) -> pd.DataFrame:
    """Cyclic encoding of hour of day."""
    h = dt.dt.hour + dt.dt.minute / 60.0
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * h / 24.0),
        "hour_cos": np.cos(2 * np.pi * h / 24.0),
    })


def stage_to_index(stage: str) -> int:
    """Map Fyllo stage label to our 4-stage FAO-56 schema."""
    s = (stage or "").strip().lower()
    if "veget" in s:
        return 0  # initial
    if "flower" in s:
        return 2  # mid (flowering = sensitive)
    if "harvest" in s or "fruit" in s:
        return 3  # late / harvest
    return 1      # development (default fallback)


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else \
        "NH_solar_summy_plot_data_fyllo_March_1st_to_April_27th_2026.xlsx"
    plots = load_fyllo_excel(src)
    for pid, ts in plots.items():
        print(f"{pid}: {ts.n} rows, "
              f"T_soil={ts.df['T_soil'].mean():.1f}, "
              f"M_surf={ts.df['M_surface'].mean():.1f}, "
              f"M_deep={ts.df['M_deep'].mean():.1f}")
    trans = build_observed_transitions(plots)
    print(f"\nTransitions built: {len(trans)} rows")
    print(trans.head())

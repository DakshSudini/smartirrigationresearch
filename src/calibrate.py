"""
calibrate.py
============
Calibrate the soil-water-balance simulator's drying parameters to the
observed Fyllo timeseries.

We fit only the parameters that are not directly observable in the
field but show up as scalar shape parameters in the simulator:
  - Ke_max          (bare-soil evaporation cap)
  - tau_drain_h     (surface → deep drainage time constant)
  - Ra_MJ_m2_day    (effective extraterrestrial radiation)
  - sigma_M         (process noise on moisture)

The fit minimises L2 error between observed and simulated ΔM_surface
across hours when no irrigation occurred (so we're isolating the
drying dynamics, not the actuator).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .simulator import SimConfig, SoilWaterSim


# Threshold above which we flag a row as having an irrigation event
# (large positive ΔM_surface). Used to mask out actuated hours.
_IRRIGATION_DM_THRESHOLD = 4.0   # %/h


def _mask_no_irrigation(transitions: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where no irrigation event occurred."""
    m = transitions["dM_surface"] < _IRRIGATION_DM_THRESHOLD
    return transitions.loc[m].dropna(subset=[
        "M_surface", "M_deep", "T_soil",
        "M_surface_next", "M_deep_next", "T_soil_next", "dt"
    ]).copy()


def _simulate_one_step(M_surf: float, M_deep: float, T_soil: float,
                       hour: float, day: int,
                       cfg: SimConfig) -> Dict[str, float]:
    """Single deterministic step (no noise) with zero irrigation."""
    sim = SoilWaterSim(cfg, rng=np.random.default_rng(0))
    sim.reset(day0=day, M_surf0=M_surf, M_deep0=M_deep, T_soil0=T_soil)
    sim.state.hour = hour
    # Suppress noise for the fit
    sim.cfg.sigma_M = 0.0
    sim.cfg.sigma_T = 0.0
    s, info = sim.step(irrigation_mm=0.0)
    return {"M_surf_next": s.M_surf, "M_deep_next": s.M_deep}


def _loss(theta: np.ndarray, transitions: pd.DataFrame,
          base_cfg: SimConfig) -> float:
    """
    Vector theta = [log(Ke_max), log(tau_drain_h), log(Ra)] (log-scale
    keeps them positive). Returns mean squared error on ΔM_surface
    + ΔM_deep across the de-irrigated dataset.
    """
    Ke   = float(np.exp(theta[0]))
    tau  = float(np.exp(theta[1]))
    Ra   = float(np.exp(theta[2]))

    cfg = SimConfig(**{**asdict(base_cfg), "Ke_max": Ke,
                       "tau_drain_h": tau, "Ra_MJ_m2_day": Ra})

    # Vectorised eval: simulate each row independently
    df = transitions.sort_values("dt").reset_index(drop=True)
    sse = 0.0
    n = 0
    t0 = df["dt"].iloc[0]
    for r in df.itertuples(index=False):
        hour = r.dt.hour + r.dt.minute / 60.0
        day  = (r.dt - t0).days
        pred = _simulate_one_step(
            r.M_surface, r.M_deep, r.T_soil, hour, day, cfg,
        )
        sse += (pred["M_surf_next"] - r.M_surface_next) ** 2 \
             + (pred["M_deep_next"] - r.M_deep_next) ** 2
        n += 1
    return sse / max(n, 1)


def calibrate(transitions: pd.DataFrame,
              base_cfg: SimConfig,
              subsample: int = 800,
              verbose: bool = True) -> SimConfig:
    """
    Fit (Ke_max, tau_drain_h, Ra_MJ_m2_day) to the no-irrigation
    portion of `transitions`. Returns an updated SimConfig.
    """
    no_irrig = _mask_no_irrigation(transitions)
    if len(no_irrig) == 0:
        if verbose:
            print("Calibration: no rows after masking — using defaults.")
        return base_cfg
    # Random subsample to keep the fit fast (~minute) on a CPU
    n = min(subsample, len(no_irrig))
    idx = np.random.default_rng(42).choice(len(no_irrig), n, replace=False)
    sample = no_irrig.iloc[idx].reset_index(drop=True)

    theta0 = np.log(np.array([base_cfg.Ke_max,
                              base_cfg.tau_drain_h,
                              base_cfg.Ra_MJ_m2_day]))
    # Physical bounds (log-space): Ke ∈ [0.05, 0.5], tau ∈ [1, 48],
    # Ra ∈ [15, 42]: lower bound widened because the surface-effective
    # radiation term (after atmospheric attenuation) for Telangana is ~20,
    # not the 36 extraterrestrial constant. Shade reduction is applied
    # separately via shade_et0_factor.
    bounds = [
        (np.log(0.05), np.log(0.50)),
        (np.log(1.0),  np.log(48.0)),
        (np.log(15.0), np.log(42.0)),
    ]
    if verbose:
        print(f"Calibration: starting from "
              f"Ke={base_cfg.Ke_max:.3f}, tau={base_cfg.tau_drain_h:.2f}, "
              f"Ra={base_cfg.Ra_MJ_m2_day:.1f}")
        print(f"Calibration sample size: {n}")

    res = minimize(
        _loss, theta0, args=(sample, base_cfg),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 60, "disp": verbose},
    )
    theta_hat = res.x
    fitted = SimConfig(**{
        **asdict(base_cfg),
        "Ke_max":         float(np.exp(theta_hat[0])),
        "tau_drain_h":    float(np.exp(theta_hat[1])),
        "Ra_MJ_m2_day":   float(np.exp(theta_hat[2])),
    })
    if verbose:
        print(f"Calibration done: Ke={fitted.Ke_max:.3f}, "
              f"tau={fitted.tau_drain_h:.2f}, Ra={fitted.Ra_MJ_m2_day:.1f}, "
              f"final_mse={res.fun:.3f}")
    return fitted


def calibrate_from_log(log_path: str,
                        base_cfg: SimConfig,
                        verbose: bool = True) -> SimConfig:
    """
    Calibrate simulator drying parameters using the hand-recorded soil
    sensor reading log (pot experiment with known irrigation actions).

    Unlike `calibrate()` which works on Fyllo data and must *infer* which
    rows have irrigation, here the action labels are explicit. We therefore:

    1. Use ONLY net house soil pots (p2-p5) — these are the target soil for
       the experiment. Sandy soil (pot 1) has fundamentally different
       drainage physics and is kept separate.

    2. Use ONLY no-irrigation, no-rain transitions for drying-parameter
       fitting — same principle as the Fyllo calibration.

    3. Fit tau_drain_h from the net house soil drying rate directly, without
       having to absorb it into Ra (because we actually see enough drying
       contrast in this short dataset from pot 1 comparison).

    4. Because we have no air-temperature data (same gap as Fyllo), Ra is
       held at its Fyllo-calibrated value (~21 MJ m-2 d-1 — the surface-effective
       radiation term after atmospheric attenuation for Telangana at ~17°N; the
       raw extraterrestrial constant of ~36 overestimates surface ET, so the
       calibrated value sits well below it). Ke_max is also held because
       bare-soil evaporation in a closed greenhouse pot is low — the surface is
       covered by the canopy.

    Parameters
    ----------
    log_path : str
        Path to data/soil_log.csv
    base_cfg : SimConfig
        Starting config (Fyllo-calibrated values are used as priors for
        parameters not identifiable from log data alone).
    verbose : bool

    Returns
    -------
    SimConfig with tau_drain_h re-fitted to net house soil drying curves.
    sigma_M is also updated from the residuals.

    Notes
    -----
    Only 5 days of data (22-26 May 2026) are available at time of writing.
    tau_drain_h can be estimated but with wide uncertainty. The estimate will
    stabilise as more log data accumulates. The returned config should be
    treated as an informed prior, not a tight estimate.
    """
    from .log_loader import (load_log_csv, build_log_transitions,
                             split_by_soil, net_soil_drying_rate,
                             SENSOR_FC, SENSOR_WP)

    df   = load_log_csv(log_path)
    all_trans = build_log_transitions(df)
    _, net_trans = split_by_soil(all_trans)

    # --- Step 1: estimate τ_drain from mean drying rate -------------------
    drying_rate = net_soil_drying_rate(net_trans)  # %/h
    if np.isnan(drying_rate) or drying_rate <= 0:
        if verbose:
            print("Log calibration: insufficient drying data — "
                  "keeping Fyllo tau_drain estimate.")
        return base_cfg

    # TAW in sensor-% units
    taw_pct = SENSOR_FC - SENSOR_WP
    # Simple first-order drying model: dM/dt = -(FC-WP)/tau
    # → tau = (FC-WP) / drying_rate
    tau_est = taw_pct / drying_rate

    if verbose:
        print(f"Log calibration: net house soil drying rate = "
              f"{drying_rate:.2f} %/h")
        print(f"  Implied tau_drain = {tau_est:.1f} h  "
              f"(Fyllo estimate was {base_cfg.tau_drain_h:.1f} h)")

    # --- Step 2: estimate sigma_M from drying residuals -------------------
    # Simulate each no-irrigation transition and collect residuals
    residuals = []
    for t in net_trans:
        if t.action_mL > 0 or t.rain:
            continue
        if t.M >= SENSOR_FC and t.M_next >= SENSOR_FC:
            continue   # both saturated — uninformative
        if t.M <= SENSOR_WP or t.M_next <= SENSOR_WP:
            continue

        # Predicted drying over the actual time gap
        dt_hrs = 1.0   # readings are ~1 h apart (we already filtered >2h)
        predicted_drop = (taw_pct / tau_est) * dt_hrs
        predicted_M_next = t.M - predicted_drop
        residuals.append(t.M_next - predicted_M_next)

    sigma_M = float(np.std(residuals)) if len(residuals) >= 5 else base_cfg.sigma_M
    if verbose:
        print(f"  Residual σ_M = {sigma_M:.2f} %  "
              f"(n_drying_transitions = {len(residuals)})")
        print(f"  Note: tau_drain upper-bounded at 48 h (sensor resolution "
              f"limit). Current estimate {min(tau_est, 48.0):.1f} h.")

    from dataclasses import asdict
    updated = SimConfig(**{
        **asdict(base_cfg),
        "tau_drain_h": min(float(tau_est), 48.0),   # keep within Fyllo bound
        "sigma_M":     max(sigma_M, 0.5),            # floor at 0.5 — sensor noise
    })
    return updated


def calibrate_from_field(field_paths, base_cfg, verbose=True):
    """
    Refine the simulator using the REAL shade-house field sensor logs
    (TST1234_*.csv), which have known irrigation/rain events.

    The field probe is at 20 cm — the DEEP layer, not the surface. Over the
    8-day window the 20 cm moisture is in near-equilibrium (mean dM/dt ~0 %/h
    across all three plots) because seedlings are just transplanted, the shade
    house keeps demand low, and 20 cm is below the fast-drying surface. So the
    field data constrains DEEP-LAYER STABILITY, not surface tau. We slow the
    deep drainage to match, keep the pot-derived surface tau as a floor, and
    leave Ke/Ra/shade (surface ET) untouched. Conservative by design: 8 days,
    two light events, single depth.
    """
    from .field_loader import (load_field_csvs, build_field_transitions,
                              field_drying_rate)
    from dataclasses import asdict
    import numpy as np

    df = load_field_csvs(field_paths)
    trans = build_field_transitions(df)
    drifts = [t.M_next - t.M for t in trans if t.action_L == 0 and not t.rain]
    mean_abs_drift = float(np.mean(np.abs(drifts))) if drifts else float("nan")
    dry_rate = field_drying_rate(trans)
    t_mean_field = float(df["T"].mean())

    if verbose:
        print(f"Field calibration: {len(trans)} hourly transitions, 3 plots, 20 cm")
        print(f"  Mean |dM/dt| at 20 cm: {mean_abs_drift:.3f} %/h (near-zero => stable deep layer)")
        print(f"  Downward-only drying: {dry_rate:.3f} %/h => field tau ~ {70.0/max(dry_rate,1e-3):.0f} h")
        print(f"  Field soil-temp mean: {t_mean_field:.1f} C")

    field_tau = 70.0 / max(dry_rate, 1e-3)
    new_tau = float(np.clip(max(base_cfg.tau_drain_h, min(field_tau, 72.0)),
                            base_cfg.tau_drain_h, 72.0))
    if verbose:
        print(f"  tau_drain_h: {base_cfg.tau_drain_h:.1f} -> {new_tau:.1f} h (deep layer slowed to field)")

    updated = SimConfig(**{**asdict(base_cfg), "tau_drain_h": new_tau})
    return updated



if __name__ == "__main__":
    import sys
    from .data_loader import load_fyllo_excel, build_observed_transitions

    src = sys.argv[1] if len(sys.argv) > 1 else "./fyllo.xlsx"
    plots = load_fyllo_excel(src)
    trans = build_observed_transitions(plots)
    print(f"Total observed transitions: {len(trans)}")
    cfg = calibrate(trans, SimConfig(), subsample=200)
    print("Fitted:", cfg.Ke_max, cfg.tau_drain_h, cfg.Ra_MJ_m2_day)

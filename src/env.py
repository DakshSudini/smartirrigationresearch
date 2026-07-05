"""
env.py
======
Gym-like environment wrapping `SoilWaterSim` for offline RL.

DEPLOYMENT-ALIGNED CONTROL MODEL (June 2026)
--------------------------------------------
The real T2 plot has NO solenoid valve. A human reads the agent's
recommendation at two fixed windows per day — 09:00 and 14:00 — and
applies a VOLUME in litres per plant by hand. Between those windows the
agent does nothing.

The environment therefore models the day as a sequence of hourly soil-water
updates (the simulator's natural timestep), but the AGENT is only queried
for an action at the two decision hours. At all other hours the env auto-steps
the simulator with zero irrigation until it reaches the next decision window.

So one env.step() = "make the 9 AM (or 2 PM) decision, then fast-forward the
simulator to the next decision window." An episode of `total_days` therefore
has 2 * total_days agent decisions.

State observation (14-d float vector, all roughly in [-3, 3] after norm):
    [ M_surf_n, M_deep_n, T_soil_n, dM_surf_n, dM_deep_n,
      hour_sin, hour_cos, day_norm,
      stage_oh_init, stage_oh_dev, stage_oh_mid, stage_oh_late,
      cum_water_today_norm, session_flag ]
  (session_flag = 0.0 at the 9 AM decision, 1.0 at the 2 PM decision)

Action: discrete index into cfg.action_litres_per_plant.

Reward (per decision):
    r = - alpha * water_norm
        - beta  * stress_pen_over_interval
        - gamma * over_sat_pen_over_interval
        + (terminal_yield_bonus if done else 0)
Stress/oversat are accumulated over the hours fast-forwarded to the next
decision so no information between windows is lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .simulator import (SimConfig, SoilWaterSim, kc_for_stage,
                        stage_from_day)
from .obs import build_obs, OBS_DIM as _OBS_DIM


@dataclass
class EnvConfig:
    action_litres_per_plant: List[float]
    plants_per_m2: float
    surface_depth_m: float
    decision_hours: List[int]            # e.g. [9, 14]
    max_daily_water_L_per_plant: float
    alpha_water: float
    beta_stress: float
    gamma_oversat: float
    yield_terminal_weight: float
    total_days: int = 90
    optimal_band: Dict[str, list] = None  # filled from yaml


class IrrigationEnv:
    """Gym-like API: reset → obs; step(action) → obs, r, done, info."""

    OBS_DIM = _OBS_DIM

    def __init__(self, env_cfg: EnvConfig, sim_cfg: SimConfig,
                 rng: np.random.Generator = None):
        self.cfg = env_cfg
        self.sim_cfg = sim_cfg
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.sim = SoilWaterSim(sim_cfg, rng=self.rng)
        self.n_actions = len(env_cfg.action_litres_per_plant)
        self.decision_hours = sorted(env_cfg.decision_hours)

        # State tracking for derived features
        self._M_surf_prev = None
        self._M_deep_prev = None
        self._decision_idx = 0   # which decision window we're at

    # ----- litres → mm conversion -----------------------------------
    def _litres_to_mm(self, litres_per_plant: float) -> float:
        # 1 L spread over the ground area of one plant (= 1/plants_per_m2 m²)
        # gives  litres / area  mm.  litres * plants_per_m2 = mm over 1 m².
        return litres_per_plant * self.cfg.plants_per_m2

    # ----- observation ----------------------------------------------
    def _obs(self, session_flag: float) -> np.ndarray:
        """Build the observation via the canonical builder (src/obs.py) so
        training and serving can never diverge on layout/normalisation."""
        s = self.sim.state
        dM_surf = (s.M_surf - self._M_surf_prev) if self._M_surf_prev is not None else 0.0
        dM_deep = (s.M_deep - self._M_deep_prev) if self._M_deep_prev is not None else 0.0
        stage = stage_from_day(s.day, self.sim_cfg.stage_days)
        cap = self.cfg.max_daily_water_L_per_plant * self.cfg.plants_per_m2
        return build_obs(
            M_surf=s.M_surf, M_deep=s.M_deep, T_soil=s.T_soil,
            hour=s.hour, day=s.day, stage_idx=stage,
            cum_water_today_mm=s.cum_water_today_mm, daily_cap_mm=cap,
            session_flag=session_flag, dM_surf=dM_surf, dM_deep=dM_deep,
        )

    # ----- reset ----------------------------------------------------
    def reset(self, day0: int = 0,
              M_surf0: float = None,
              M_deep0: float = None,
              T_soil0: float = 23.0) -> np.ndarray:
        if M_surf0 is None:
            M_surf0 = float(self.rng.uniform(60.0, 85.0))
        if M_deep0 is None:
            M_deep0 = float(self.rng.uniform(60.0, 80.0))
        self.sim.reset(day0=day0, M_surf0=M_surf0, M_deep0=M_deep0,
                       T_soil0=T_soil0)
        # Fast-forward to the first decision hour of the day
        self._advance_to_hour(self.decision_hours[0])
        self._M_surf_prev = self.sim.state.M_surf
        self._M_deep_prev = self.sim.state.M_deep
        self._decision_idx = 0
        return self._obs(session_flag=0.0)

    # ----- helper: advance sim (no irrigation) until target hour -----
    def _advance_to_hour(self, target_hour: int) -> Dict:
        """Step the simulator with zero irrigation until state.hour == target.
        Returns accumulated stress/oversat info over the advanced hours."""
        acc = {"stress_hours": 0.0, "oversat_hours": 0.0, "n_hours": 0}
        guard = 0
        while int(round(self.sim.state.hour)) != int(target_hour) and guard < 48:
            s, info = self.sim.step(0.0)
            acc["stress_hours"] += self._stress_indicator(s, info)
            acc["oversat_hours"] += 1.0 if s.M_surf > 95.0 else 0.0
            acc["n_hours"] += 1
            guard += 1
        return acc

    def _stress_indicator(self, s, info) -> float:
        stage = info["stage"]
        stage_name = ["initial", "development", "mid", "late"][stage]
        lo, hi = self.cfg.optimal_band[stage_name]
        return 1.0 if info["M_root"] < lo else 0.0

    # ----- step (one decision) --------------------------------------
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        litres = self.cfg.action_litres_per_plant[int(action_idx)]
        irrigation_mm = self._litres_to_mm(litres)

        # Daily cap enforcement
        cap_mm = self.cfg.max_daily_water_L_per_plant * self.cfg.plants_per_m2
        if self.sim.state.cum_water_today_mm + irrigation_mm > cap_mm:
            irrigation_mm = max(0.0, cap_mm - self.sim.state.cum_water_today_mm)
            litres = irrigation_mm / max(self.cfg.plants_per_m2, 1e-6)

        self._M_surf_prev = self.sim.state.M_surf
        self._M_deep_prev = self.sim.state.M_deep

        # Apply the irrigation at the decision hour (single hourly step WITH water)
        s, info = self.sim.step(irrigation_mm)
        stress_acc = self._stress_indicator(s, info)
        oversat_acc = 1.0 if s.M_surf > 95.0 else 0.0
        n_hours = 1

        # Determine the next decision window and fast-forward to it
        self._decision_idx = (self._decision_idx + 1) % len(self.decision_hours)
        next_hour = self.decision_hours[self._decision_idx]
        fwd = self._advance_to_hour(next_hour)
        stress_acc += fwd["stress_hours"]
        oversat_acc += fwd["oversat_hours"]
        n_hours += fwd["n_hours"]

        # ----- reward (accumulated over the interval just simulated) -----
        max_litres = max(self.cfg.action_litres_per_plant)
        max_mm = self._litres_to_mm(max_litres)
        water_norm = irrigation_mm / max(max_mm, 1e-6)
        # normalise stress/oversat by hours so reward scale is per-decision stable
        stress_pen = stress_acc / max(n_hours, 1)
        over_pen = oversat_acc / max(n_hours, 1)

        stage = stage_from_day(self.sim.state.day, self.sim_cfg.stage_days)
        if stage == 2:                       # mid (flowering/fruiting) most sensitive
            stress_pen *= 1.5

        r = -(self.cfg.alpha_water * water_norm
              + self.cfg.beta_stress * stress_pen
              + self.cfg.gamma_oversat * over_pen)

        done = self.sim.state.day >= self.cfg.total_days
        if done:
            r += self.cfg.yield_terminal_weight * self.sim.state.cum_yield_proxy / 50.0

        session_flag = float(self._decision_idx)  # 0 at first window, 1 at second
        info.update({
            "litres_per_plant": litres,
            "irrigation_mm": irrigation_mm,
            "water_norm": water_norm,
            "stress_pen": stress_pen,
            "over_pen": over_pen,
            "interval_hours": n_hours,
            "reward_components": {
                "water": -self.cfg.alpha_water * water_norm,
                "stress": -self.cfg.beta_stress * stress_pen,
                "oversat": -self.cfg.gamma_oversat * over_pen,
            },
        })
        return self._obs(session_flag), float(r), done, info

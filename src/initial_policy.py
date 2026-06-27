"""
initial_policy.py
=================
Deterministic FAO-56 MAD initial policy for English cucumber, used inside
the SIMULATOR as (a) the day-1 behavioural baseline and (b) one of the
behaviour policies that generate the offline RL buffer.

This is the simulator-side twin of `initial_policy_deploy.py` (which is the
field-facing version with the 9 AM / 2 PM card). Both share the same FAO-56
logic; this one is expressed in the simulator's sensor-% state and returns a
VOLUME in litres per plant (matching the new litres-based action space).

Decision timing (which hours to irrigate) is handled by the environment — it
only queries the policy at the configured decision windows — so this policy
does not gate on hour. It decides only HOW MUCH to apply given the soil state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .simulator import SimConfig, stage_from_day


@dataclass
class InitialPolicyConfig:
    optimal_band: Dict[str, list]
    plants_per_m2: float
    surface_depth_m: float
    deep_depth_m: float
    daily_cap_L_per_plant: float
    MAD: float
    et0_shade_mm_day: float = 4.8        # shade-house ET0 (matches deploy policy)
    drip_efficiency: float = 0.90
    decisions_per_day: int = 2           # 9 AM + 2 PM
    max_session_L: float = 2.0


class InitialPolicy:
    """FAO-56 MAD initial controller (simulator-side, litres output)."""

    def __init__(self, cfg: InitialPolicyConfig, sim_cfg: SimConfig):
        self.cfg = cfg
        self.sim_cfg = sim_cfg
        self.area_per_plant = 1.0 / max(cfg.plants_per_m2, 1e-6)

    # ----- helpers ---------------------------------------------------
    def _band_for_stage(self, stage: int) -> tuple:
        name = ["initial", "development", "mid", "late"][stage]
        return tuple(self.cfg.optimal_band[name])

    def _kc_for_stage(self, stage: int) -> float:
        kc = self.sim_cfg.Kc
        if stage == 0:
            return kc["initial"]
        if stage == 1:
            return 0.5 * (kc["initial"] + kc["mid"])
        if stage == 2:
            return kc["mid"]
        return kc["late"]

    def _root_zone_moisture(self, M_surf: float, M_deep: float, day: int) -> float:
        root_depth = min(
            self.sim_cfg.root_depth_init_m
            + (self.sim_cfg.root_depth_max_m - self.sim_cfg.root_depth_init_m)
            * (max(day, 0) / 60.0),
            self.sim_cfg.root_depth_max_m,
        )
        frac_surf = self.sim_cfg.surface_depth_m / max(root_depth, 1e-3)
        frac_surf = float(np.clip(frac_surf, 0.2, 1.0))
        return frac_surf * M_surf + (1 - frac_surf) * M_deep

    # ----- main API --------------------------------------------------
    def recommend_litres(self, *, M_surf: float, M_deep: float, day: int) -> float:
        """Return litres-per-plant to apply this session given soil state.

        Mirrors the deploy policy:
          - if root-zone M >= upper band → 0 (already wet)
          - base = half daily ETc replacement
          - if below MAD trigger → add deficit top-up
          - cap at max_session_L
        """
        cfg = self.cfg
        stage = stage_from_day(day, self.sim_cfg.stage_days)
        lo, hi = self._band_for_stage(stage)
        kc = self._kc_for_stage(stage)

        M_root = self._root_zone_moisture(M_surf, M_deep, day)

        # Already wet enough → skip
        if M_root >= hi:
            return 0.0

        # Base ETc replacement, split across the day's decision windows
        etc_mm_day = kc * cfg.et0_shade_mm_day
        base_L = (etc_mm_day * self.area_per_plant / cfg.drip_efficiency) \
            / max(cfg.decisions_per_day, 1)

        # Deficit top-up if below the MAD trigger
        TAW = self.sim_cfg.FC - self.sim_cfg.WP
        trigger = self.sim_cfg.FC - cfg.MAD * TAW
        topup_L = 0.0
        if M_root < trigger:
            frac = min((trigger - M_root) / max(trigger - self.sim_cfg.WP, 1.0), 1.0)
            topup_L = frac * base_L

        vol = min(base_L + topup_L, cfg.max_session_L)
        return float(vol)

    # Back-compat alias: some callers used .act() returning an amount.
    def act(self, *, M_surf: float, M_deep: float, day: int, **kwargs) -> float:
        return self.recommend_litres(M_surf=M_surf, M_deep=M_deep, day=day)

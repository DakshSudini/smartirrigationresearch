"""
simulator.py
============
Two-layer FAO-56-style soil water balance + cucumber crop model.

References:
  - Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998).
    "Crop Evapotranspiration — Guidelines for computing crop water
    requirements." FAO Irrigation and Drainage Paper 56.
  - Castilla, N. (2013). "Greenhouse Technology and Management" (2nd ed.).

State (continuous, what the policy sees):
  M_surf [%], M_deep [%], T_soil [°C], stage [0..3],
  hour-of-day (cyclic), days since transplant, cum water today,
  hours since last irrigation.

Internal hidden variables (the simulator tracks but they're not
necessarily observed in the field):
  ETc_t  (mm/h)  — crop evapotranspiration this step
  D_t    (mm)    — root-zone depletion

Sensor scale: Fyllo "Moisture %" is normalised to saturation (i.e.
0 ≡ wilting point, 100 ≡ saturation in the sensor's calibration).
We work natively on that scale; FC and WP are configured as %
of the sensor scale, not in volumetric m³/m³.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------- #
# Crop stage helpers
# --------------------------------------------------------------------- #
STAGE_NAMES = ["initial", "development", "mid", "late"]


def stage_from_day(day: int, stage_days: Dict[str, int]) -> int:
    """Return stage index 0..3 from day-since-transplant."""
    s_init = stage_days["initial"]
    s_dev  = s_init + stage_days["development"]
    s_mid  = s_dev  + stage_days["mid"]
    if day < s_init:  return 0
    if day < s_dev:   return 1
    if day < s_mid:   return 2
    return 3


def kc_for_stage(stage: int, kc: Dict[str, float]) -> float:
    """Linear interpolation across stages, as per FAO-56."""
    if stage == 0: return kc["initial"]
    if stage == 1:                  # development: linear init -> mid
        return 0.5 * (kc["initial"] + kc["mid"])
    if stage == 2: return kc["mid"]
    return kc["late"]


def stage_to_name(stage: int) -> str:
    return STAGE_NAMES[stage]


# --------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------- #
@dataclass
class SimConfig:
    """All physical parameters of the soil-water balance."""
    # Soil (sensor-scale %)
    FC: float = 80.0
    WP: float = 25.0
    SAT: float = 100.0
    surface_depth_m: float = 0.10
    deep_depth_m: float = 0.30
    tau_drain_h: float = 6.0
    Ke_max: float = 0.20            # bare-soil evaporation coefficient
    # Crop (FAO-56)
    stage_days: Dict[str, int] = field(default_factory=lambda: {
        "initial": 12, "development": 18, "mid": 35, "late": 25,
    })
    Kc: Dict[str, float] = field(default_factory=lambda: {
        "initial": 0.60, "mid": 1.00, "late": 0.75,
    })
    plants_per_m2: float = 2.5
    root_depth_init_m: float = 0.10
    root_depth_max_m: float = 0.40
    # Reference ET model (Hargreaves, since we only have T_soil)
    # ET0 = 0.0023 * Ra * (T_mean + 17.8) * sqrt(Tmax - Tmin)
    # Ra (extraterrestrial radiation) is fixed for Telangana ~17°N.
    # Effective solar radiation term for Hargreaves at 17°N. Tuned so the
    # OPEN-FIELD ET0 lands at the realistic Telangana summer ~6 mm/day
    # (the extraterrestrial constant ~36 overestimates surface ET because it
    # ignores atmospheric attenuation; 20-22 reproduces measured pan/PM ET0).
    Ra_MJ_m2_day: float = 21.0
    # Shade-house reduction factor applied to open-field ET0.
    # Confirmed setup: 25% shade net + closed side walls, Telangana.
    #   25% net -> radiation-driven ET reduction ~0.85
    #   closed walls -> reduced ventilation ~0.95  =>  combined ~0.80
    # VALIDATE with in-house temp/RH sensors once logging.
    shade_et0_factor: float = 0.80
    # Daily ET0 cap (mm/day). Raised from 8.0 so the Hargreaves value is NOT
    # pinned at the ceiling for normal Telangana temps (raw open-field
    # ~11-13 mm/day). After the shade factor this lands at a realistic
    # ~4-5 mm/day inside the shade house.
    ET0_max_mm_day: float = 14.0
    # Stochastic noise on dynamics
    sigma_M: float = 0.5
    sigma_T: float = 0.3


@dataclass
class SimState:
    """Internal simulator state (deterministic part)."""
    M_surf: float
    M_deep: float
    T_soil: float
    day: int
    hour: float
    cum_water_today_mm: float
    hours_since_irrig: float
    # Bookkeeping
    cum_yield_proxy: float = 0.0
    cum_water_total_mm: float = 0.0
    stress_hours_in_sensitive: float = 0.0


class SoilWaterSim:
    """
    Two-layer soil-water balance, hourly step.

    Dynamics (per hour):
      ET_h          = (Kc * ET0_daily / 24) * f_canopy * Ks(depletion)
      ET_surf       = ET_h * frac_surface_root
      ET_deep       = ET_h * (1 - frac_surface_root)
      drain_to_deep = (M_surf - FC)+ / tau_drain  if M_surf > FC else 0
      M_surf_next   = M_surf + I_h - ET_surf - drain_to_deep
      M_deep_next   = M_deep + drain_to_deep - ET_deep - deep_drain
      T_soil_next   = T_soil + diurnal_term(hour, day)

    Yield proxy: at each hour, dY = Ks * Kc * 1.0 (a unitless growth rate).
    Final yield is cum_yield_proxy; this is the terminal reward driver.
    """

    def __init__(self, cfg: SimConfig, rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.state: Optional[SimState] = None
        # Internal cache for daily ET0
        self._daily_et0: Dict[int, float] = {}

    # Sum of the diurnal half-sine weights over a 24-h day. Used to normalise
    # the hourly ET0 distribution so it integrates to et0_day exactly.
    _DIURNAL_WEIGHT_SUM = float(sum(
        max(np.sin(np.pi * (h - 5) / 14.0), 0.0) for h in range(24)
    ))

    # ----- ET0 (Hargreaves; coarse, since only T_soil is observed) ---
    def _et0_mm_day(self, day: int, T_mean: float, T_range: float = 8.0) -> float:
        if day in self._daily_et0:
            return self._daily_et0[day]
        # T_mean comes from the simulated T_soil (proxy); T_range is
        # the typical greenhouse diurnal swing in Hyderabad spring.
        T_max = T_mean + T_range / 2.0
        T_min = T_mean - T_range / 2.0
        et0_open = 0.0023 * self.cfg.Ra_MJ_m2_day * (T_mean + 17.8) * np.sqrt(
            max(T_max - T_min, 0.1)
        )
        et0_open = float(np.clip(et0_open, 0.5, self.cfg.ET0_max_mm_day))
        # Apply shade-house reduction (25% net + closed walls).
        et0 = et0_open * self.cfg.shade_et0_factor
        # MJ/m²/day -> mm/day conversion already built into Hargreaves constant.
        self._daily_et0[day] = et0
        return et0

    # ----- Stress coefficient Ks (FAO-56 eq. 84) ---------------------
    def _Ks(self, M_root: float, MAD: float = 0.35) -> float:
        """Stress reduction; M_root in sensor %."""
        TAW = self.cfg.FC - self.cfg.WP
        RAW = MAD * TAW                       # readily available water
        D   = max(self.cfg.FC - M_root, 0.0)  # depletion below FC
        if D <= RAW:
            return 1.0
        if D >= TAW:
            return 0.0
        return max((TAW - D) / max(TAW - RAW, 1e-6), 0.0)

    # ----- Diurnal soil-temperature surrogate -----------------------
    def _T_soil_next(self, T_now: float, hour: float, day: int) -> float:
        """
        Simple sinusoidal model fitted on observed plot1 data:
            T(h) = T_mean + amp * sin(2π (h - 5)/24)
        T_mean drifts slowly with day.
        """
        T_mean = 25.0 + 0.05 * day            # mild seasonal drift
        amp    = 4.0                          # observed greenhouse swing
        target = T_mean + amp * np.sin(2 * np.pi * (hour - 5) / 24.0)
        # Lag toward target (thermal mass)
        return T_now + 0.4 * (target - T_now) + \
            self.rng.normal(0.0, self.cfg.sigma_T)

    # ----- Reset ----------------------------------------------------
    def reset(self, day0: int = 0, M_surf0: float = 78.0,
              M_deep0: float = 70.0, T_soil0: float = 23.0) -> SimState:
        self.state = SimState(
            M_surf=M_surf0, M_deep=M_deep0, T_soil=T_soil0,
            day=day0, hour=6.0,
            cum_water_today_mm=0.0, hours_since_irrig=24.0,
        )
        self._daily_et0.clear()
        return self.state

    # ----- One hourly step ------------------------------------------
    def step(self, irrigation_mm: float) -> Tuple[SimState, Dict]:
        """
        Apply an irrigation pulse (mm equivalent over surface layer) and
        advance the simulation one hour.

        Returns next state and an info dict (ET, Ks, yield_increment).
        """
        s = self.state
        cfg = self.cfg
        if s is None:
            raise RuntimeError("Call reset() first")

        stage = stage_from_day(s.day, cfg.stage_days)
        Kc    = kc_for_stage(stage, cfg.Kc)

        # Reference ET (daily mean) → per hour, distributed over daylight.
        et0_day = self._et0_mm_day(s.day, T_mean=s.T_soil)
        # Diurnal distribution: half-sine peaking ~noon, zero at night.
        # Normalise by the ACTUAL 24-h sum of the weights so the hourly values
        # integrate back to et0_day exactly (previously two hand-tuned constants,
        # 1/0.637 and 24/14, cancelled to ~0.5% only by coincidence; this is
        # exact and self-documenting).
        diurnal_w = max(np.sin(np.pi * (s.hour - 5) / 14.0), 0.0)
        et0_h = et0_day * diurnal_w / self._DIURNAL_WEIGHT_SUM
        # Crop ET (potential)
        etc_h = Kc * et0_h
        # Stress reduction based on the weighted root zone moisture
        # Root depth fraction in surface layer decreases with day
        root_depth = min(
            cfg.root_depth_init_m + (cfg.root_depth_max_m - cfg.root_depth_init_m)
            * (max(s.day, 0) / 60.0), cfg.root_depth_max_m,
        )
        root_depth = max(root_depth, cfg.root_depth_init_m)
        frac_surf = cfg.surface_depth_m / root_depth
        frac_surf = float(np.clip(frac_surf, 0.2, 1.0))
        frac_deep = 1.0 - frac_surf
        M_root = frac_surf * s.M_surf + frac_deep * s.M_deep
        Ks = self._Ks(M_root)
        et_actual = Ks * etc_h                  # actual ET this hour

        # Convert mm to sensor-% change. The sensor reads 0..100 between
        # WP and SAT. 1 mm of water over the layer depth (m) lifts the
        # sensor reading by approximately:
        #   ΔM% ≈ (1 mm / layer_depth_m) / (SAT - WP) * 100 * 100
        # We absorb the constants into a calibrated scale factor.
        depth_factor_surf = 100.0 / (cfg.surface_depth_m * 1000.0)
        depth_factor_deep = 100.0 / (cfg.deep_depth_m    * 1000.0)

        # Apply irrigation to surface
        I_pct_surf = irrigation_mm * depth_factor_surf

        # Drainage from surface to deep (only above FC)
        excess = max(s.M_surf + I_pct_surf - cfg.FC, 0.0)
        drain_to_deep_pct = excess * (1.0 - np.exp(-1.0 / cfg.tau_drain_h))

        # ET in sensor-%
        et_surf_pct = (frac_surf * et_actual) * depth_factor_surf
        et_deep_pct = (frac_deep * et_actual) * depth_factor_deep

        # Update moistures
        M_surf_new = s.M_surf + I_pct_surf - et_surf_pct - drain_to_deep_pct
        # Deep gets drainage; loses to ET; deep-drainage (slow) if above FC
        deep_drain = max(s.M_deep - cfg.FC, 0.0) * 0.01
        M_deep_new = s.M_deep + drain_to_deep_pct * (cfg.surface_depth_m
                                                     / cfg.deep_depth_m) \
                     - et_deep_pct - deep_drain
        # Clip to physical range and add noise
        M_surf_new = float(np.clip(
            M_surf_new + self.rng.normal(0, cfg.sigma_M), 0.0, 100.0))
        M_deep_new = float(np.clip(
            M_deep_new + self.rng.normal(0, cfg.sigma_M), 0.0, 100.0))

        # Time advance
        hour_new = (s.hour + 1.0) % 24.0
        day_new  = s.day + (1 if hour_new < s.hour else 0)
        T_new    = self._T_soil_next(s.T_soil, hour_new, day_new)

        # Yield proxy: growth rate proportional to Ks * Kc * daylight factor
        daylight = 1.0 if 6 <= int(s.hour) < 18 else 0.0
        dY = Ks * Kc * daylight / 24.0
        s.cum_yield_proxy += dY
        s.cum_water_total_mm += irrigation_mm

        # Cumulative water reset at midnight
        if hour_new < s.hour:
            s.cum_water_today_mm = 0.0
        s.cum_water_today_mm += irrigation_mm

        # Sensitivity tracking
        is_sensitive = stage in (2,)              # mid (flowering+fruiting)
        if is_sensitive and Ks < 0.7:
            s.stress_hours_in_sensitive += 1.0

        # Last-irrigation timer
        if irrigation_mm > 0.01:
            s.hours_since_irrig = 0.0
        else:
            s.hours_since_irrig = min(s.hours_since_irrig + 1.0, 48.0)

        s.M_surf = M_surf_new
        s.M_deep = M_deep_new
        s.T_soil = T_new
        s.day    = day_new
        s.hour   = hour_new

        info = {
            "ET_actual_mm_h": float(et_actual),
            "ET0_day_mm": float(et0_day),
            "Ks": float(Ks),
            "Kc": float(Kc),
            "stage": stage,
            "yield_increment": float(dY),
            "M_root": float(M_root),
        }
        return s, info

"""
initial_policy_deploy.py
========================
Deployment-ready FAO-56 initial irrigation policy for the T2 plot, adapted
to the ACTUAL field setup (confirmed June 2026):

  - Two decision windows per day: 09:00 and 14:00 (no solenoid yet; a human
    reads the recommendation and waters by hand)
  - Output is LITRES PER PLANT to apply now (not pulse minutes)
  - No pre-irrigation reading: the policy reads whatever the most recent
    15-minute sensor sample shows at decision time
  - Shade house: 25% shade net, closed side walls, Telangana
  - Net house soil (same as the pot experiment): FC≈95%, WP≈25% (sensor scale)

This is a transparent, agronomist-auditable rule meant to run from day one of
the trial and to GENERATE action-labelled data. Every recommendation it makes
is a real, logged (soil state → volume applied → next soil state) tuple, which
is exactly what the IQL agent needs for its next training cycle.

------------------------------------------------------------------------------
DECISION LOGIC (per session, at 09:00 and 14:00)
------------------------------------------------------------------------------
1. Read current root-zone soil moisture M (%) from the latest sensor sample.
   (If two depths are available, root-zone M = 0.6*surface + 0.4*deep.
    If single depth, use it directly.)

2. Determine the crop stage from days-after-transplant, which sets:
     - Kc (crop coefficient)
     - the target moisture band [lo, hi] for that stage

3. Decide volume:
     a. If M >= hi  (soil already at/above the upper band):
          → apply 0 L  (skip; soil is wet enough, avoid waterlogging)
     b. If M <  hi:
          → base volume = half the daily ETc requirement for the stage
            (because there are two sessions a day)
          → if M < MAD_trigger (soil is genuinely dry), add a deficit
            top-up proportional to how far below the lo-band it is
          → cap the result at max_session_L for safety

4. Round to the nearest 0.25 L (practical for hand-watering with a measuring
   jug / calibrated container).

------------------------------------------------------------------------------
PARAMETERS YOU SHOULD VALIDATE
------------------------------------------------------------------------------
  - ET0_shade_mm_day: currently 4.8 mm/day, a literature estimate for a 25%
    shade net + closed walls in Telangana summer. ONCE THE IN-HOUSE TEMP/RH
    SENSORS ARE LOGGING, replace this with a Hargreaves estimate computed from
    the measured shade-house temperature. The function update_et0_from_temp()
    is provided for that.
  - plant spacing / area_per_plant: 0.40 m × 0.45 m = 0.18 m²/plant (confirmed)
  - FC, WP: from the pot calibration (net house soil)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ===========================================================================
# Configuration — EDIT THESE as field reality is confirmed
# ===========================================================================

@dataclass
class DeployPolicyConfig:
    # --- Decision windows (24-h local time) ---
    decision_hours: Tuple[int, int] = (9, 14)   # 09:00 and 14:00

    # --- Field geometry (CONFIRMED) ---
    plant_to_plant_m: float = 0.40
    row_to_row_m: float = 0.45
    plot_length_m: float = 33.0
    plot_width_m: float = 14.0

    # --- Shade house ET0 (ESTIMATE — validate with in-house sensors) ---
    et0_shade_mm_day: float = 4.8       # 25% net + closed walls, Telangana summer
    drip_efficiency: float = 0.90       # fraction of applied water reaching root zone

    # --- Soil (net house soil, from pot calibration) ---
    FC: float = 95.0                    # field capacity, sensor %
    WP: float = 25.0                    # wilting point, sensor %
    MAD: float = 0.35                   # management allowed depletion (cucumber)

    # --- Crop stages (days after transplant) ---
    # English cucumber, ~90-day cycle
    stage_days: Dict[str, int] = field(default_factory=lambda: {
        "initial": 12, "development": 18, "mid": 35, "late": 25,
    })
    Kc: Dict[str, float] = field(default_factory=lambda: {
        "initial": 0.60, "development": 0.80, "mid": 1.00, "late": 0.75,
    })
    # Target moisture band per stage [lo, hi], sensor %
    optimal_band: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "initial":     (65, 90),
        "development": (70, 90),
        "mid":         (75, 95),   # flowering/fruiting — keep wettest
        "late":        (60, 85),
    })

    # --- Safety caps ---
    max_session_L: float = 1.50         # never recommend more than this per plant per session
    round_to_L: float = 0.25            # round recommendation to nearest this

    # --- Root-zone weighting (if two sensor depths available) ---
    use_two_depths: bool = False
    surface_weight: float = 0.6
    deep_weight: float = 0.4


# ===========================================================================
# Policy
# ===========================================================================

class DeployInitialPolicy:
    """FAO-56 MAD initial policy for hand-executed 9 AM / 2 PM irrigation."""

    def __init__(self, cfg: DeployPolicyConfig | None = None):
        self.cfg = cfg or DeployPolicyConfig()
        self.area_per_plant = self.cfg.plant_to_plant_m * self.cfg.row_to_row_m
        self._stage_starts = self._compute_stage_starts()

    # -- stage helpers -----------------------------------------------------
    def _compute_stage_starts(self) -> List[Tuple[str, int, int]]:
        out, d0 = [], 0
        for name in ["initial", "development", "mid", "late"]:
            length = self.cfg.stage_days[name]
            out.append((name, d0, d0 + length))
            d0 += length
        return out

    def stage_for_day(self, days_after_transplant: int) -> str:
        for name, start, end in self._stage_starts:
            if start <= days_after_transplant < end:
                return name
        return "late"   # after the last stage, stay in 'late' until harvest end

    # -- ET0 update from in-house temperature (use once sensors live) ------
    def update_et0_from_temp(self, t_mean_c: float, t_min_c: float,
                             t_max_c: float, ra_mj_m2_day: float = 38.0) -> float:
        """
        Hargreaves-Samani ET0 from shade-house air temperature.
        Call this daily once the in-house temp sensors are logging, and
        assign the result to cfg.et0_shade_mm_day.

        ET0 = 0.0023 * Ra * (Tmean + 17.8) * sqrt(Tmax - Tmin)
        Ra in mm/day equivalent (≈ 0.408 * Ra_MJ). For 17°N June, Ra≈38 MJ.
        """
        ra_mm = 0.408 * ra_mj_m2_day
        et0 = 0.0023 * ra_mm * (t_mean_c + 17.8) * max(t_max_c - t_min_c, 0.0) ** 0.5
        # Shade-house reduction already implicit in measured (lower) temps,
        # but apply a modest screen factor for intercepted radiation.
        et0 *= 0.90
        self.cfg.et0_shade_mm_day = round(float(et0), 2)
        return self.cfg.et0_shade_mm_day

    # -- core decision -----------------------------------------------------
    def recommend(self, *,
                  soil_moisture_pct: float,
                  days_after_transplant: int,
                  hour: int,
                  soil_moisture_deep_pct: Optional[float] = None) -> Dict:
        """
        Return an irrigation recommendation for the current session.

        Parameters
        ----------
        soil_moisture_pct : latest surface (or single-probe) moisture, %
        days_after_transplant : integer days since transplant
        hour : current hour (must be one of cfg.decision_hours)
        soil_moisture_deep_pct : optional deep-layer moisture if 2-depth probe

        Returns
        -------
        dict with: litres_per_plant, litres_per_plot, reason, stage, M_root,
                   target_band, mm_equivalent
        """
        cfg = self.cfg

        if hour not in cfg.decision_hours:
            return {"litres_per_plant": 0.0, "litres_per_plot": 0.0,
                    "reason": f"Not a decision window (only {cfg.decision_hours})",
                    "stage": None, "M_root": soil_moisture_pct}

        # Root-zone moisture
        if cfg.use_two_depths and soil_moisture_deep_pct is not None:
            M = (cfg.surface_weight * soil_moisture_pct
                 + cfg.deep_weight * soil_moisture_deep_pct)
        else:
            M = soil_moisture_pct

        stage = self.stage_for_day(days_after_transplant)
        lo, hi = cfg.optimal_band[stage]
        kc = cfg.Kc[stage]
        trigger = cfg.FC - cfg.MAD * (cfg.FC - cfg.WP)

        # Case (a): soil already at/above upper band → skip
        if M >= hi:
            return self._result(0.0, M, stage, (lo, hi),
                                 f"M={M:.0f}% ≥ upper band {hi}% — soil wet enough, skip")

        # Base requirement: half the daily ETc (two sessions/day)
        etc_mm_day = kc * cfg.et0_shade_mm_day
        base_L = (etc_mm_day * self.area_per_plant / cfg.drip_efficiency) / 2.0

        # Deficit top-up: if below MAD trigger, add water proportional to how
        # far below the lo-band the soil is (fraction of the lo→trigger span)
        topup_L = 0.0
        if M < trigger:
            deficit_fraction = min((trigger - M) / max(trigger - cfg.WP, 1.0), 1.0)
            topup_L = deficit_fraction * base_L   # up to one extra base dose

        vol = base_L + topup_L
        vol = min(vol, cfg.max_session_L)
        # round
        vol = round(vol / cfg.round_to_L) * cfg.round_to_L

        if M < trigger:
            reason = (f"M={M:.0f}% < trigger {trigger:.0f}% (dry) — "
                      f"base {base_L:.2f}L + deficit top-up {topup_L:.2f}L")
        else:
            reason = (f"M={M:.0f}% in band [{lo},{hi}] — "
                      f"routine ETc replacement {base_L:.2f}L")

        return self._result(vol, M, stage, (lo, hi), reason)

    def _result(self, litres_per_plant: float, M: float, stage, band, reason) -> Dict:
        n_plants = (self.cfg.plot_length_m * self.cfg.plot_width_m) / self.area_per_plant
        # mm equivalent over the plot
        mm = (litres_per_plant / self.area_per_plant) if self.area_per_plant else 0.0
        return {
            "litres_per_plant": round(litres_per_plant, 2),
            "litres_per_plot": round(litres_per_plant * n_plants, 1),
            "mm_equivalent": round(mm, 2),
            "stage": stage,
            "M_root": round(M, 1),
            "target_band": band,
            "reason": reason,
        }


# ===========================================================================
# Printable reference table (run this to get a wall-card for the field)
# ===========================================================================

def print_reference_card(cfg: DeployPolicyConfig | None = None) -> None:
    pol = DeployInitialPolicy(cfg)
    c = pol.cfg
    trigger = c.FC - c.MAD * (c.FC - c.WP)
    print("=" * 70)
    print("  T2 INITIAL IRRIGATION POLICY — FIELD REFERENCE CARD")
    print("  English cucumber | shade house 25% net | net house soil")
    print("=" * 70)
    print(f"  Decide at: 09:00 and 14:00 daily")
    print(f"  Read soil moisture from the latest sensor sample at each time.")
    print(f"  Plant area: {c.plant_to_plant_m}×{c.row_to_row_m} m = "
          f"{pol.area_per_plant:.3f} m²/plant")
    print(f"  Plot: {c.plot_width_m}×{c.plot_length_m} m = "
          f"{c.plot_width_m*c.plot_length_m:.0f} m² "
          f"(~{c.plot_width_m*c.plot_length_m/pol.area_per_plant:.0f} plants)")
    print(f"  Shade-house ET0: {c.et0_shade_mm_day} mm/day (validate w/ sensors)")
    print("-" * 70)
    print(f"  RULE: if moisture ≥ upper band → DO NOT water.")
    print(f"        if moisture < {trigger:.0f}% → water (dry, needs top-up).")
    print(f"        else → routine half-ETc replacement.")
    print("-" * 70)
    print(f"  Recommended L/plant per SESSION, by stage and moisture:")
    print(f"  {'Stage (days)':<20}{'M≥upper':<10}{'M mid-band':<13}{'M<trigger':<12}")
    examples = [
        ("initial (0-12)",   95, 78, 55),
        ("develop (12-30)",  92, 80, 55),
        ("mid/flower(30-65)",96, 85, 65),
        ("late (65-90)",     88, 72, 50),
    ]
    stage_keys = ["initial", "development", "mid", "late"]
    for (label, m_hi, m_mid, m_lo), skey in zip(examples, stage_keys):
        start = pol._stage_starts[stage_keys.index(skey)][1]
        days = start + 1
        r_hi = pol.recommend(soil_moisture_pct=m_hi, days_after_transplant=days, hour=9)
        r_md = pol.recommend(soil_moisture_pct=m_mid, days_after_transplant=days, hour=9)
        r_lo = pol.recommend(soil_moisture_pct=m_lo, days_after_transplant=days, hour=9)
        print(f"  {label:<20}"
              f"{r_hi['litres_per_plant']:<10.2f}"
              f"{r_md['litres_per_plant']:<13.2f}"
              f"{r_lo['litres_per_plant']:<12.2f}")
    print("=" * 70)
    print("  NOTE: per-session volumes are PER PLANT. Multiply by plant count")
    print("  for the whole plot, or water proportionally along the drip line.")
    print("=" * 70)


if __name__ == "__main__":
    print_reference_card()
    print()
    # Worked example
    pol = DeployInitialPolicy()
    print("Worked example — day 40 (mid/flowering stage), 9 AM, moisture 68%:")
    rec = pol.recommend(soil_moisture_pct=68, days_after_transplant=40, hour=9)
    for k, v in rec.items():
        print(f"  {k}: {v}")

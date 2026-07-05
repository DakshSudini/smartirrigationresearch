"""
obs.py
======
Single source of truth for the 14-D observation vector.

Every code path that builds an observation — the simulator env (training
rollouts), the offline warm-start augmentation, the online updater, and the
webapp serving layer — MUST go through `build_obs` here. Previously each path
built the vector inline and slot 13 drifted apart (session_flag vs.
hours_since_irrig vs. a hardcoded 0.5), so a slice of the training data taught
the network a different meaning for that input than the deployed model used.

The layout below is the one the live checkpoint (iql_final.pt) was trained on,
i.e. it matches the original IrrigationEnv._obs. Do NOT reorder or change the
normalisation of any slot without retraining, or the existing model's inputs
become meaningless.

Index | Feature                         | Definition
------+---------------------------------+-------------------------------------
  0   | moisture_surface_norm           | (M_surf - 50) / 30
  1   | moisture_deep_norm              | (M_deep - 50) / 30
  2   | soil_temperature_norm           | (T_soil - 25) / 5
  3   | delta_moisture_surface_1h_norm  | dM_surf / 10
  4   | delta_moisture_deep_1h_norm     | dM_deep / 10
  5   | hour_sin                        | sin(2*pi*hour/24)
  6   | hour_cos                        | cos(2*pi*hour/24)
  7   | day_since_transplant_norm       | min(day, 100) / 100
  8   | stage_onehot_initial            | {0,1}
  9   | stage_onehot_development        | {0,1}
 10   | stage_onehot_mid                | {0,1}
 11   | stage_onehot_late               | {0,1}
 12   | cum_water_today_norm            | cum_water_today_mm / cap
 13   | session_flag                    | 0.0 at 09:00 window, 1.0 at 14:00

`cap` = max_daily_water_L_per_plant * plants_per_m2 (mm-equivalent daily cap).

NOTE on slot 13: it carries the decision-window flag, NOT
"hours_since_last_irrigation". The config `state.features` list has been
corrected to match. If a future model is retrained with hours_since_irrig as a
15th feature, add it here as index 14 and bump OBS_DIM.
"""

from __future__ import annotations

import numpy as np

OBS_DIM = 14

STAGE_NAMES = ["initial", "development", "mid", "late"]


def stage_onehot(stage_idx: int) -> list:
    oh = [0.0] * 4
    oh[int(stage_idx)] = 1.0
    return oh


def build_obs(*,
              M_surf: float,
              M_deep: float,
              T_soil: float,
              hour: float,
              day: int,
              stage_idx: int,
              cum_water_today_mm: float,
              daily_cap_mm: float,
              session_flag: float,
              dM_surf: float = 0.0,
              dM_deep: float = 0.0) -> np.ndarray:
    """Build the canonical 14-D observation. See module docstring for layout.

    All callers pass RAW physical values; normalisation happens here exactly
    once so it can never drift between training and serving.
    """
    cap = max(daily_cap_mm, 1e-3)
    obs = np.array([
        (M_surf - 50.0) / 30.0,
        (M_deep - 50.0) / 30.0,
        (T_soil - 25.0) / 5.0,
        dM_surf / 10.0,
        dM_deep / 10.0,
        float(np.sin(2 * np.pi * hour / 24.0)),
        float(np.cos(2 * np.pi * hour / 24.0)),
        min(day, 100) / 100.0,
        *stage_onehot(stage_idx),
        cum_water_today_mm / cap,
        float(session_flag),
    ], dtype=np.float32)
    return obs

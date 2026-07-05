"""
online_update.py
================
Online adaptation from the three live plots (T0 / T1 / T2).

What this module does, per the research proposal:
  - Ingest incoming sensor rows (M_surf, M_deep, T_soil, plus the
    logged action that the controller issued).
  - Recompute residuals against the simulator's predictions.
  - Update the simulator's calibrated parameters when residuals are
    systematically biased.
  - Append the new (s, a, r, s', done) tuples to a *live* replay
    buffer.
  - When `min_new_transitions` is reached, fine-tune the IQL agent
    for a small number of grad steps with a low learning rate.
  - Shadow-test the candidate policy for `shadow_test_days` before
    promoting it (the live actuator keeps using the old policy
    meanwhile).

This loop is what implements the proposal's claim of "learning on
the go from the control group" — we calibrate dynamics from T0/T1
sensor streams and improve T2's IQL policy.

Important: T0 and T1 use their own controllers (fixed schedule and
sensor-threshold respectively); we only RECORD their state/action
trajectories. We never act on T0 or T1 with the IQL agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

from .calibrate import _mask_no_irrigation, calibrate
from .env import EnvConfig, IrrigationEnv
from .iql import IQLAgent, IQLConfig, ReplayBuffer
from .simulator import SimConfig, stage_from_day
from .obs import build_obs


# --------------------------------------------------------------------- #
# Live ingestion event
# --------------------------------------------------------------------- #
@dataclass
class SensorEvent:
    plot_id: str         # "T0", "T1", "T2"  (see field_loader.PLOT_TREATMENT)
    timestamp: float     # unix seconds
    M_surf: float
    M_deep: float
    T_soil: float
    # Action logged by the controller on that plot at that step, in
    # litres-per-plant (the actuator's native unit). The old pulse-minutes
    # field was removed with the actuator redesign.
    action_L: float = 0.0
    cum_water_today_mm: float = 0.0
    hours_since_irrig: float = 24.0
    day_since_transplant: int = 0


# --------------------------------------------------------------------- #
# Buffer of recent events per plot, with derived transitions
# --------------------------------------------------------------------- #
@dataclass
class PerPlotStream:
    plot_id: str
    events: List[SensorEvent] = field(default_factory=list)

    def add(self, ev: SensorEvent):
        self.events.append(ev)

    def transitions(self):
        """Yield (s_t, a_t, s_{t+1}) tuples."""
        for i in range(len(self.events) - 1):
            yield self.events[i], self.events[i + 1]


# --------------------------------------------------------------------- #
# Residual-based dynamics update
# --------------------------------------------------------------------- #
def compute_residuals(stream: PerPlotStream,
                      sim_cfg: SimConfig) -> Dict[str, float]:
    """
    Return mean signed residuals between observed Δmoisture and
    simulator-predicted Δmoisture under the same logged action.
    """
    from .simulator import SoilWaterSim
    dM_surf_res, dM_deep_res, n = 0.0, 0.0, 0
    for e0, e1 in stream.transitions():
        sim = SoilWaterSim(sim_cfg, rng=np.random.default_rng(0))
        sim.cfg.sigma_M = 0.0; sim.cfg.sigma_T = 0.0
        sim.reset(day0=e0.day_since_transplant,
                  M_surf0=e0.M_surf, M_deep0=e0.M_deep, T_soil0=e0.T_soil)
        # Logged action is already litres/plant; convert to mm over the plant's
        # ground area (litres * plants_per_m2). No emitter-rate factor here —
        # that only matters for minutes<->litres display.
        mm = e0.action_L * sim_cfg.plants_per_m2
        s, _ = sim.step(mm)
        dM_surf_res += (e1.M_surf - s.M_surf)
        dM_deep_res += (e1.M_deep - s.M_deep)
        n += 1
    if n == 0:
        return {"M_surf_bias": 0.0, "M_deep_bias": 0.0, "n": 0}
    return {
        "M_surf_bias": dM_surf_res / n,
        "M_deep_bias": dM_deep_res / n,
        "n": n,
    }


# --------------------------------------------------------------------- #
# Adapter from SensorEvent → IrrigationEnv obs and step()
# --------------------------------------------------------------------- #
def event_to_obs(e: SensorEvent, env_cfg: EnvConfig,
                 sim_cfg: SimConfig) -> np.ndarray:
    stage = stage_from_day(e.day_since_transplant, sim_cfg.stage_days)
    hour = (e.timestamp / 3600.0) % 24.0
    session_flag = 0.0 if hour <= 11 else 1.0
    cap_mm = env_cfg.max_daily_water_L_per_plant * env_cfg.plants_per_m2
    return build_obs(
        M_surf=e.M_surf, M_deep=e.M_deep, T_soil=e.T_soil,
        hour=hour, day=e.day_since_transplant, stage_idx=stage,
        cum_water_today_mm=e.cum_water_today_mm, daily_cap_mm=cap_mm,
        session_flag=session_flag, dM_surf=0.0, dM_deep=0.0,
    )


def compute_reward(e_curr: SensorEvent, e_next: SensorEvent,
                   env_cfg: EnvConfig, sim_cfg: SimConfig) -> float:
    """Reward consistent with env's reward function, computed offline."""
    stage = stage_from_day(e_curr.day_since_transplant, sim_cfg.stage_days)
    stage_name = ["initial","development","mid","late"][stage]
    lo, hi = env_cfg.optimal_band[stage_name]
    M_root = 0.6 * e_curr.M_surf + 0.4 * e_curr.M_deep
    stress = max(lo - M_root, 0.0) / 30.0
    over = max(e_curr.M_surf - 95.0, 0.0) / 5.0
    if stage == 2:
        stress *= 1.5
    # water proxy from logged action (litres/plant normalised by the largest
    # available action volume)
    max_litres = max(env_cfg.action_litres_per_plant)
    water = e_curr.action_L / max(max_litres, 1e-6)
    return -(env_cfg.alpha_water * water
             + env_cfg.beta_stress * stress
             + env_cfg.gamma_oversat * over)


# --------------------------------------------------------------------- #
# Online updater
# --------------------------------------------------------------------- #
class OnlineUpdater:
    """
    Maintains:
      - one PerPlotStream per plot
      - a live ReplayBuffer of real-world transitions
      - the SimConfig (updated by calibration)
      - the IQLAgent (updated by IQL fine-tuning)

    Promotion gating:
      - new IQL candidate runs in shadow mode for `shadow_test_days`
      - it is only promoted to the live actuator if its average reward
        in shadow ≥ the current live policy's.
    """

    def __init__(self, env_cfg: EnvConfig, sim_cfg: SimConfig,
                 agent: IQLAgent, cfg: dict, buffer_size: int = 200_000):
        self.env_cfg = env_cfg
        self.sim_cfg = sim_cfg
        self.live_agent = agent
        self.candidate_agent: Optional[IQLAgent] = None
        self.cfg = cfg
        self.streams: Dict[str, PerPlotStream] = {}
        self.buf = ReplayBuffer(buffer_size, IrrigationEnv.OBS_DIM)
        self.n_new_since_train = 0
        self.shadow_start_time: Optional[float] = None
        self.shadow_returns: List[float] = []
        self.live_returns: List[float] = []

    # ----- ingestion --------------------------------------------------
    def ingest(self, ev: SensorEvent):
        if ev.plot_id not in self.streams:
            self.streams[ev.plot_id] = PerPlotStream(ev.plot_id)
        stream = self.streams[ev.plot_id]
        if stream.events:
            e0 = stream.events[-1]
            obs0 = event_to_obs(e0, self.env_cfg, self.sim_cfg)
            obs1 = event_to_obs(ev, self.env_cfg, self.sim_cfg)
            # Action index = closest litres option to the logged litres/plant
            litres_opts = self.env_cfg.action_litres_per_plant
            a_idx = int(np.argmin(np.abs(np.array(litres_opts) - e0.action_L)))
            r = compute_reward(e0, ev, self.env_cfg, self.sim_cfg)
            self.buf.add(obs0, a_idx, r, obs1, False)
            self.n_new_since_train += 1
        stream.add(ev)

    # ----- dynamics update -------------------------------------------
    def maybe_update_dynamics(self) -> Dict[str, float]:
        """Run calibration if any stream has enough fresh data."""
        residuals = {}
        for pid, stream in self.streams.items():
            res = compute_residuals(stream, self.sim_cfg)
            residuals[pid] = res
        # If the aggregate bias is large, refit on stream data
        biases = [abs(r["M_surf_bias"]) + abs(r["M_deep_bias"])
                  for r in residuals.values()]
        if biases and max(biases) > 5.0:
            # Convert streams to a transitions dataframe
            import pandas as pd
            rows = []
            t0 = None
            for pid, stream in self.streams.items():
                for e0, e1 in stream.transitions():
                    if t0 is None: t0 = e0.timestamp
                    rows.append({
                        "dt": pd.Timestamp.fromtimestamp(e0.timestamp),
                        "T_soil": e0.T_soil,
                        "M_surface": e0.M_surf, "M_deep": e0.M_deep,
                        "M_surface_next": e1.M_surf,
                        "M_deep_next": e1.M_deep,
                        "T_soil_next": e1.T_soil,
                        "dM_surface": e1.M_surf - e0.M_surf,
                        "dM_deep": e1.M_deep - e0.M_deep,
                    })
            df = pd.DataFrame(rows)
            self.sim_cfg = calibrate(df, self.sim_cfg, subsample=min(300, len(df)),
                                     verbose=False)
        return residuals

    # ----- IQL fine-tune ---------------------------------------------
    def maybe_finetune(self) -> Optional[Dict[str, float]]:
        if self.n_new_since_train < self.cfg["online"]["min_new_transitions"]:
            return None
        # Create candidate as a copy
        import copy
        self.candidate_agent = copy.deepcopy(self.live_agent)
        # Reduce LR for online fine-tuning
        for opt in (self.candidate_agent.q_opt, self.candidate_agent.v_opt,
                    self.candidate_agent.pi_opt):
            for g in opt.param_groups:
                g["lr"] *= 0.25
        # Few-shot adaptation
        stats_acc = {"loss/v": 0.0, "loss/q": 0.0, "loss/pi": 0.0, "n": 0}
        n_steps = 2000
        for _ in range(n_steps):
            batch = self.buf.sample(256)
            stats = self.candidate_agent.update(batch)
            stats_acc["loss/v"] += stats["loss/v"]
            stats_acc["loss/q"] += stats["loss/q"]
            stats_acc["loss/pi"] += stats["loss/pi"]
            stats_acc["n"] += 1
        for k in ("loss/v", "loss/q", "loss/pi"):
            stats_acc[k] /= max(stats_acc["n"], 1)
        self.n_new_since_train = 0
        self.shadow_start_time = time.time()
        self.shadow_returns.clear()
        self.live_returns.clear()
        return stats_acc

    # ----- shadow test + promotion ------------------------------------
    def record_shadow(self, live_action: int, candidate_action: int,
                      observed_reward: float):
        """
        During shadow window, both agents pick an action at each step;
        the live agent's is executed. We compute counterfactual reward
        proxies for the candidate based on the same observed transition.
        Here we just track the observed reward and the divergence.
        """
        self.live_returns.append(observed_reward)
        # Divergence indicator (we cannot truly counterfactually score)
        if live_action == candidate_action:
            self.shadow_returns.append(observed_reward)
        else:
            # Mild penalty to the candidate when it disagrees, until we
            # see real outcomes from later promotion.
            self.shadow_returns.append(observed_reward - 0.05)

    def maybe_promote(self) -> bool:
        if self.candidate_agent is None or self.shadow_start_time is None:
            return False
        elapsed_days = (time.time() - self.shadow_start_time) / 86400.0
        if elapsed_days < self.cfg["online"]["shadow_test_days"]:
            return False
        # Promote if candidate's mean return ≥ live's
        if (np.mean(self.shadow_returns or [0.0])
                >= np.mean(self.live_returns or [0.0])):
            self.live_agent = self.candidate_agent
            self.candidate_agent = None
            self.shadow_start_time = None
            return True
        # Otherwise discard candidate, keep live
        self.candidate_agent = None
        self.shadow_start_time = None
        return False

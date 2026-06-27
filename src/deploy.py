"""
deploy.py
=========
Runtime inference + safety wrapper. The trained IQL policy never
issues an action that would (a) exceed the daily volume cap, (b) fire
outside permitted hours, or (c) fall under the min-interval safety
fence.

Also provides `compare_policies(...)`, the head-to-head harness used
by the research evaluation (T0 vs T1 vs T2).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

from .data_loader import load_fyllo_excel, build_observed_transitions
from .calibrate import calibrate
from .simulator import SimConfig
from .env import EnvConfig, IrrigationEnv
from .iql import IQLAgent, IQLConfig
from .initial_policy import InitialPolicy, InitialPolicyConfig


# --------------------------------------------------------------------- #
# Safety wrapper
# --------------------------------------------------------------------- #
@dataclass
class SafetyConfig:
    allowed_hours: List[int]
    pulse_minutes: List[int]
    daily_cap_L_per_plant: float
    plants_per_m2: float
    min_interval_min: int
    # If moisture above this fraction of SAT, never irrigate
    no_irrigate_above_pct: float = 95.0
    # If both layers below this %, force a fixed safety pulse
    emergency_below_pct: float = 28.0
    emergency_pulse_min: int = 5


class SafeController:
    """Wrap a policy (IQL or initial) and apply guards."""

    def __init__(self, base_policy, cfg: SafetyConfig):
        self.base = base_policy
        self.cfg = cfg
        # pulse_minutes -> action index
        self.pulse_to_idx = {p: i for i, p in enumerate(cfg.pulse_minutes)}

    def act(self, *, obs: np.ndarray, M_surf: float, M_deep: float,
            hour: float, day: int,
            cum_water_today_mm: float, hours_since_irrig: float,
            ) -> Tuple[int, str]:
        """
        Returns (action_idx, reason).  reason ∈ {"policy", "safety_block",
        "out_of_window", "daily_cap", "min_interval", "oversat",
        "emergency_dry"}.
        """
        # Emergency override: both layers below dry threshold → force pulse
        if (M_surf < self.cfg.emergency_below_pct and
                M_deep < self.cfg.emergency_below_pct and
                hours_since_irrig * 60 >= self.cfg.min_interval_min):
            return (self.pulse_to_idx.get(self.cfg.emergency_pulse_min, 1),
                    "emergency_dry")
        # Out of permitted window?
        if int(hour) not in self.cfg.allowed_hours:
            return self.pulse_to_idx.get(0, 0), "out_of_window"
        # Min interval?
        if hours_since_irrig * 60 < self.cfg.min_interval_min:
            return self.pulse_to_idx.get(0, 0), "min_interval"
        # Daily cap?
        cap_mm = self.cfg.daily_cap_L_per_plant * self.cfg.plants_per_m2
        if cum_water_today_mm >= cap_mm:
            return self.pulse_to_idx.get(0, 0), "daily_cap"
        # Over-saturation?
        if M_surf >= self.cfg.no_irrigate_above_pct:
            return self.pulse_to_idx.get(0, 0), "oversat"
        # Defer to policy
        a = self.base(obs=obs, M_surf=M_surf, M_deep=M_deep,
                      hour=hour, day=day,
                      cum_water_today_mm=cum_water_today_mm,
                      hours_since_irrig=hours_since_irrig)
        return int(a), "policy"


# --------------------------------------------------------------------- #
# Policy adapters
# --------------------------------------------------------------------- #
def make_iql_policy(agent: IQLAgent):
    """Adapter: signature compatible with SafeController.base."""
    def _p(*, obs, **kwargs):
        return agent.act(obs, deterministic=True)
    return _p


def make_initial_policy(ip: InitialPolicy, pulse_to_idx: Dict[int, int]):
    def _p(*, obs, M_surf, M_deep, hour, day,
           cum_water_today_mm, hours_since_irrig):
        pulse = ip.act(M_surf=M_surf, M_deep=M_deep, hour=hour, day=day,
                       cum_water_today_mm=cum_water_today_mm,
                       hours_since_irrig=hours_since_irrig)
        return pulse_to_idx.get(pulse, 0)
    return _p


def make_fixed_schedule_policy(pulse_to_idx: Dict[int, int],
                               irrig_hour: int = 6,
                               pulse_min: int = 10):
    """Mimic a 'traditional farmer' (T0) baseline: fixed daily schedule."""
    def _p(*, obs, M_surf, M_deep, hour, day,
           cum_water_today_mm, hours_since_irrig):
        if int(hour) == irrig_hour:
            return pulse_to_idx.get(pulse_min, 0)
        return pulse_to_idx.get(0, 0)
    return _p


# --------------------------------------------------------------------- #
# Single-policy rollout (returns full trace + summary)
# --------------------------------------------------------------------- #
def rollout(env: IrrigationEnv, controller: SafeController,
            label: str = "") -> dict:
    obs = env.reset()
    trace = []
    while True:
        s = env.sim.state
        a, reason = controller.act(
            obs=obs, M_surf=s.M_surf, M_deep=s.M_deep,
            hour=s.hour, day=s.day,
            cum_water_today_mm=s.cum_water_today_mm,
            hours_since_irrig=s.hours_since_irrig,
        )
        next_obs, r, done, info = env.step(a)
        trace.append({
            "day": s.day, "hour": s.hour,
            "M_surf": s.M_surf, "M_deep": s.M_deep,
            "T_soil": s.T_soil,
            "action_idx": a, "pulse_min": info["pulse_min"],
            "reason": reason, "reward": r,
            "Ks": info["Ks"], "stage": info["stage_name"],
            "irrigation_mm": info["irrigation_mm"],
        })
        obs = next_obs
        if done: break
    end = env.sim.state
    return {
        "label": label,
        "total_return": sum(t["reward"] for t in trace),
        "total_water_mm": end.cum_water_total_mm,
        "total_yield_proxy": end.cum_yield_proxy,
        "stress_hours_sensitive": end.stress_hours_in_sensitive,
        "water_use_efficiency": end.cum_yield_proxy /
            max(end.cum_water_total_mm, 1e-6),
        "trace": trace,
    }


# --------------------------------------------------------------------- #
# Three-policy comparison: T0 fixed schedule, T1 sensor-threshold (initial
# policy), T2 IQL
# --------------------------------------------------------------------- #
def compare_policies(cfg_path: str, ckpt_path: str, fyllo_path: str,
                     calibrate_sim: bool = True,
                     n_seeds: int = 5) -> dict:
    cfg = yaml.safe_load(open(cfg_path))
    sim_cfg = SimConfig(
        FC=cfg["soil"]["field_capacity_pct"],
        WP=cfg["soil"]["wilting_point_pct"],
        SAT=cfg["soil"]["saturation_pct"],
        surface_depth_m=cfg["soil"]["surface_layer_depth_m"],
        deep_depth_m=cfg["soil"]["deep_layer_depth_m"],
        tau_drain_h=cfg["soil"]["tau_drain_h"],
        Ke_max=cfg["soil"]["Ke_max"],
        stage_days=cfg["crop"]["stage_days"],
        Kc=cfg["crop"]["Kc"],
        plants_per_m2=cfg["crop"]["plants_per_m2"],
        root_depth_init_m=cfg["crop"]["root_depth_m_initial"],
        root_depth_max_m=cfg["crop"]["root_depth_m_max"],
    )
    if calibrate_sim and Path(fyllo_path).exists():
        plots = load_fyllo_excel(fyllo_path)
        trans = build_observed_transitions(plots)
        sim_cfg = calibrate(trans, sim_cfg, subsample=200, verbose=False)

    env_cfg = EnvConfig(
        action_pulse_minutes=cfg["actuator"]["action_pulse_minutes"],
        emitter_rate_L_per_h=cfg["actuator"]["emitter_rate_L_per_h"],
        plants_per_m2=cfg["crop"]["plants_per_m2"],
        surface_depth_m=cfg["soil"]["surface_layer_depth_m"],
        decision_interval_minutes=cfg["actuator"]["decision_interval_minutes"],
        max_daily_water_L_per_plant=cfg["actuator"]["max_daily_volume_L_per_plant"],
        alpha_water=cfg["reward"]["alpha_water"],
        beta_stress=cfg["reward"]["beta_stress"],
        gamma_oversat=cfg["reward"]["gamma_oversat"],
        yield_terminal_weight=cfg["reward"]["yield_terminal_weight"],
        optimal_band=cfg["crop"]["optimal_moisture_band"],
        total_days=sum(cfg["crop"]["stage_days"].values()),
    )

    # IQL agent
    iql_cfg = IQLConfig(
        obs_dim=IrrigationEnv.OBS_DIM,
        n_actions=len(env_cfg.action_pulse_minutes),
        hidden_dim=cfg["iql"]["hidden_dim"],
        n_hidden=cfg["iql"]["hidden_layers"],
    )
    agent = IQLAgent(iql_cfg)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    agent.load_state_dict(sd)

    # Initial policy
    init_cfg = InitialPolicyConfig(
        optimal_band=cfg["crop"]["optimal_moisture_band"],
        allowed_hours=cfg["actuator"]["allowed_hours"],
        pulse_minutes_options=cfg["actuator"]["action_pulse_minutes"],
        emitter_rate_L_per_h=cfg["actuator"]["emitter_rate_L_per_h"],
        plants_per_m2=cfg["crop"]["plants_per_m2"],
        surface_depth_m=cfg["soil"]["surface_layer_depth_m"],
        deep_depth_m=cfg["soil"]["deep_layer_depth_m"],
        daily_cap_L_per_plant=cfg["actuator"]["max_daily_volume_L_per_plant"],
        min_interval_min=cfg["actuator"]["min_interval_minutes"],
        MAD=cfg["crop"]["MAD"],
    )
    ip = InitialPolicy(init_cfg, sim_cfg)
    pulse_to_idx = {p: i for i, p in enumerate(env_cfg.action_pulse_minutes)}

    safe_cfg = SafetyConfig(
        allowed_hours=cfg["actuator"]["allowed_hours"],
        pulse_minutes=cfg["actuator"]["action_pulse_minutes"],
        daily_cap_L_per_plant=cfg["actuator"]["max_daily_volume_L_per_plant"],
        plants_per_m2=cfg["crop"]["plants_per_m2"],
        min_interval_min=cfg["actuator"]["min_interval_minutes"],
    )

    # Three controllers
    controllers = {
        "T0_fixed_schedule": SafeController(
            make_fixed_schedule_policy(pulse_to_idx), safe_cfg),
        "T1_initial_policy": SafeController(
            make_initial_policy(ip, pulse_to_idx), safe_cfg),
        "T2_IQL":            SafeController(
            make_iql_policy(agent), safe_cfg),
    }

    results = {name: [] for name in controllers}
    for seed in range(n_seeds):
        rng = np.random.default_rng(100 + seed)
        for name, ctrl in controllers.items():
            env = IrrigationEnv(env_cfg, sim_cfg, rng=rng)
            r = rollout(env, ctrl, label=name)
            r.pop("trace")  # drop trace for summary
            results[name].append(r)

    # Aggregate
    summary = {}
    for name, runs in results.items():
        summary[name] = {
            "n_seeds": len(runs),
            "return_mean":       float(np.mean([r["total_return"] for r in runs])),
            "return_std":        float(np.std([r["total_return"] for r in runs])),
            "water_mm_mean":     float(np.mean([r["total_water_mm"] for r in runs])),
            "water_mm_std":      float(np.std([r["total_water_mm"] for r in runs])),
            "yield_mean":        float(np.mean([r["total_yield_proxy"] for r in runs])),
            "yield_std":         float(np.std([r["total_yield_proxy"] for r in runs])),
            "stress_h_mean":     float(np.mean([r["stress_hours_sensitive"] for r in runs])),
            "stress_h_std":      float(np.std([r["stress_hours_sensitive"] for r in runs])),
            "wue_mean":          float(np.mean([r["water_use_efficiency"] for r in runs])),
            "wue_std":           float(np.std([r["water_use_efficiency"] for r in runs])),
        }
    return summary


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--fyllo", default="./fyllo.xlsx")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--no-calibrate", action="store_true")
    args = ap.parse_args()
    s = compare_policies(args.config, args.ckpt, args.fyllo,
                         calibrate_sim=not args.no_calibrate,
                         n_seeds=args.seeds)
    print("\n=== Policy comparison (mean ± std across seeds) ===\n")
    fmt = lambda m, sd: f"{m:>8.2f} ± {sd:>6.2f}"
    cols = ["return", "water_mm", "yield", "stress_h", "wue"]
    header = f"{'controller':24} " + "  ".join(f"{c:>17}" for c in cols)
    print(header); print("-" * len(header))
    for name, st in s.items():
        line = f"{name:24} " + "  ".join([
            fmt(st["return_mean"], st["return_std"]),
            fmt(st["water_mm_mean"], st["water_mm_std"]),
            fmt(st["yield_mean"], st["yield_std"]),
            fmt(st["stress_h_mean"], st["stress_h_std"]),
            fmt(st["wue_mean"], st["wue_std"]),
        ])
        print(line)

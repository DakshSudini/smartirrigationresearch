"""
train.py
========
End-to-end offline-RL training pipeline:

  1. Load Fyllo Excel -> calibrate SoilWaterSim.
  2. Generate offline transitions by rolling out a *mix* of behaviour
     policies in the calibrated simulator (FAO-56 heuristic, deficit,
     over-water, and random). This gives the offline dataset coverage
     in (s, a) space.
  3. If the soil sensor reading log (data/soil_log.csv) is available:
       a. Re-calibrate tau_drain_h from net house soil drying curves.
       b. Add PROPERLY LABELLED real (s, a, r, s') transitions to the
          buffer — these have known action_mL from the Remarks column.
  4. Augment with Fyllo observed transitions (action=0 imputed, valid for
     the non-irrigation rows only; these are state-prior contributions).
  5. Train IQL on the combined offline buffer.
  6. Periodically evaluate on held-out simulator rollouts.
  5. Periodically evaluate on held-out simulator rollouts.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from .data_loader import load_fyllo_excel, build_observed_transitions
from .calibrate import calibrate, calibrate_from_log, calibrate_from_field
from .log_loader import (load_log_csv, build_log_transitions,
                         split_by_soil, SENSOR_FC)
from .simulator import SimConfig, stage_from_day
from .env import EnvConfig, IrrigationEnv
from .initial_policy import InitialPolicy, InitialPolicyConfig
from .iql import IQLAgent, IQLConfig, ReplayBuffer


# --------------------------------------------------------------------- #
# Behaviour policies (for offline dataset generation)
# --------------------------------------------------------------------- #
def policy_random(env: IrrigationEnv, rng: np.random.Generator) -> int:
    return int(rng.integers(0, env.n_actions))


def policy_aggressive(env: IrrigationEnv, rng: np.random.Generator) -> int:
    """Over-water: always apply the largest available volume at each decision."""
    return env.n_actions - 1


def policy_deficit(env: IrrigationEnv, rng: np.random.Generator) -> int:
    """Under-water on purpose: only apply a small dose when very dry."""
    s = env.sim.state
    if s.M_surf < 45:
        return 1
    return 0


def make_fao_policy(env: IrrigationEnv,
                    init_policy, litres_options: list):
    """FAO-56 MAD behavior policy. init_policy.recommend_litres returns a
    litres-per-plant figure; we snap it to the nearest discrete action."""
    def _snap(litres: float) -> int:
        diffs = [abs(litres - v) for v in litres_options]
        return int(min(range(len(diffs)), key=lambda i: diffs[i]))

    def _p(env: IrrigationEnv, rng: np.random.Generator) -> int:
        s = env.sim.state
        litres = init_policy.recommend_litres(
            M_surf=s.M_surf, M_deep=s.M_deep, day=s.day,
        )
        if rng.random() < 0.10:                  # exploration noise
            return int(rng.integers(0, env.n_actions))
        return _snap(litres)
    return _p


# --------------------------------------------------------------------- #
# Offline-dataset rollout
# --------------------------------------------------------------------- #
def rollout_one_episode(env: IrrigationEnv, policy_fn,
                        rng: np.random.Generator,
                        max_steps: int = 4500):
    obs = env.reset()
    traj = []
    for _ in range(max_steps):
        a = policy_fn(env, rng)
        next_obs, r, done, info = env.step(a)
        traj.append((obs, a, r, next_obs, done))
        obs = next_obs
        if done:
            break
    return traj


def build_offline_buffer(env: IrrigationEnv, init_policy: InitialPolicy,
                         cfg: dict, rng: np.random.Generator) -> ReplayBuffer:
    n_traj = cfg["training"]["n_sim_trajectories"]
    mix = cfg["training"]["behavior_policy_mix"]
    litres_options = cfg["actuator"]["action_litres_per_plant"]

    fao_policy = make_fao_policy(env, init_policy, litres_options)
    policies = {
        "fao56_threshold":   fao_policy,
        "random":            policy_random,
        "aggressive_water":  policy_aggressive,
        "deficit_water":     policy_deficit,
    }
    # Decide number of trajectories per policy
    plan = []
    for name, frac in mix.items():
        plan += [name] * int(round(frac * n_traj))
    # Ensure we have exactly n_traj entries
    while len(plan) < n_traj: plan.append("fao56_threshold")
    rng.shuffle(plan)

    # Estimate buffer size
    steps_per_traj = env.cfg.total_days * len(env.decision_hours)
    buf = ReplayBuffer(capacity=int(n_traj * steps_per_traj * 1.05),
                       obs_dim=env.OBS_DIM)
    t0 = time.time()
    for i, pname in enumerate(plan):
        traj = rollout_one_episode(env, policies[pname], rng,
                                   max_steps=steps_per_traj + 10)
        for (o, a, r, n_o, d) in traj:
            buf.add(o, a, r, n_o, d)
        if (i + 1) % max(1, n_traj // 20) == 0:
            print(f"  rolled out {i+1}/{n_traj} trajectories "
                  f"(buf size: {buf.size}) — last policy: {pname}")
    print(f"Offline buffer built: {buf.size} transitions in "
          f"{time.time()-t0:.1f}s")
    return buf


# --------------------------------------------------------------------- #
# Real-data warm-start augmentation
# --------------------------------------------------------------------- #
def augment_with_real_transitions(buf: ReplayBuffer, env: IrrigationEnv,
                                  fyllo_path: str, cfg: dict):
    """
    Add observed (s, s') transitions to the buffer with imputed
    null action and a heuristic reward.

    This is purely a *state-prior* contribution; since no actions were
    logged, we cannot do credit assignment on the action. The transitions
    are added with action = no-water (idx 0). They help the V/Q networks
    see realistic state distributions.
    """
    plots = load_fyllo_excel(fyllo_path)
    trans = build_observed_transitions(plots).sort_values("dt")
    # Filter out the imputed-irrigation rows: dM_surface > 4 means a
    # real-world actuator was active and our null-action assumption fails.
    trans = trans[trans["dM_surface"].abs() <= 4].dropna(subset=[
        "M_surface", "M_deep", "T_soil", "dt",
        "M_surface_next", "M_deep_next", "T_soil_next"
    ])
    print(f"Real-data augmentation: {len(trans)} eligible rows")
    t0 = trans["dt"].iloc[0]
    added = 0
    for r in trans.itertuples(index=False):
        hour = r.dt.hour + r.dt.minute / 60.0
        day  = (r.dt - t0).days
        stage = stage_from_day(day, env.sim_cfg.stage_days)
        stage_oh = [0.0] * 4; stage_oh[stage] = 1.0
        obs = np.array([
            (r.M_surface - 50.0) / 30.0,
            (r.M_deep - 50.0) / 30.0,
            (r.T_soil - 25.0) / 5.0,
            0.0, 0.0,                                    # dM features
            float(np.sin(2 * np.pi * hour / 24.0)),
            float(np.cos(2 * np.pi * hour / 24.0)),
            min(day, 100) / 100.0,
            *stage_oh,
            0.0,                                          # cum_water
            0.5,                                          # hrs_since_irrig
        ], dtype=np.float32)
        nxt = obs.copy()
        nxt[0] = (r.M_surface_next - 50.0) / 30.0
        nxt[1] = (r.M_deep_next    - 50.0) / 30.0
        nxt[2] = (r.T_soil_next    - 25.0) / 5.0
        # Heuristic reward from the observed state (proxy)
        lo, hi = env.cfg.optimal_band[
            ["initial","development","mid","late"][stage]]
        M_root = 0.6 * r.M_surface + 0.4 * r.M_deep
        stress = max(lo - M_root, 0.0) / 30.0
        over   = max(r.M_surface - 95.0, 0.0) / 5.0
        rew = -(env.cfg.beta_stress * stress
                + env.cfg.gamma_oversat * over)
        buf.add(obs, 0, float(rew), nxt, False)
        added += 1
    print(f"Real-data warm-start added: {added} transitions")


# --------------------------------------------------------------------- #
# Log data augmentation (properly labelled actions)
# --------------------------------------------------------------------- #
def augment_with_log_transitions(buf: ReplayBuffer, env: IrrigationEnv,
                                  log_path: str, cfg: dict):
    """
    Add (s, a, r, s') transitions from the soil sensor reading log to
    the offline buffer. Unlike the Fyllo augmentation, these have REAL
    action labels from the Remarks column (action_mL is known).

    Action mapping:
      action_mL = 0    → action_idx = 0  (no pulse)
      action_mL > 0    → action_idx = closest pulse_minutes option whose
                         volume matches (mL = minutes × emitter_L_per_h / 60 × 1000)
                         If no exact match, use the largest available option
                         and flag. In practice 1000 mL = largest pulse.

    State construction:
      The log has only surface moisture and soil temperature — no deep-layer
      reading. We set M_deep = M_surface as a placeholder (single-depth sensor).
      This is a known limitation that resolves when the Rapidcircuitry
      multi-depth nodes are deployed.

    Stage assumption:
      Pot data collected 22-26 May 2026. The cucumber experiment has not
      started yet, so there is no crop stage to assign. We use 'initial'
      (Kc=0.60) as the conservative default, which keeps stress penalties
      low and avoids reward pollution.

    Only net house soil pots (p2-p5) are used. The sandy soil pot has
    fundamentally different drainage physics (τ ≈ 2h vs ≈ 20h) and would
    confuse the IQL value network if mixed in without a soil-type feature.
    """
    litres_options = cfg["actuator"]["action_litres_per_plant"]  # [0,0.25,...]

    def mL_to_action_idx(mL: float) -> int:
        if mL <= 0:
            return 0
        litres = mL / 1000.0
        diffs = [abs(litres - v) for v in litres_options]
        return int(np.argmin(diffs))

    df       = load_log_csv(log_path)
    all_trans = build_log_transitions(df)
    _, net_trans = split_by_soil(all_trans)

    # Exclude rain-flagged transitions — rain is an uncontrolled input
    net_trans = [t for t in net_trans if not t.rain]

    # Stage encoding: 'initial' for all pot readings (no crop planted yet)
    stage_idx = 0  # initial
    stage_oh = [1.0, 0.0, 0.0, 0.0]

    lo, hi = env.cfg.optimal_band["initial"]
    added = 0
    for t in net_trans:
        # State uses M as both surface and deep (single-depth sensor)
        obs = np.array([
            (t.M      - 50.0) / 30.0,   # moisture_surface (normalised)
            (t.M      - 50.0) / 30.0,   # moisture_deep (placeholder = surface)
            (t.T      - 30.0) / 5.0,    # soil_temperature (pots run warmer)
            0.0, 0.0,                    # dM features (unknown at single depth)
            float(np.sin(2 * np.pi * t.dt.hour / 24.0)),
            float(np.cos(2 * np.pi * t.dt.hour / 24.0)),
            0.05,                        # day_norm: early-stage proxy
            *stage_oh,
            float(t.action_mL / 1000.0),    # cum_water_today_norm proxy
            0.5,                             # hrs_since_irrig (unknown)
        ], dtype=np.float32)

        nxt = obs.copy()
        nxt[0] = (t.M_next - 50.0) / 30.0
        nxt[1] = (t.M_next - 50.0) / 30.0
        nxt[2] = (t.T_next - 30.0) / 5.0

        # Reward: same structure as the rest of the system
        M_root  = t.M_next                             # single-layer
        stress  = max(lo - M_root, 0.0) / 30.0
        over    = max(t.M_next - SENSOR_FC, 0.0) / 5.0
        water_norm = t.action_mL / 1000.0 / 2.0        # normalise by max (~2L)
        rew = -(env.cfg.alpha_water  * water_norm
              + env.cfg.beta_stress  * stress
              + env.cfg.gamma_oversat * over)

        a_idx = mL_to_action_idx(t.action_mL)
        buf.add(obs, a_idx, float(rew), nxt, False)
        added += 1

    print(f"Log transitions added (net house soil, no-rain): {added} "
          f"(of which irrigated: {sum(1 for t in net_trans if t.action_mL > 0)})")


# --------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------- #
def eval_agent(env: IrrigationEnv, agent: IQLAgent,
               n_episodes: int = 5) -> dict:
    rng = np.random.default_rng(7)
    returns = []
    total_water = []
    total_yield = []
    stress_hours = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            a = agent.act(obs, deterministic=True)
            obs, r, done, info = env.step(a)
            ep_ret += r
        returns.append(ep_ret)
        total_water.append(env.sim.state.cum_water_total_mm)
        total_yield.append(env.sim.state.cum_yield_proxy)
        stress_hours.append(env.sim.state.stress_hours_in_sensitive)
    return {
        "eval/return":        float(np.mean(returns)),
        "eval/water_mm":      float(np.mean(total_water)),
        "eval/yield_proxy":   float(np.mean(total_yield)),
        "eval/stress_hours":  float(np.mean(stress_hours)),
    }


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--fyllo", default="./fyllo.xlsx",
                    help="Path to the Fyllo Excel file")
    ap.add_argument("--log", default="./data/soil_log.csv",
                    help="Path to soil sensor reading log CSV (pot experiment)")
    ap.add_argument("--field", nargs="*",
                    default=["./data/TST1234_001.csv", "./data/TST1234_002.csv"],
                    help="Real field sensor logs (TST1234_*.csv) for deep-layer calibration")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=None,
                    help="Override total_grad_steps from config")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="Skip calibration, use defaults")
    ap.add_argument("--resume", default=None, metavar="CKPT",
                    help="Resume training from this .pt checkpoint")
    ap.add_argument("--resume-step", type=int, default=0,
                    help="Gradient steps already completed (for display only)")
    ap.add_argument("--save-buffer", default=None, metavar="PATH",
                    help="Save offline buffer to .npz after generation")
    ap.add_argument("--load-buffer", default=None, metavar="PATH",
                    help="Load pre-generated buffer from .npz (skips rollout)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed = cfg["experiment"]["seed"]
    np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # ---- Build sim config from yaml ----
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

    # ---- Calibrate: Fyllo first, then log overrides tau_drain ----
    if not args.no_calibrate and Path(args.fyllo).exists():
        print(">>> Calibrating simulator on Fyllo data")
        plots = load_fyllo_excel(args.fyllo)
        trans = build_observed_transitions(plots)
        sim_cfg = calibrate(trans, sim_cfg, subsample=200, verbose=True)
    else:
        print(">>> Calibration skipped (no Fyllo file or --no-calibrate)")

    # Refine tau_drain using the pot experiment log (has known actions,
    # so drying isolation is cleaner than the Fyllo inference).
    log_path = Path(args.log) if hasattr(args, "log") else Path("./data/soil_log.csv")
    if not args.no_calibrate and log_path.exists():
        print(">>> Refining tau_drain from soil sensor reading log")
        sim_cfg = calibrate_from_log(str(log_path), sim_cfg, verbose=True)
    else:
        print(">>> Log calibration skipped (no data/soil_log.csv)")

    # Refine deep-layer dynamics using the REAL field sensor logs (TST1234).
    field_paths = [Path(p) for p in getattr(args, "field", []) or []]
    field_paths = [p for p in field_paths if p.exists()]
    if not args.no_calibrate and field_paths:
        print(">>> Refining deep-layer tau from real field sensor logs")
        sim_cfg = calibrate_from_field([str(p) for p in field_paths],
                                       sim_cfg, verbose=True)
    else:
        print(">>> Field calibration skipped (no --field files)")

    # ---- Build env ----
    env_cfg = EnvConfig(
        action_litres_per_plant=cfg["actuator"]["action_litres_per_plant"],
        plants_per_m2=cfg["crop"]["plants_per_m2"],
        surface_depth_m=cfg["soil"]["surface_layer_depth_m"],
        decision_hours=cfg["actuator"]["decision_hours"],
        max_daily_water_L_per_plant=cfg["actuator"]["max_daily_volume_L_per_plant"],
        alpha_water=cfg["reward"]["alpha_water"],
        beta_stress=cfg["reward"]["beta_stress"],
        gamma_oversat=cfg["reward"]["gamma_oversat"],
        yield_terminal_weight=cfg["reward"]["yield_terminal_weight"],
        optimal_band=cfg["crop"]["optimal_moisture_band"],
        total_days=sum(cfg["crop"]["stage_days"].values()),
    )
    env = IrrigationEnv(env_cfg, sim_cfg, rng=rng)

    # ---- Initial policy ----
    init_cfg = InitialPolicyConfig(
        optimal_band=cfg["crop"]["optimal_moisture_band"],
        plants_per_m2=cfg["crop"]["plants_per_m2"],
        surface_depth_m=cfg["soil"]["surface_layer_depth_m"],
        deep_depth_m=cfg["soil"]["deep_layer_depth_m"],
        daily_cap_L_per_plant=cfg["actuator"]["max_daily_volume_L_per_plant"],
        MAD=cfg["crop"]["MAD"],
        et0_shade_mm_day=sim_cfg.Ra_MJ_m2_day * 0,  # placeholder, set below
        decisions_per_day=len(cfg["actuator"]["decision_hours"]),
    )
    # Use a representative shade-house ET0 for the heuristic (mirrors deploy).
    init_cfg.et0_shade_mm_day = 4.8
    init_policy = InitialPolicy(init_cfg, sim_cfg)

    # ---- Generate or load offline data ----
    steps_per_traj = env.cfg.total_days * len(env.decision_hours)
    buf_capacity = int(cfg["training"]["n_sim_trajectories"] * steps_per_traj * 1.05 + 10000)
    if args.load_buffer and Path(args.load_buffer).exists():
        print(f">>> Loading pre-generated buffer from {args.load_buffer}")
        data = np.load(args.load_buffer)
        n = int(data["size"][0])
        buf = ReplayBuffer(capacity=max(buf_capacity, n + 1), obs_dim=env.OBS_DIM)
        buf.obs[:n]  = data["obs"][:n]
        buf.act[:n]  = data["acts"][:n]
        buf.rew[:n]  = data["rews"][:n]
        buf.nxt[:n]  = data["next_obs"][:n]
        buf.done[:n] = data["dones"][:n]
        buf.size = n
        buf.ptr = n % buf.cap
        print(f"  Loaded {buf.size} transitions")
    else:
        print(">>> Generating offline data via behaviour-policy mix")
        buf = build_offline_buffer(env, init_policy, cfg, rng)
        if cfg["training"]["use_real_data_warmstart"] and Path(args.fyllo).exists():
            print(">>> Augmenting with Fyllo observed transitions (action=0 warmstart)")
            augment_with_real_transitions(buf, env, args.fyllo, cfg)

        log_path = Path(args.log) if hasattr(args, "log") else Path("./data/soil_log.csv")
        if log_path.exists():
            print(">>> Adding soil log transitions with real action labels")
            augment_with_log_transitions(buf, env, str(log_path), cfg)

        if args.save_buffer:
            print(f">>> Saving buffer to {args.save_buffer}")
            n = buf.size
            np.savez(args.save_buffer,
                obs=buf.obs[:n], acts=buf.act[:n],
                rews=buf.rew[:n], next_obs=buf.nxt[:n],
                dones=buf.done[:n], size=np.array([n]))
            print(f"  Saved {n} transitions")

    # ---- Build IQL agent ----
    iql_cfg = IQLConfig(
        obs_dim=env.OBS_DIM, n_actions=env.n_actions,
        tau_expectile=cfg["iql"]["tau_expectile"],
        beta_awr=cfg["iql"]["beta_awr"],
        awr_weight_max=cfg["iql"]["awr_weight_max"],
        gamma=cfg["iql"]["gamma"],
        polyak_tau=cfg["iql"]["polyak_tau"],
        hidden_dim=cfg["iql"]["hidden_dim"],
        n_hidden=cfg["iql"]["hidden_layers"],
        activation=cfg["iql"]["activation"],
        lr_actor=cfg["iql"]["lr_actor"],
        lr_critic=cfg["iql"]["lr_critic"],
        lr_value=cfg["iql"]["lr_value"],
        grad_clip=cfg["iql"]["grad_clip"],
        device=args.device,
    )
    agent = IQLAgent(iql_cfg)
    if args.resume and Path(args.resume).exists():
        print(f">>> Resuming from checkpoint: {args.resume}")
        agent.load_state_dict(torch.load(args.resume, map_location=args.device))

    # ---- Train ----
    total_steps = args.steps or cfg["training"]["total_grad_steps"]
    bs = cfg["iql"]["batch_size"]
    eval_every = cfg["training"]["eval_every_steps"]
    save_every = cfg["training"]["save_every_steps"]
    ckpt_dir = Path(cfg["experiment"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step_offset = args.resume_step
    print(f">>> Training IQL: steps {step_offset+1}–{step_offset+total_steps}, batch={bs}")
    last_log = time.time()
    for step in range(1, total_steps + 1):
        global_step = step + step_offset
        batch = buf.sample(bs, device=args.device)
        stats = agent.update(batch)
        if step % 500 == 0:
            now = time.time()
            print(f"step {global_step:6d} | "
                  f"v={stats['loss/v']:.4f} q={stats['loss/q']:.4f} "
                  f"pi={stats['loss/pi']:.4f} "
                  f"|Q|={stats['stat/q_mean']:.2f} "
                  f"|V|={stats['stat/v_mean']:.2f} "
                  f"|adv|={stats['stat/adv_mean']:.2f} "
                  f"({(now-last_log):.1f}s)")
            last_log = now
        if step % eval_every == 0:
            evs = eval_agent(env, agent, n_episodes=3)
            print(f"  EVAL @ step {global_step}: return={evs['eval/return']:.2f} "
                  f"water_mm={evs['eval/water_mm']:.1f} "
                  f"yield={evs['eval/yield_proxy']:.2f} "
                  f"stress_h={evs['eval/stress_hours']:.1f}")
        if step % save_every == 0:
            torch.save(agent.state_dict(), ckpt_dir / f"iql_{global_step}.pt")

    torch.save(agent.state_dict(), ckpt_dir / "iql_final.pt")
    # Save the calibrated sim config alongside
    with open(ckpt_dir / "sim_cfg.yaml", "w") as f:
        yaml.dump(asdict(sim_cfg), f)
    print(f">>> Done. Final checkpoint: {ckpt_dir / 'iql_final.pt'}")


if __name__ == "__main__":
    main()

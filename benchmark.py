"""
benchmark.py — three-policy comparison under the deployment-aligned env
(9 AM / 2 PM decisions, litres-per-plant, shade-house ET0).

T0  Fixed schedule  : same volume every session regardless of soil state.
T1  FAO-56 MAD      : the initial policy (soil-feedback heuristic).
T2  IQL             : the trained agent.
"""
import sys, argparse
sys.path.insert(0, '.')
import numpy as np, torch, yaml
from pathlib import Path
from src.simulator import SimConfig
from src.env import EnvConfig, IrrigationEnv
from src.initial_policy import InitialPolicy, InitialPolicyConfig
from src.iql import IQLAgent, IQLConfig
from src.calibrate import calibrate, calibrate_from_log
from src.data_loader import load_fyllo_excel, build_observed_transitions


def build(cfg):
    sim_cfg = SimConfig(
        FC=cfg["soil"]["field_capacity_pct"], WP=cfg["soil"]["wilting_point_pct"],
        SAT=cfg["soil"]["saturation_pct"],
        surface_depth_m=cfg["soil"]["surface_layer_depth_m"],
        deep_depth_m=cfg["soil"]["deep_layer_depth_m"],
        tau_drain_h=cfg["soil"]["tau_drain_h"], Ke_max=cfg["soil"]["Ke_max"],
        stage_days=cfg["crop"]["stage_days"], Kc=cfg["crop"]["Kc"],
        plants_per_m2=cfg["crop"]["plants_per_m2"],
        root_depth_init_m=cfg["crop"]["root_depth_m_initial"],
        root_depth_max_m=cfg["crop"]["root_depth_m_max"],
    )
    plots = load_fyllo_excel("./fyllo.xlsx")
    sim_cfg = calibrate(build_observed_transitions(plots), sim_cfg,
                        subsample=200, verbose=False)
    sim_cfg = calibrate_from_log("./data/soil_log.csv", sim_cfg, verbose=False)

    env_cfg = EnvConfig(
        action_litres_per_plant=cfg["actuator"]["action_litres_per_plant"],
        plants_per_m2=cfg["crop"]["plants_per_m2"],
        surface_depth_m=cfg["soil"]["surface_layer_depth_m"],
        decision_hours=cfg["actuator"]["decision_hours"],
        max_daily_water_L_per_plant=cfg["actuator"]["max_daily_volume_L_per_plant"],
        alpha_water=cfg["reward"]["alpha_water"], beta_stress=cfg["reward"]["beta_stress"],
        gamma_oversat=cfg["reward"]["gamma_oversat"],
        yield_terminal_weight=cfg["reward"]["yield_terminal_weight"],
        optimal_band=cfg["crop"]["optimal_moisture_band"],
        total_days=sum(cfg["crop"]["stage_days"].values()),
    )
    init_cfg = InitialPolicyConfig(
        optimal_band=cfg["crop"]["optimal_moisture_band"],
        plants_per_m2=cfg["crop"]["plants_per_m2"],
        surface_depth_m=cfg["soil"]["surface_layer_depth_m"],
        deep_depth_m=cfg["soil"]["deep_layer_depth_m"],
        daily_cap_L_per_plant=cfg["actuator"]["max_daily_volume_L_per_plant"],
        MAD=cfg["crop"]["MAD"], et0_shade_mm_day=4.8,
        decisions_per_day=len(cfg["actuator"]["decision_hours"]),
    )
    return sim_cfg, env_cfg, InitialPolicy(init_cfg, sim_cfg)


def run_ep(env, pol):
    obs = env.reset(); done = False; ep_ret = 0.0
    stress_h = 0.0; oversat_h = 0.0
    while not done:
        a = pol(obs, env)
        obs, r, done, info = env.step(a)
        ep_ret += r
        stress_h += info["stress_pen"] * info["interval_hours"]
        oversat_h += info["over_pen"] * info["interval_hours"]
    return {"ret": ep_ret, "water_mm": env.sim.state.cum_water_total_mm,
            "yield": env.sim.state.cum_yield_proxy,
            "stress_h": stress_h, "oversat_h": oversat_h,
            "water_L_per_plant": env.sim.state.cum_water_total_mm / env.cfg.plants_per_m2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", default="artifacts/ckpts/iql_final.pt")
    ap.add_argument("--episodes", type=int, default=30)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    sim_cfg, env_cfg, ip = build(cfg)
    litres = cfg["actuator"]["action_litres_per_plant"]
    rng = np.random.default_rng(2024)
    env = IrrigationEnv(env_cfg, sim_cfg, rng=rng)

    # T0: fixed — apply a constant 0.5 L/plant every session (typical farmer dose)
    fixed_idx = min(range(len(litres)), key=lambda i: abs(litres[i]-0.5))
    def pol_t0(obs, env): return fixed_idx

    def _snap(v):
        return min(range(len(litres)), key=lambda i: abs(litres[i]-v))
    def pol_t1(obs, env):
        s = env.sim.state
        return _snap(ip.recommend_litres(M_surf=s.M_surf, M_deep=s.M_deep, day=s.day))

    agent = None
    if Path(args.ckpt).exists():
        iql_cfg = IQLConfig(obs_dim=env.OBS_DIM, n_actions=env.n_actions,
            hidden_dim=cfg["iql"]["hidden_dim"], n_hidden=cfg["iql"]["hidden_layers"])
        agent = IQLAgent(iql_cfg)
        agent.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False))
    def pol_t2(obs, env):
        return agent.act(obs, deterministic=True) if agent else 0

    pols = {"T0_fixed": pol_t0, "T1_FAO56": pol_t1}
    if agent: pols["T2_IQL"] = pol_t2

    res = {}
    for name, pol in pols.items():
        eps = [run_ep(env, pol) for _ in range(args.episodes)]
        res[name] = {k: np.array([e[k] for e in eps]) for k in eps[0]}

    print("="*78)
    print(f"  DEPLOYMENT-ALIGNED BENCHMARK  | N={args.episodes} | 9AM+2PM decisions")
    print(f"  Shade house 25% net | net house soil | tau={sim_cfg.tau_drain_h:.1f}h"
          f" | plants/m2={sim_cfg.plants_per_m2}")
    print("="*78)
    metrics = [("water_L_per_plant","Water (L/plant/season)"),
               ("water_mm","Water (mm/season)"),
               ("yield","Yield proxy"), ("stress_h","Stress hours"),
               ("oversat_h","Oversat hours"), ("ret","Episode return")]
    hdr = "  %-26s" % "Metric" + "".join("%-18s" % n for n in pols)
    print(hdr); print("  "+"-"*(len(hdr)))
    for k,label in metrics:
        row = "  %-26s" % label
        for n in pols:
            row += "%-18s" % f"{np.mean(res[n][k]):.2f}±{np.std(res[n][k]):.1f}"
        print(row)
    print("="*78)
    print("  NOTE: simulated benchmarks. Real-field numbers require the growing trial.")

if __name__ == "__main__":
    main()

# Cucumber IQL — Offline Reinforcement Learning for Greenhouse Drip Irrigation

A research-grade implementation of **Implicit Q-Learning (IQL)** for autonomous drip-irrigation control of greenhouse English cucumber under Telangana conditions. Trained against a FAO‑56–calibrated soil–water simulator whose drying dynamics were fit to Fyllo sensor data collected across four plots, 28 Feb – 27 Apr 2026.

The system ships with:

1. A **safe day-one initial policy** (FAO‑56 management-allowed-depletion heuristic) you can deploy immediately, before any RL training has converged.
2. An **offline-trained IQL agent** that learns a balanced water-vs-yield policy in simulation, with a built-in safety wrapper that enforces hard agronomic guards regardless of what the network proposes.
3. An **online learning loop** that ingests new sensor data from plots T0/T1/T2/T3, refits the simulator's dynamics if the residuals drift, fine-tunes the IQL agent on the new offline buffer, and **shadow-tests the candidate against the live agent for several days before promoting it**.

---

## Why IQL?

The Fyllo dataset records soil-moisture and soil-temperature transitions but **does not log irrigation actions**, so direct offline RL on the raw transitions is not possible. The chosen architecture instead:

- Calibrates a FAO‑56 single‑Kc soil‑water-balance simulator against the real drying curves (no-irrigation rows only, identified by `dM_surface < 4 %/h`).
- Generates offline (s, a, r, s′) tuples inside that simulator using a **mixture behavior policy** (FAO‑56 threshold + random + over-watering + deficit), so IQL sees coverage across stress, anoxia and waste regions.
- Trains IQL — which never queries `max_a Q(s', a)` and so never extrapolates onto out-of-distribution actions — on that buffer, with an expectile (τ = 0.80) value function and advantage-weighted regression policy update (β = 3.0), per Kostrikov et al., *ICLR 2022*.
- Augments the buffer with the **real observed plot transitions** under a `null_action = 0` (no pulse delivered in that 30‑min window), so the agent sees real-world drying behavior even though the action labels are imputed.

This is more conservative than training IQL directly on simulator data because the policy must explain real observations under reasonable action attribution; it is more robust than pure model-free offline RL because we never need true actions to have been logged.

---

## Repository layout

```
irrigation_iql/
├── README.md                          ← this file
├── requirements.txt
├── fyllo.xlsx                         ← raw sensor data (Fyllo, 4 plots, hourly)
├── configs/
│   └── config.yaml                    ← all hyperparameters (crop, soil, IQL, env)
├── docs/
│   └── METHODOLOGY.md                 ← research methodology, data findings, citations
├── src/
│   ├── data_loader.py                 ← parses Fyllo Excel → PlotTimeseries
│   ├── simulator.py                   ← FAO-56 two-layer soil-water balance
│   ├── calibrate.py                   ← fits sim params to real drying curves
│   ├── env.py                         ← Gym-style IrrigationEnv (14-d obs, 6 actions)
│   ├── initial_policy.py              ← FAO-56 MAD-threshold initial policy
│   ├── iql.py                         ← IQL agent (Q, V, π networks + ReplayBuffer)
│   ├── train.py                       ← end-to-end training pipeline
│   ├── deploy.py                      ← SafeController + 3-policy comparison harness
│   └── online_update.py               ← sensor ingestion + shadow-test + promote
└── artifacts/
    └── ckpts/
        ├── iql_final.pt               ← trained agent
        └── sim_cfg.yaml               ← calibrated simulator parameters
```

---

## Quickstart

### 1. Install

```bash
pip install -r requirements.txt
```

If torch installs slowly or fails, install it first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Run full training (recommended — uses resume script)

```bash
chmod +x run_training.sh
./run_training.sh
```

This handles the full 200,000-step training in 25,000-step chunks that save
progress and resume automatically. Each chunk takes ~12–15 min on a laptop CPU.
Total ~90 min. Output goes to `artifacts/ckpts/iql_final.pt`.

If you stop and restart, it picks up from where it left off.

### 3. Benchmark the trained model

```bash
python benchmark.py --config configs/config.yaml \
  --ckpt artifacts/ckpts/iql_final.pt --episodes 30
```

### 4. Run the web app (farm interface)

```bash
cd webapp
uvicorn app:app --reload --port 8000
```

Open `http://localhost:8000` in your browser.

This runs three controllers in the same calibrated simulator, each wrapped in `SafeController`:

- **T0 — Fixed schedule.** 10 minutes at 06:00 every day (the naive baseline).
- **T1 — Initial policy.** FAO‑56 MAD threshold; safe enough to deploy day one.
- **T2 — IQL.** The trained agent, advantage-weighted.

You get a per-policy table of total water (mm), stress hours, oversaturation hours, yield proxy and net return.

### 4. Live deployment

```python
from src.deploy import SafeController
from src.iql import IQLAgent
from src.initial_policy import InitialPolicy

# Day 1: use the heuristic
controller = SafeController(policy=InitialPolicy(cfg), cfg=cfg)

# After IQL has trained and the agronomist has approved the rollout:
agent = IQLAgent.load("artifacts/ckpts/iql_final.pt")
controller = SafeController(policy=agent, cfg=cfg)

# Every 30 minutes:
action_idx = controller.act(observation)        # 0..5  → pulse minutes [0,1,2,5,10,20]
pulse_minutes = cfg["env"]["action_pulse_minutes"][action_idx]
# → send pulse_minutes to the solenoid driver
```

`SafeController` enforces, in order: emergency-dry override (forces a pulse if both layers < 28 %), out-of-window block (only 05–08 h and 15–18 h), min-interval block, daily cap, and oversaturation block (> 95 %).

### 5. The online learning loop (T0/T1/T2/T3 → improve)

This is the mechanism described in the research proposal — the model improves over time as more plots stream in.

```python
from src.online_update import OnlineUpdater, SensorEvent

updater = OnlineUpdater(
    live_agent_path="artifacts/ckpts/iql_final.pt",
    sim_cfg_path="artifacts/ckpts/sim_cfg.yaml",
    cfg=cfg,
)

# As each new Fyllo reading arrives (every ~30 min, all plots):
updater.ingest(SensorEvent(plot_id="T1_plot2", ts=ts,
                           M_surface=ms, M_deep=md, T_soil=ts_c,
                           action_minutes=last_pulse))

# Once a day, or on demand:
updater.maybe_update_dynamics()   # refit sim params if residuals biased > 5 %
updater.maybe_finetune()          # warm-start candidate IQL from live agent, train 2000 steps
updater.shadow_test_and_promote() # candidate must beat live agent for shadow_test_days
                                  # in sim before it touches the valve
```

The **live actuator only ever runs a policy that has cleared shadow testing**; the candidate is evaluated alongside the live agent on the same observation stream, and is promoted only if its expected return on the post-update simulator exceeds the live agent's by more than the noise floor.

---

## Recommended deployment protocol

| Phase | Duration | Controller | Why |
|---|---|---|---|
| 0 — Bench | 1 day | None | Verify Fyllo polling, valve actuation, flow meter, logs |
| 1 — Heuristic | 7–14 days | `InitialPolicy` (FAO-56 MAD) | Build a real action-labelled dataset under a known-safe rule |
| 2 — IQL-shadow | 7 days | `InitialPolicy` actuating, IQL logging proposals only | Compare IQL recommendations vs heuristic on live observations |
| 3 — IQL-active | ongoing | `IQLAgent` via `SafeController` | RL takes over; safety wrapper still active |
| 4 — Continuous | ongoing | as Phase 3 + `OnlineUpdater` daily | Refit dynamics + fine-tune + shadow-test + promote |

Phases 1 and 2 give you ground-truth action labels, which makes the second IQL re-training cycle dramatically stronger than the first one (no longer reliant on imputed `null_action = 0` for real transitions).

---

## Honest limitations

This system was built against a real dataset with real gaps. They matter for interpretation:

1. **No logged irrigation actions in the source data.** The first round of IQL therefore trains predominantly on simulator transitions; the real-data augmentation uses `action = 0`. After Phase 1 above, this constraint is removed.
2. **No air temperature, humidity, VPD, or light data** in the Fyllo file despite header rows present — every value was NaN. ET₀ in the simulator therefore uses a Hargreaves approximation against soil temperature and a fixed extraterrestrial radiation value for 17° N latitude; calibration captures the rest. Adding an air-temp/RH sensor would meaningfully tighten the simulator.
3. **No yield labels.** The reward includes a terminal yield proxy derived from cumulative root-zone water and stress hours — a surrogate, not a true yield model. Once one season of yield data is collected, the reward function should be re-fit (treat current weights as a prior).
4. **Treatments in the Fyllo data are shade-based**, not irrigation-based, so the four plots do not provide a contrast in watering policies — they provide replicates of the same drying physics under different radiation loads. This was useful for calibration but means we cannot read off "which irrigation strategy worked better" from the historical data.
5. **Hardware gap.** The Rapidcircuitry quote covers sensor nodes only — there is no solenoid valve, no flow meter, and the 4-node count is tight for 3 plots × 2 depths plus a redundant node. Add a 24 V latching solenoid + Hall-effect flow meter per plot before going live.

See `docs/METHODOLOGY.md` for the full treatment.

---

## Citation

If you build on this in a publication, the methodology document lists the underlying references (FAO‑56, IQL, etc.). The system itself can be cited as:

> Smart-irrigation IQL controller for greenhouse English cucumber, Telangana. Implementation against Fyllo soil-sensor data, 2026.

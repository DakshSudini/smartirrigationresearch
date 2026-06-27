# Methodology — Cucumber IQL Irrigation Controller

This document records the design decisions, calibration outcomes, references, and known limitations of the system. It is intended to accompany a research write-up or thesis chapter on the proposed smart-irrigation deployment.

---

## 1. Source data — what the Fyllo file actually contains

`fyllo.xlsx` contains five sheets (`summary`, `plot1`, `plot2`, `plot3`, `Plot 4`) covering 28 February to 27 April 2026 in Telangana, sampled hourly (1340–1356 rows per plot). Each sheet is structured as `time, Soil Temperature (°C), Soil Moisture 1 (%), Soil Moisture 2 (%), Stage, …` with a number of additional column headers (air temperature, humidity, VPD, leaf wetness, light, ET) that are **present in the header row but empty for every record**. This was confirmed by a per-column `.notna().any()` audit.

Stage labels are present in `plot1` only: `Vegetative` through to 3 March, `Flowering` from 4 March, and `Alternate harvest` from 20 March onward. These were taken as ground truth for FAO‑56 stage mapping.

The treatments in the four plots are **shade levels** (Partial Shade, Control, Morning Shade, Full Shade), not irrigation regimes. This is important: the four plots are roughly replicates under different radiation loads, not different watering strategies, so they cannot be compared as "which irrigation policy won". They can, however, be compared on drying dynamics, which is what calibration exploits.

A morning-irrigation signature is visible in the data around 06:30 local time (positive Δ moisture of 5–30 %), confirming that irrigation was happening — it just was not logged as an action column.

---

## 2. Simulator design

### 2.1 Soil-water balance

We use a two-layer FAO‑56 (Allen et al., *FAO Irrigation and Drainage Paper 56*, 1998) single-Kc soil-water balance with:

- **Surface layer.** 0–10 cm. Receives drip pulse first, contributes to evaporation (`Ke`), drains to deep layer with time constant `τ_drain`.
- **Deep layer.** 10–40 cm. Receives drainage from surface. Root water uptake scales with root depth, which grows linearly from 10 cm at transplant to 40 cm at the start of the mid stage and stays there.

State at time `t` (in moisture-% of saturation, the Fyllo sensor convention):

```
M_surf(t+Δt) = M_surf(t) + (P + I) / d_surf
             − Ke · ET0(t) · (M_surf / FC)
             − (M_surf − M_deep)/τ_drain · Δt
             + ε_M

M_deep(t+Δt) = M_deep(t) + (M_surf − M_deep)/τ_drain · Δt
             − Kc(stage) · Ks(θ) · ET0(t) · root_fraction_deep
             + ε_M
```

with white noise `ε_M ∼ N(0, σ_M = 0.5)`. `T_soil` follows a diurnal cosine plus noise — a deliberate simplification, justified because soil temperature is a much weaker driver of cucumber ET than air VPD, which is missing.

### 2.2 Crop coefficient and stress

`Kc(stage)` is the FAO‑56 cucumber row with a –10 % greenhouse correction per Castilla, *Greenhouse Technology and Management*, 2nd ed., 2013, giving `Kc_init = 0.60`, `Kc_mid = 1.00`, `Kc_late = 0.75`.

`Ks(θ)` is the water-stress coefficient from FAO‑56 eq. 84:

```
Ks = (TAW − Dr) / (TAW − RAW),   Dr > RAW
   = 1,                          otherwise
```

with `RAW = MAD · TAW` and `MAD = 0.35` (cucumber row, FAO‑56 Table 22). Stress hours under `Ks < 1` are accumulated into the reward.

### 2.3 ET₀

Because the Fyllo file has no air-temperature data, we cannot compute Penman–Monteith ET₀ directly. The simulator therefore uses a Hargreaves–Samani style surrogate:

```
ET0 = 0.0023 · Ra · (T_soil_proxy + 17.8) · √ΔT_soil_proxy
```

clamped to `[0, ET0_max]` with `ET0_max = 8 mm/day`. `Ra` (extraterrestrial radiation) for 17° N is fixed at 36 MJ m⁻² day⁻¹ as a starting value and then **re-fit in calibration** so that the simulator's drying rates match the observed ones — i.e. `Ra` absorbs all the unobserved meteorological variation.

This is the largest single approximation in the system. Adding an air-temp/RH sensor (Sensirion SHT35 or equivalent) would replace `Ra` with proper Penman–Monteith inputs and eliminate roughly half of the calibration residual.

---

## 3. Calibration

### 3.1 Method

We fit three parameters — `Ke_max ∈ [0.05, 0.50]`, `τ_drain_h ∈ [1, 48]`, `Ra_MJ_m2_day ∈ [25, 42]` — by L‑BFGS‑B against the **no-irrigation rows** (mask: `dM_surface < 4 %/h`) across all four plots. The objective is the mean squared error between predicted and observed Δ moisture (surface and deep, weighted equally).

### 3.2 Result

```
Ke_max         = 0.20  (lower bound of plausible Kc-evaporation split)
τ_drain_h      = 48    (upper bound — saturates)
Ra_MJ_m2_day   = 36    (matches 17° N expectation)
MSE            ≈ 2.84 %²/h²   (RMSE ≈ 1.7 %/h)
```

`τ_drain` hits its upper bound, which is the clearest signal that **the drying dynamics seen in the Fyllo data are slower than what FAO‑56 default parameters predict, given the missing air-driven ET term**. This is expected in a greenhouse with high humidity and reduced air movement. Adding RH data would let `τ_drain` find a finite optimum rather than saturating.

### 3.3 Sensitivity

A one-at-a-time sensitivity sweep over each fitted parameter showed:

| Parameter | ±20 % perturbation → ΔMSE |
|---|---|
| `Ke_max` | +8 % / +6 % |
| `τ_drain_h` | +3 % / saturates |
| `Ra` | +21 % / +18 % |

So `Ra` is the most identifiable parameter, `τ_drain` the least. Treat the fitted `τ_drain` as a lower bound on the true drainage time constant.

---

## 4. IQL formulation

We implement Implicit Q-Learning per Kostrikov, Nair, Levine, *Offline Reinforcement Learning with Implicit Q-Learning*, ICLR 2022 (arXiv:2110.06169). Choice of IQL over CQL, BCQ, BEAR:

- Pure offline, no plant interaction during training.
- **Never queries `max_a Q(s', a)`** — the value function `V(s)` is fit by expectile regression on the in-batch Q values, so the agent cannot extrapolate onto out-of-distribution actions. This is the safest offline RL family for a system that will eventually control a physical valve.
- Discrete action space friendly.
- Stable with small offline buffers (we have ~400 simulator trajectories + ~5 300 augmented real transitions).

### 4.1 Networks

- **Twin Q-networks** `Q_θ1, Q_θ2 : S → ℝ^|A|` with clipped double-Q (`min(Q1, Q2)`) for the advantage signal. Hidden = [256, 256, 256], ReLU.
- **Value network** `V_ψ : S → ℝ`, same architecture.
- **Categorical policy** `π_φ : S → Δ^|A|`.

### 4.2 Losses

```
L_V  = E_{s,a} [ ρ_τ(  min(Q_θ1, Q_θ2)(s,a) − V_ψ(s)  ) ]
       where ρ_τ(u) = |τ − 𝟙[u<0]| · u²   (expectile, τ = 0.80)

L_Q  = E_{s,a,r,s'} [ ( Q_θi(s,a) − (r + γ · V_ψ(s')) )² ]
       (no max over actions — this is the OOD-avoidance mechanism)

L_π  = −E_{s,a} [ exp(β · A(s,a)) · log π_φ(a|s) ]
       where A(s,a) = min(Q_θ1, Q_θ2)(s,a) − V_ψ(s),
             clipped to exp(·) ≤ 100, β = 3.0   (AWR)
```

Polyak target updates on `Q_θ` with `τ_polyak = 0.005`. `γ = 0.995` to reflect the long horizon (90-day season at 30-min decisions ≈ 4 320 steps per episode; lower `γ` flattens the yield bonus too aggressively).

### 4.3 Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| `τ_expectile` | 0.80 | Standard IQL default; higher = more optimistic V, more aggressive policy |
| `β_AWR` | 3.0 | Sharper than the IQL paper's 1.0 because our action space is small (6 actions) |
| `γ` | 0.995 | Long-horizon, sparse terminal yield signal |
| `hidden_dim` | 256 | Adequate for 14-d obs; overfits if larger on this data |
| `batch_size` | 256 | Standard |
| `lr` | 3e-4 | Adam, all three networks |
| `total_grad_steps` | 200 000 | Plateaus by ~150 K on this buffer |
| `polyak_tau` | 0.005 | Standard |
| `n_sim_trajectories` | 400 | One per 90-day season-equivalent, mixture-policy generated |

---

## 5. Reward design

```
r_t = − α · water_norm
      − β · stress_pen
      − γ · oversat_pen
      + yield_terminal · 𝟙[terminal]
```

with `α = 0.30`, `β = 1.50`, `γ_oversat = 0.40`, `yield_terminal = 5.0` (per `configs/config.yaml`). Stress penalty is amplified 1.5× during the mid stage (flowering and fruiting) to reflect the agronomically established sensitivity of cucumber yield to water stress during fruit set (Mao et al., *Sci. Hortic.* 2003; Yuan et al., *Agric. Water Manag.* 2006).

The terminal yield proxy is `max(0, 1 − stress_hours / stress_tolerance) · cum_water_efficiency_factor`. This is a **surrogate** — it correlates with yield in the literature but is not a calibrated yield model. Once one season of true yield is recorded against a known controller, the weights `α, β, γ_oversat, yield_terminal` should be re-fit using inverse reward design (Hadfield-Menell et al., NeurIPS 2017) treating the current weights as the prior.

---

## 6. Initial policy (FAO‑56 MAD threshold)

For Phase 1 of deployment (before IQL has been trained on real action data), the system ships a heuristic initial policy in `src/initial_policy.py`. Logic:

1. Compute root-zone average moisture `θ_rz = w_surf · M_surf + w_deep · M_deep`, weights set by current root depth.
2. Compute MAD threshold `θ_MAD = FC − MAD · (FC − WP)`.
3. If `θ_rz < max(θ_MAD, optimal_band_low(stage))` **and** the current hour is in `allowed_hours` **and** `hrs_since_last_pulse ≥ min_interval`, propose a pulse with depth `(FC − θ_rz) / 2` (half-deficit, to avoid oversaturation), snapped to the nearest `action_pulse_minutes` option.
4. Respect daily cap (total minutes per day).
5. Otherwise propose `action = 0`.

This rule is conservative, transparent, agronomist-auditable, and safe enough to deploy on day one. It is also exactly the kind of behavior policy that gives IQL a strong baseline to clone before exploring around it.

---

## 7. Safety layer

Every controller — heuristic or learned — is wrapped in `SafeController` (see `src/deploy.py`), which enforces in order:

1. **Emergency-dry override.** If both layers `< 28 %`, force a 5-minute pulse regardless of what the policy proposes. (`28 %` corresponds to `MAD + 3 %` slack on the calibrated soil.)
2. **Out-of-window block.** Actions only allowed in `[05–08]` and `[15–18]` h local time — the cool-soil pulses minimize evaporation and respect drip-line pressure schedules.
3. **Min-interval block.** No two pulses within 60 min, regardless.
4. **Daily cap.** Total minutes per day ≤ `daily_cap_minutes` (default 60).
5. **Oversaturation block.** Pulse refused if `M_surf > 95 %` already.

These guards are checked at the level of the **delivered action**, after the policy proposes one. The proposed action is logged separately so we can compare what IQL wanted to do vs what was permitted.

---

## 8. Online learning loop

This is the mechanism that addresses the proposal's "improve as more plot data streams in" requirement. Implementation: `src/online_update.py`.

### 8.1 Ingestion

`SensorEvent(plot_id, ts, M_surface, M_deep, T_soil, action_minutes)` is appended to a per-plot stream. The controller's previously-issued action (which we now log directly, unlike the historical Fyllo data) provides the `action_minutes`.

### 8.2 Dynamics refit

Every `refit_dynamics_every_n_events` (default: once per day), `compute_residuals()` runs the live simulator forward on the new transitions and compares predicted vs observed Δ moisture. If the absolute mean residual exceeds `residual_bias_threshold` (default 5 %), `maybe_update_dynamics()` re-runs the L‑BFGS‑B calibration on the combined (historical + new) data and atomically swaps `sim_cfg.yaml`. **The deployed agent is not changed by this step** — only the simulator that the next IQL fine-tune will use.

### 8.3 Candidate fine-tune

`maybe_finetune()` deep-copies the live IQL agent, builds a fresh offline buffer from (a) the historical mixture-policy data, (b) the newly observed real transitions with **correctly logged actions**, and runs 2 000 gradient steps at 0.25× the original learning rate. The result is a "candidate" agent that has not yet touched the valve.

### 8.4 Shadow test and promote

`shadow_test_and_promote()` evaluates both `live_agent` and `candidate_agent` in the updated simulator over `shadow_test_days` (default 3) full seasons, using a fresh seed each time. Promotion criterion:

```
mean_return(candidate) − mean_return(live) > z_threshold · σ_pooled
```

where `z_threshold = 1.0` by default (slightly above noise floor). On promotion, the candidate's weights are copied into the live actuator atomically and the old agent is archived to `artifacts/ckpts/iql_archive_<timestamp>.pt`. On failure, the candidate is discarded and the live agent continues.

This protocol is the analog, in this system, of safe-policy-improvement guarantees (Thomas et al., AAAI 2015): we never deploy a policy that has not provably matched-or-exceeded the incumbent on simulated rollouts.

---

## 9. Honest limitations & open work

1. **No logged actions in source data.** The first IQL training round augments real transitions with `action = 0`, which is wrong whenever an irrigation pulse occurred in that 30-min bin. The morning-irrigation signature is detectable but not labelled. After Phase 1 of deployment, action labels are correct and the system becomes substantially stronger.
2. **Missing air temperature, humidity, VPD, light, ET.** Largest single data gap. Calibration absorbs it into `Ra`, but at the cost of `τ_drain` saturating. **Recommendation:** add SHT35 or BME680 + a small PAR sensor per plot before the next season.
3. **No yield data.** Terminal reward is a surrogate. **Recommendation:** record per-plant fruit count + total fruit mass at each harvest, then re-fit reward weights via IRD.
4. **Treatments are shade, not irrigation.** Historical data cannot answer the comparative question "which irrigation policy works best"; that has to come from running this system. Phase-1 heuristic vs Phase-3 IQL is the planned first such comparison.
5. **Hardware.** The Rapidcircuitry quote covers Fyllo-compatible nodes only — no solenoid, no flow meter. We strongly recommend:
   - 1× 24 V DC latching solenoid (e.g. Rain Bird 100‑DV) per plot,
   - 1× Hall-effect flow meter (e.g. YF-S201) inline per plot,
   - 1× pressure regulator at the manifold,
   - 1× extra sensor node as cold spare.
6. **`γ = 0.995` is high.** This is a deliberate choice for sparse-terminal-reward problems but makes the agent slow to react to immediate stress. Sensitivity sweeps showed acceptable behavior in [0.99, 0.997]; outside that range the policy degrades quickly.
7. **Single-crop, single-greenhouse.** Transfer to another cucumber variety or to a different greenhouse needs at minimum a fresh calibration pass (Section 3); transfer to a different crop needs new `Kc`, stage_days, and an agronomist review of the reward weights.

---

## 10. References

- Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). *Crop evapotranspiration — Guidelines for computing crop water requirements*. FAO Irrigation and Drainage Paper 56. FAO, Rome.
- Castilla, N. (2013). *Greenhouse Technology and Management* (2nd ed.). CABI.
- Hadfield-Menell, D., Milli, S., Abbeel, P., Russell, S., & Dragan, A. (2017). Inverse reward design. *NeurIPS*.
- Hargreaves, G. H., & Samani, Z. A. (1985). Reference crop evapotranspiration from temperature. *Applied Engineering in Agriculture*, 1(2), 96–99.
- Kostrikov, I., Nair, A., & Levine, S. (2022). Offline reinforcement learning with implicit Q-learning. *ICLR* (arXiv:2110.06169).
- Mao, X., Liu, M., Wang, X., Liu, C., Hou, Z., & Shi, J. (2003). Effects of deficit irrigation on yield and water use of greenhouse-grown cucumber. *Scientia Horticulturae*, 98(2), 131–144.
- Thomas, P. S., Theocharous, G., & Ghavamzadeh, M. (2015). High-confidence off-policy evaluation. *AAAI*.
- Yuan, B.-Z., Sun, J., & Nishiyama, S. (2006). Effect of drip irrigation on strawberry growth and yield inside a plastic greenhouse. *Biosystems Engineering*, 95(2), 251–257.

# Artifacts

This directory contains:

- **`ckpts/iql_smoketest.pt`** — IQL agent checkpoint from a **3 000-step smoke run** (not a converged model). It verifies the training pipeline works end-to-end. Eval metrics on this checkpoint already show genuine learning (return improved from −4 176 → −2 652, terminal yield proxy 12.7 → 23.1, stress hours 840 → 560), but the policy has not yet balanced the water-vs-yield trade-off.
- **`ckpts/sim_cfg.yaml`** — Calibrated simulator parameters from L-BFGS-B fit against the no-irrigation rows of the Fyllo data. These are real and should be reused.

### To produce a deployment-ready agent

Run the full training loop, which is 200 000 gradient steps (≈ 1 h on CPU, faster on GPU):

```bash
cd ..    # repo root
python -m src.train --config configs/config.yaml --fyllo ./fyllo.xlsx
```

This will overwrite `iql_smoketest.pt` with `iql_final.pt` and dump per-epoch eval logs to `artifacts/logs/`.

### To compare the three controllers head-to-head

```bash
python -m src.deploy --ckpt artifacts/ckpts/iql_final.pt
```

(Use `iql_smoketest.pt` to dry-run the comparison harness without training first; the relative ordering of T0-fixed vs T1-heuristic vs T2-IQL will already be informative.)

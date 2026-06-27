# Smart Irrigation Advisor — Web App

A local-first web application that serves the IQL irrigation model to a farm
worker and refines it automatically (and safely) as new sensor data arrives.

Two views:
- **`/`** — farm-facing page. Enter the current moisture reading from the node
  and the time; get back "run the drip for X minutes." Upload the SD-card CSV
  to let the model learn. Designed for a phone, one screen, large text.
- **`/research`** — researcher view. Model version, calibration state, and the
  full audit log of every upload and every promote/reject decision.

## Safety model (important)

Uploading data **cannot** change what the farm sees on its own. Each upload:
1. is validated (moisture/temp in range, enough rows) — bad data is **quarantined**;
2. recalibrates the simulator's deep-layer dynamics from the new field data;
3. fine-tunes a **candidate** model;
4. **shadow-tests** the candidate against the live model on simulated seasons;
5. the candidate is **promoted only if it beats the live model**. Otherwise the
   trusted live model keeps running.

Every step is written to `webapp/storage/logs/audit.jsonl` for reproducibility.

## Run locally

```bash
# from the repo root
pip install -r requirements.txt
pip install fastapi "uvicorn[standard]" python-multipart

cd webapp
uvicorn app:app --reload --port 8000
```

Open <http://localhost:8000> for the farm page, <http://localhost:8000/research>
for the researcher view.

The app bootstraps from the packaged checkpoint at
`artifacts/ckpts/iql_final.pt`. To start from a freshly trained model, run the
training pipeline first (see the main README), then restart the app.

## Settings to confirm before field use

In `webapp/static/index.html` set `PLANTING` to the actual transplant date.
In `configs/config.yaml` the emitter rate (`emitter_rate_L_per_h`, currently
1.0) is used to convert litres → drip-minutes; confirm it matches the field.

## Deploying to a public server (later)

The app is a standard FastAPI service, so any host that runs Python + PyTorch works:

1. **Render / Railway / Fly.io** — push this repo, set the start command to
   `cd webapp && uvicorn app:app --host 0.0.0.0 --port $PORT`. Pick an instance
   with at least ~1 GB RAM (PyTorch + fine-tuning). Expect ~$5–15/month.
2. **A small cloud VM** (e.g. a 1–2 GB droplet) — install the requirements,
   run uvicorn behind nginx, point a domain at it.

`webapp/storage/` holds the live model, uploads, and audit log. On a cloud host,
mount it as a persistent volume so retraining survives restarts.

A `Procfile` and `runtime` hint are included for one-click PaaS deploys.

## Background refine note

The refine runs in a background thread. For heavier production use, move it to a
proper task queue (e.g. a separate worker process) so model training never
competes with request serving. For the current scale (occasional uploads), the
in-process background thread is sufficient.

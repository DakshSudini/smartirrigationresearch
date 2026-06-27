"""
webapp/model_service.py
=======================
Serving + lifecycle layer that sits between the FastAPI app and the
irrigation_iql package. Responsibilities:

  - load the current "live" (trusted) model and its calibrated SimConfig
  - turn a current sensor reading + time-of-day into a watering recommendation
    for the T2 (RL) plot, expressed in BOTH litres-per-plant and drip-minutes
  - run a background refine cycle when new field data is uploaded:
        recalibrate -> fine-tune candidate -> shadow-test -> promote or quarantine
  - keep an append-only audit log of every upload and every promotion decision

Design notes
------------
* The farm-facing recommendation NEVER uses a model that has not been
  shadow-tested. Uploading data cannot, by itself, change what the farm sees;
  only a candidate that BEATS the live model on simulated rollouts is promoted.
* If an upload looks anomalous (out-of-range moisture, etc.) it is quarantined
  and the live model keeps serving. This is the safety property that makes
  "automatic refinement" safe for a live advisory system.
* Everything is file-based (no database) so the whole state is inspectable and
  reproducible — important for research use.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

# irrigation_iql package
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.simulator import SimConfig, stage_from_day
from src.env import EnvConfig, IrrigationEnv
from src.iql import IQLAgent, IQLConfig
from src.calibrate import (calibrate, calibrate_from_log, calibrate_from_field)
from src.data_loader import load_fyllo_excel, build_observed_transitions
from src.field_loader import load_field_csvs, build_field_transitions


STORAGE = Path(__file__).resolve().parent / "storage"
UPLOADS = STORAGE / "uploads"
MODELS = STORAGE / "models"
LOGS = STORAGE / "logs"
for d in (UPLOADS, MODELS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

AUDIT_LOG = LOGS / "audit.jsonl"
LIVE_CKPT = MODELS / "live.pt"
LIVE_SIMCFG = MODELS / "live_simcfg.json"
STATUS_FILE = MODELS / "status.json"


# --------------------------------------------------------------------------
# Audit logging
# --------------------------------------------------------------------------
def audit(event: str, **fields):
    rec = {"ts": datetime.utcnow().isoformat() + "Z", "event": event, **fields}
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def read_status() -> Dict:
    if STATUS_FILE.exists():
        return json.loads(STATUS_FILE.read_text())
    return {"state": "uninitialised", "model_version": 0,
            "last_refine": None, "message": "No model trained yet."}


def write_status(**fields):
    st = read_status()
    st.update(fields)
    STATUS_FILE.write_text(json.dumps(st, indent=2))
    return st


# --------------------------------------------------------------------------
# Config / agent loading
# --------------------------------------------------------------------------
class ModelService:
    def __init__(self, config_path: str = None):
        self.config_path = Path(config_path or (ROOT / "configs" / "config.yaml"))
        self.cfg = yaml.safe_load(self.config_path.read_text())
        self._lock = threading.Lock()           # guards model swap + refine
        self._refining = False
        self.agent: Optional[IQLAgent] = None
        self.sim_cfg: Optional[SimConfig] = None
        self.env_cfg: Optional[EnvConfig] = None
        self._load_live_or_bootstrap()

    # ---- env/sim config builders --------------------------------------
    def _build_sim_cfg(self) -> SimConfig:
        c = self.cfg
        return SimConfig(
            FC=c["soil"]["field_capacity_pct"], WP=c["soil"]["wilting_point_pct"],
            SAT=c["soil"]["saturation_pct"],
            surface_depth_m=c["soil"]["surface_layer_depth_m"],
            deep_depth_m=c["soil"]["deep_layer_depth_m"],
            tau_drain_h=c["soil"]["tau_drain_h"], Ke_max=c["soil"]["Ke_max"],
            stage_days=c["crop"]["stage_days"], Kc=c["crop"]["Kc"],
            plants_per_m2=c["crop"]["plants_per_m2"],
            root_depth_init_m=c["crop"]["root_depth_m_initial"],
            root_depth_max_m=c["crop"]["root_depth_m_max"],
        )

    def _build_env_cfg(self) -> EnvConfig:
        c = self.cfg
        return EnvConfig(
            action_litres_per_plant=c["actuator"]["action_litres_per_plant"],
            plants_per_m2=c["crop"]["plants_per_m2"],
            surface_depth_m=c["soil"]["surface_layer_depth_m"],
            decision_hours=c["actuator"]["decision_hours"],
            max_daily_water_L_per_plant=c["actuator"]["max_daily_volume_L_per_plant"],
            alpha_water=c["reward"]["alpha_water"], beta_stress=c["reward"]["beta_stress"],
            gamma_oversat=c["reward"]["gamma_oversat"],
            yield_terminal_weight=c["reward"]["yield_terminal_weight"],
            optimal_band=c["crop"]["optimal_moisture_band"],
            total_days=sum(c["crop"]["stage_days"].values()),
        )

    def _new_agent(self) -> IQLAgent:
        env = IrrigationEnv(self._build_env_cfg(), self.sim_cfg,
                            rng=np.random.default_rng(0))
        iql_cfg = IQLConfig(obs_dim=env.OBS_DIM, n_actions=env.n_actions,
                            hidden_dim=self.cfg["iql"]["hidden_dim"],
                            n_hidden=self.cfg["iql"]["hidden_layers"])
        return IQLAgent(iql_cfg)

    def _load_live_or_bootstrap(self):
        self.sim_cfg = self._build_sim_cfg()
        self.env_cfg = self._build_env_cfg()
        if LIVE_SIMCFG.exists():
            saved = json.loads(LIVE_SIMCFG.read_text())
            for k, v in saved.items():
                if hasattr(self.sim_cfg, k):
                    setattr(self.sim_cfg, k, v)
        self.agent = self._new_agent()
        if LIVE_CKPT.exists():
            self.agent.load_state_dict(torch.load(LIVE_CKPT, map_location="cpu",
                                                  weights_only=False))
            write_status(state="ready",
                         message="Live model loaded.")
        else:
            # bootstrap: copy the packaged checkpoint if present
            pkg_ckpt = ROOT / "artifacts" / "ckpts" / "iql_final.pt"
            if pkg_ckpt.exists():
                self.agent.load_state_dict(torch.load(pkg_ckpt, map_location="cpu",
                                                      weights_only=False))
                self._save_live(model_version=1,
                                note="bootstrapped from packaged checkpoint")
                write_status(state="ready", model_version=1,
                             message="Bootstrapped from packaged model.")
            else:
                write_status(state="needs_training",
                             message="No trained model found. Upload data and "
                                     "run an initial refine, or train offline.")

    def _save_live(self, model_version: int, note: str = ""):
        torch.save(self.agent.state_dict(), LIVE_CKPT)
        LIVE_SIMCFG.write_text(json.dumps(asdict(self.sim_cfg), indent=2,
                                          default=float))
        write_status(state="ready", model_version=model_version,
                     last_refine=datetime.utcnow().isoformat() + "Z", note=note)

    # ---- recommendation ------------------------------------------------
    # ---- worker-reported rain / irrigation events ---------------------
    def log_events(self, csv_path, events):
        """Persist worker-reported rain/irrigation events alongside the upload.
        These are the labels the node CSV cannot capture. Stored to an
        events file and the audit log so the researcher can later fold them
        into calibration."""
        if not events:
            return
        ev_file = LOGS / "field_events.jsonl"
        rec = {"ts": datetime.utcnow().isoformat() + "Z",
               "source_csv": Path(csv_path).name, "events": events}
        with open(ev_file, "a") as f:
            f.write(json.dumps(rec) + "\n")
        for e in events:
            audit("field_event", source=Path(csv_path).name, **e)

    # ---- read latest moisture from an uploaded node CSV ---------------
    def latest_moisture_from_csv(self, csv_path, plot_filter=None):
        """Most recent moisture/temp from a node CSV (sensor state only; the
        file has no rain/irrigation column)."""
        try:
            df = load_field_csvs([csv_path])
        except Exception as e:
            return {"ok": False, "error": f"Could not read CSV: {e}"}
        if len(df) == 0:
            return {"ok": False, "error": "CSV has no readable rows."}
        if plot_filter:
            df = df[df["plot_id"].str.upper() == plot_filter.upper()]
            if len(df) == 0:
                return {"ok": False, "error": f"No rows for plot {plot_filter}."}
        row = df.sort_values("dt").iloc[-1]
        return {"ok": True,
                "moisture_pct": round(float(row["M"]), 1),
                "temp_c": round(float(row["T"]), 1),
                "dt": str(row["dt"]),
                "plot_id": str(row["plot_id"])}

    def next_decision_window(self, now_hour=None):
        """Next decision hour (9 or 14) given the current hour."""
        import datetime as _dt
        if now_hour is None:
            now_hour = _dt.datetime.now().hour
        windows = sorted(self.cfg["actuator"]["decision_hours"])
        for w in windows:
            if now_hour <= w:
                return w
        return windows[0]

    def recommend(self, moisture_pct: float, hour: int,
                  days_after_transplant: int,
                  moisture_deep_pct: float = None) -> Dict:
        """Return a watering recommendation for the T2 plot."""
        if self.agent is None:
            return {"ok": False, "error": "No model available yet."}

        # Validate input
        if not (0.0 <= moisture_pct <= 110.0):
            return {"ok": False,
                    "error": f"Moisture {moisture_pct}% out of range (0-110)."}

        env = IrrigationEnv(self.env_cfg, self.sim_cfg,
                            rng=np.random.default_rng(0))
        # Build an observation matching env._obs at the given state
        s_deep = moisture_deep_pct if moisture_deep_pct is not None else moisture_pct
        stage = stage_from_day(days_after_transplant, self.sim_cfg.stage_days)
        stage_oh = [0.0] * 4
        stage_oh[stage] = 1.0
        session_flag = 0.0 if hour <= 11 else 1.0
        cap = max(self.env_cfg.max_daily_water_L_per_plant
                  * self.env_cfg.plants_per_m2, 1e-3)
        obs = np.array([
            (moisture_pct - 50.0) / 30.0,
            (s_deep - 50.0) / 30.0,
            (30.0 - 25.0) / 5.0,           # soil temp placeholder (shade ~30C)
            0.0, 0.0,
            float(np.sin(2 * np.pi * hour / 24.0)),
            float(np.cos(2 * np.pi * hour / 24.0)),
            min(days_after_transplant, 100) / 100.0,
            *stage_oh, 0.0, session_flag,
        ], dtype=np.float32)

        a_idx = self.agent.act(obs, deterministic=True)
        litres = self.env_cfg.action_litres_per_plant[a_idx]

        # Convert litres -> drip minutes using the emitter rate (1 L/h confirmed)
        emitter_L_per_h = self.cfg["actuator"].get("emitter_rate_L_per_h", 1.0)
        minutes = round(litres / max(emitter_L_per_h, 1e-6) * 60.0)

        stage_name = ["initial", "development", "mid (flowering)", "late"][stage]
        return {
            "ok": True,
            "plot": "T2",
            "time": f"{hour:02d}:00",
            "moisture_pct": moisture_pct,
            "days_after_transplant": days_after_transplant,
            "stage": stage_name,
            "litres_per_plant": round(litres, 2),
            "drip_minutes": int(minutes),
            "model_version": read_status().get("model_version", 0),
            "note": ("No water needed now." if litres == 0
                     else f"Run drip ~{int(minutes)} min ({litres:.2f} L/plant)."),
        }

    # ---- background refine cycle ---------------------------------------
    def refine_async(self, upload_path: str, planting_date: str,
                     shadow_days: int = 3, finetune_steps: int = 400):
        """Kick off a background refine; returns immediately."""
        if self._refining:
            return {"ok": False, "error": "A refine is already running."}
        t = threading.Thread(target=self._refine_worker,
                             args=(upload_path, planting_date,
                                   shadow_days, finetune_steps), daemon=True)
        t.start()
        return {"ok": True, "message": "Refine started in background."}

    def _validate_upload(self, paths: List[str]) -> Dict:
        """Anomaly checks before a file is allowed to influence the model."""
        try:
            df = load_field_csvs(paths)
        except Exception as e:
            return {"ok": False, "reason": f"Could not parse: {e}"}
        if len(df) < 20:
            return {"ok": False, "reason": f"Too few rows ({len(df)})."}
        m = df["M"]
        if m.min() < 0 or m.max() > 110:
            return {"ok": False,
                    "reason": f"Moisture out of range ({m.min():.0f}-{m.max():.0f})."}
        if df["T"].min() < 5 or df["T"].max() > 65:
            return {"ok": False,
                    "reason": f"Temp out of range ({df['T'].min():.0f}-{df['T'].max():.0f})."}
        return {"ok": True, "rows": len(df),
                "span_days": (df["dt"].max() - df["dt"].min()).days}

    def _eval_return(self, agent: IQLAgent, n_episodes: int = 4) -> float:
        env = IrrigationEnv(self.env_cfg, self.sim_cfg,
                            rng=np.random.default_rng(123))
        rets = []
        for _ in range(n_episodes):
            obs = env.reset(); done = False; R = 0.0
            while not done:
                a = agent.act(obs, deterministic=True)
                obs, r, done, _ = env.step(a)
                R += r
            rets.append(R)
        return float(np.mean(rets))

    def _refine_worker(self, upload_path, planting_date,
                       shadow_days, finetune_steps):
        self._refining = True
        write_status(state="refining",
                     message="Recalibrating and fine-tuning candidate model...")
        try:
            audit("refine_start", upload=str(upload_path),
                  planting_date=planting_date)

            # 1. Validate
            v = self._validate_upload([upload_path])
            if not v["ok"]:
                audit("refine_quarantine", reason=v["reason"])
                write_status(state="ready",
                             message=f"Upload quarantined: {v['reason']} "
                                     f"Live model unchanged.")
                return

            # 2. Recalibrate deep-layer tau from the new field data
            new_sim = self._build_sim_cfg()
            # carry forward any previously-learned calibration
            if LIVE_SIMCFG.exists():
                saved = json.loads(LIVE_SIMCFG.read_text())
                for k, val in saved.items():
                    if hasattr(new_sim, k):
                        setattr(new_sim, k, val)
            try:
                new_sim = calibrate_from_field([upload_path], new_sim, verbose=False)
            except Exception as e:
                audit("refine_calib_warn", warn=str(e))

            # 3. Fine-tune a candidate from the live agent
            old_sim = self.sim_cfg
            self.sim_cfg = new_sim   # candidate evaluates on the new sim
            candidate = self._new_agent()
            candidate.load_state_dict(self.agent.state_dict())
            self._finetune(candidate, steps=finetune_steps)

            # 4. Shadow-test: candidate must beat live on the SAME sim
            live_score = self._eval_return(self.agent)
            cand_score = self._eval_return(candidate)
            audit("shadow_test", live_return=round(live_score, 2),
                  candidate_return=round(cand_score, 2))

            promote = cand_score > live_score + 1e-6
            if promote:
                with self._lock:
                    self.agent = candidate
                    ver = read_status().get("model_version", 0) + 1
                    self._save_live(model_version=ver,
                                    note=f"promoted (cand {cand_score:.1f} > "
                                         f"live {live_score:.1f})")
                audit("refine_promote", model_version=ver,
                      live_return=round(live_score, 2),
                      candidate_return=round(cand_score, 2))
                write_status(state="ready",
                             message=f"Model improved and promoted (v{ver}).")
            else:
                self.sim_cfg = old_sim   # roll back sim too
                audit("refine_reject",
                      live_return=round(live_score, 2),
                      candidate_return=round(cand_score, 2))
                write_status(state="ready",
                             message="Candidate did not beat live model. "
                                     "Live model kept unchanged.")
        except Exception as e:
            audit("refine_error", error=str(e), tb=traceback.format_exc())
            write_status(state="ready",
                         message=f"Refine failed: {e}. Live model unchanged.")
        finally:
            self._refining = False

    def _finetune(self, agent: IQLAgent, steps: int):
        """Light fine-tune on a fresh simulator buffer (deep-layer updated)."""
        from src.iql import ReplayBuffer
        env = IrrigationEnv(self.env_cfg, self.sim_cfg,
                            rng=np.random.default_rng(7))
        buf = ReplayBuffer(capacity=40000, obs_dim=env.OBS_DIM)
        # quick behaviour rollouts to fill buffer
        for _ in range(20):
            obs = env.reset(); done = False
            while not done:
                a = int(np.random.randint(env.n_actions))
                nxt, r, done, _ = env.step(a)
                buf.add(obs, a, r, nxt, done)
                obs = nxt
        for _ in range(steps):
            agent.update(buf.sample(256))

"""
webapp/app.py
=============
FastAPI backend for the smart-irrigation advisory tool.

Endpoints
---------
GET  /                      -> farm-facing page (static index.html)
GET  /research              -> research/admin page (status, audit, manual refine)
POST /api/recommend         -> {moisture_pct, hour, planting_date} -> recommendation
POST /api/upload            -> multipart CSV from SD card; kicks off background refine
GET  /api/status            -> current model status (version, state, last refine)
GET  /api/audit             -> recent audit-log entries (research view)

Run locally:
    cd webapp
    uvicorn app:app --reload --port 8000
Then open http://localhost:8000
"""

from __future__ import annotations

import shutil
from datetime import datetime, date
from pathlib import Path

import os, secrets
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from model_service import (ModelService, read_status, AUDIT_LOG, UPLOADS, MODELS)

HERE = Path(__file__).resolve().parent
app = FastAPI(title="Smart Irrigation Advisor (T2)")

service = ModelService()


def _days_after(planting_date: str) -> int:
    try:
        pd = datetime.strptime(planting_date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, "planting_date must be YYYY-MM-DD")
    return max((date.today() - pd).days, 0)


@app.get("/", response_class=HTMLResponse)
def home():
    return (HERE / "static" / "index.html").read_text()


@app.get("/research", response_class=HTMLResponse)
def research():
    return (HERE / "static" / "research.html").read_text()


@app.post("/api/upload_events")
async def api_upload_events(file: UploadFile = File(...),
                            planting_date: str = Form("2026-06-08")):
    """Upload a table of rain/irrigation events (+ optional farmer judgment).
    Returns each row with the model's recommendation alongside, for comparison.
    No admin key needed — the worker can use this."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv events table.")
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    dest = UPLOADS / f"events_{ts}_{file.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    res = service.process_events_csv(str(dest), planting_date)
    return JSONResponse(res)


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...),
                     events_file: UploadFile = File(None),
                     planting_date: str = Form("2026-06-08"),
                     rained: str = Form("no"),
                     rain_hours: float = Form(0.0)):
    """Single farm action: upload the SD-card sensor CSV (and optionally an
    events-history CSV in the same submit). The app reads the latest moisture
    from the sensor file, advises for the next 9 AM / 2 PM window, and (if an
    events file is given) logs that history with model-vs-farmer comparison."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv from the sensor node.")
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    dest = UPLOADS / f"{ts}_{file.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    latest = service.latest_moisture_from_csv(str(dest))
    if not latest["ok"]:
        return JSONResponse({"ok": False, "error": latest["error"]})

    # today's rain toggle
    events = []
    if str(rained).lower() in ("yes", "true", "1") and rain_hours > 0:
        events.append({"type": "rain", "hours": rain_hours})
    if events:
        service.log_events(str(dest), events)

    # optional events-history file uploaded in the same submit
    events_result = None
    if events_file is not None and events_file.filename:
        if events_file.filename.lower().endswith(".csv"):
            ed = UPLOADS / f"events_{ts}_{events_file.filename}"
            with open(ed, "wb") as f:
                shutil.copyfileobj(events_file.file, f)
            events_result = service.process_events_csv(str(ed), planting_date)

    window = service.next_decision_window()
    dat = _days_after(planting_date)
    rec = service.recommend(moisture_pct=latest["moisture_pct"], hour=window,
                            days_after_transplant=dat)
    started = service.refine_async(str(dest), planting_date)
    return JSONResponse({"ok": True, "saved": dest.name,
                         "latest_reading": latest, "events_logged": events,
                         "events_history": events_result,
                         "recommendation": rec, "refine": started})


@app.post("/api/farmer_judgment")
async def api_farmer_judgment(model_minutes: int = Form(...),
                              farmer_minutes: float = Form(...),
                              moisture_pct: float = Form(...),
                              reason: str = Form(""),
                              date: str = Form(None)):
    """Record today's on-screen farmer judgment next to the model's rec."""
    from datetime import date as _date
    d = date or _date.today().isoformat()
    service.log_farmer_judgment(model_minutes, farmer_minutes, reason,
                                moisture_pct, d)
    return JSONResponse({"ok": True,
                         "message": "Recorded the farmer's choice alongside "
                                    "the model's recommendation."})


@app.post("/api/recommend")
async def api_recommend(moisture_pct: float = Form(...),
                        hour: int = Form(None),
                        planting_date: str = Form("2026-06-08")):
    """Fallback manual path (card not available). Normal flow uses /api/upload."""
    dat = _days_after(planting_date)
    if hour is None:
        hour = service.next_decision_window()
    res = service.recommend(moisture_pct=moisture_pct, hour=hour,
                            days_after_transplant=dat)
    return JSONResponse(res)


@app.get("/api/status")
def api_status():
    return JSONResponse(read_status())


@app.post("/api/upload_model")
async def api_upload_model(file: UploadFile = File(...),
                           x_admin_key: str = Header(...)):
    """
    Researcher uploads a newly trained iql_final.pt from their laptop.
    Protected by a secret key set in the ADMIN_KEY environment variable.
    """
    expected = os.environ.get("ADMIN_KEY", "")
    if not expected or not secrets.compare_digest(x_admin_key, expected):
        raise HTTPException(403, "Invalid admin key.")
    if not file.filename.endswith(".pt"):
        raise HTTPException(400, "Please upload a .pt checkpoint file.")
    import shutil
    dest = MODELS / "live.pt"
    tmp  = MODELS / "incoming.pt"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # Validate: try loading as a state dict
    try:
        import torch
        torch.load(tmp, map_location="cpu", weights_only=False)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, f"File does not look like a valid checkpoint: {e}")
    tmp.replace(dest)
    # Reload the live model
    service._load_live_or_bootstrap()
    from model_service import audit
    audit("model_uploaded", filename=file.filename)
    return JSONResponse({"ok": True,
                         "message": f"Model updated from {file.filename}. "
                                    f"New version: {read_status().get('model_version',0)}"})


@app.get("/api/audit")
def api_audit(limit: int = 50):
    if not AUDIT_LOG.exists():
        return JSONResponse({"entries": []})
    lines = AUDIT_LOG.read_text().strip().splitlines()[-limit:]
    import json
    return JSONResponse({"entries": [json.loads(x) for x in lines][::-1]})

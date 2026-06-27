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

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from model_service import (ModelService, read_status, AUDIT_LOG, UPLOADS)

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


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...),
                     planting_date: str = Form("2026-06-08"),
                     rained: str = Form("no"),
                     rain_hours: float = Form(0.0),
                     irrigated: str = Form("no"),
                     irrigation_minutes: float = Form(0.0)):
    """Worker uploads the SD-card CSV and logs any rain / irrigation the file
    cannot record. App reads the latest moisture from the CSV automatically and
    advises for the NEXT 9 AM / 2 PM window."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv from the sensor node.")
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    dest = UPLOADS / f"{ts}_{file.filename}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    latest = service.latest_moisture_from_csv(str(dest))
    if not latest["ok"]:
        return JSONResponse({"ok": False, "error": latest["error"]})

    events = []
    if str(irrigated).lower() in ("yes", "true", "1") and irrigation_minutes > 0:
        events.append({"type": "irrigation", "minutes": irrigation_minutes})
    if str(rained).lower() in ("yes", "true", "1") and rain_hours > 0:
        events.append({"type": "rain", "hours": rain_hours})
    service.log_events(str(dest), events)

    window = service.next_decision_window()
    dat = _days_after(planting_date)
    rec = service.recommend(moisture_pct=latest["moisture_pct"], hour=window,
                            days_after_transplant=dat)
    started = service.refine_async(str(dest), planting_date)
    return JSONResponse({"ok": True, "saved": dest.name,
                         "latest_reading": latest, "events_logged": events,
                         "recommendation": rec, "refine": started})


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


@app.get("/api/audit")
def api_audit(limit: int = 50):
    if not AUDIT_LOG.exists():
        return JSONResponse({"entries": []})
    lines = AUDIT_LOG.read_text().strip().splitlines()[-limit:]
    import json
    return JSONResponse({"entries": [json.loads(x) for x in lines][::-1]})

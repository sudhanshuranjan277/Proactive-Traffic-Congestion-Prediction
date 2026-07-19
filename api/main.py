"""
FastAPI Backend

Exposes the live traffic engine (realtime/live_engine.py) over:
- REST: start/stop the simulation, get a status snapshot
- WebSocket: push live updates as new observation windows complete

Run with:
    uvicorn api.main:app --host 127.0.0.1 --port 8000
or:
    python api/main.py
"""

import os
import sys
import asyncio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from config import (
    API_HOST,
    API_PORT,
    WEBSOCKET_POLL_INTERVAL_SECONDS,
)

from realtime.live_engine import LiveTrafficEngine


app = FastAPI(
    title="Proactive Traffic Congestion Prediction — Live API",
    description=(
        "Real-time traffic simulation, LSTM prediction, and "
        "fixed-time signal control status."
    ),
    version="1.0.0",
)

# Streamlit (and any other frontend) runs on a different port, so it
# needs CORS enabled to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared engine instance — one live simulation at a time.
engine = LiveTrafficEngine()


@app.get("/")
def root():
    return {
        "service": "Proactive Traffic Congestion Prediction API",
        "status": "ok",
        "endpoints": [
            "GET /simulation/status",
            "GET /simulation/history/{junction_id}",
            "POST /simulation/start",
            "POST /simulation/stop",
            "WS   /ws/live",
        ],
    }


@app.get("/simulation/status")
def get_status():
    return engine.get_status()


@app.get("/simulation/history/{junction_id}")
def get_history(junction_id: str, limit: int = 200):
    return {"junction_id": junction_id, "history": engine.get_history(junction_id, limit)}


@app.post("/simulation/start")
def start_simulation():
    started = engine.start()
    return {
        "started": started,
        "message": (
            "Simulation started." if started
            else "Simulation is already running."
        ),
    }


@app.post("/simulation/stop")
def stop_simulation():
    stopped = engine.stop()
    return {
        "stopped": stopped,
        "message": (
            "Simulation stopped." if stopped
            else "Simulation was not running."
        ),
    }


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    Pushes a new status snapshot whenever engine.sequence_number
    advances (i.e. whenever a new observation window completed) —
    not a fixed tick rate, so clients only get genuinely new data.
    """

    await websocket.accept()

    last_sequence_number = -1

    try:
        while True:

            status = engine.get_status()

            if status["sequence_number"] != last_sequence_number:
                last_sequence_number = status["sequence_number"]
                await websocket.send_json(status)

            await asyncio.sleep(WEBSOCKET_POLL_INTERVAL_SECONDS)

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)

"""
Streamlit Dashboard — Real-Time Monitoring

Connects to the FastAPI backend (api/main.py) over REST and displays:
- Real-Time Monitoring (current per-junction traffic state)
- Traffic Analytics (recent trend charts)
- Prediction Visualization (LSTM forecast vs. recent actuals)
- Traffic Event Status
- Performance Reports (cumulative metrics)

Run with:
    streamlit run visualization/streamlit_app.py

Requires api/main.py to already be running (separately):
    uvicorn api.main:app --host 127.0.0.1 --port 8000
"""

import os
import sys

import requests
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    API_BASE_URL,
    FRONTEND_REFRESH_INTERVAL_MS,
    PROJECT_NAME,
)


st.set_page_config(
    page_title="Proactive Traffic Control — Live Dashboard",
    layout="wide",
)


# ==============================================================
# Backend calls (each wrapped so a stopped backend doesn't crash
# the dashboard — it just shows a clear connection message)
# ==============================================================

def get_status():
    try:
        response = requests.get(f"{API_BASE_URL}/simulation/status", timeout=2)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.RequestException as exc:
        return None, str(exc)


def get_history(junction_id, limit=200):
    try:
        response = requests.get(
            f"{API_BASE_URL}/simulation/history/{junction_id}",
            params={"limit": limit},
            timeout=2,
        )
        response.raise_for_status()
        return response.json()["history"]
    except requests.exceptions.RequestException:
        return []


def start_simulation():
    try:
        requests.post(f"{API_BASE_URL}/simulation/start", timeout=5)
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not start simulation: {exc}")


def stop_simulation():
    try:
        requests.post(f"{API_BASE_URL}/simulation/stop", timeout=5)
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not stop simulation: {exc}")


# ==============================================================
# Header + controls
# ==============================================================

st.title(f"{PROJECT_NAME} — Live Dashboard")

st_autorefresh(interval=FRONTEND_REFRESH_INTERVAL_MS, key="dashboard_refresh")

status, connection_error = get_status()

control_col1, control_col2, control_col3 = st.columns([1, 1, 4])

with control_col1:
    if st.button("Start Simulation", type="primary", use_container_width=True):
        start_simulation()

with control_col2:
    if st.button("Stop Simulation", use_container_width=True):
        stop_simulation()

with control_col3:
    if connection_error:
        st.error(
            f"Cannot reach backend at {API_BASE_URL} — is api/main.py running? "
            f"({connection_error})"
        )
    elif status["is_running"]:
        st.success(
            f"Running — Controller: {status['controller_mode']} | "
            f"Simulation Time: {status['simulation_time']:.0f}s"
        )
    else:
        st.info("Idle — press Start Simulation.")

if connection_error:
    st.stop()

if status.get("error"):
    st.error(f"Simulation error: {status['error']}")

if not status.get("predictor_available"):
    st.warning(
        "No trained LSTM model found — live predictions are disabled. "
        f"Run scripts/train_model.py first. ({status.get('predictor_error')})"
    )

junctions = status.get("junctions", [])

if not junctions:
    st.info("Waiting for the first observation window from the simulation...")
    st.stop()


# ==============================================================
# Junction selector
# ==============================================================

selected_junction = st.selectbox("Junction", junctions)

latest = status["latest"].get(selected_junction)

if latest is None:
    st.info("No data yet for this junction.")
    st.stop()

features = latest["features"]
metrics = latest["cumulative_metrics"]
prediction = latest.get("prediction")


# ==============================================================
# Traffic Event Status
# ==============================================================

event_label = latest["traffic_event_label"]
event_severity = {
    "Normal Traffic": st.success,
    "Accident": st.error,
    "Road Maintenance": st.warning,
    "Emergency Vehicle": st.error,
    "Special Event": st.warning,
}.get(event_label, st.info)

event_severity(f"Traffic Event: {event_label}")


# ==============================================================
# Real-Time Monitoring
# ==============================================================

st.subheader("Real-Time Monitoring")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Queue Length", features["queue_length"])
m2.metric("Average Speed", f"{features['average_speed']:.1f}")
m3.metric("Waiting Time", f"{features['waiting_time']:.1f}s")
m4.metric("Vehicle Count", features["vehicle_count"])
m5.metric("Signal Phase", features["current_signal_phase"])


# ==============================================================
# Performance Reports (cumulative)
# ==============================================================

st.subheader("Performance Report (Cumulative)")

r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("Avg Queue Length", metrics["avg_queue_length"])
r2.metric("Avg Waiting Time", metrics["avg_waiting_time"])
r3.metric("Avg Speed", metrics["avg_speed"])
r4.metric("Total Vehicles", metrics["total_vehicles_observed"])
r5.metric("Congestion Ratio", f"{metrics['congestion_ratio'] * 100:.1f}%")


# ==============================================================
# Traffic Analytics (recent trends)
# ==============================================================

st.subheader("Traffic Analytics")

history = get_history(selected_junction, limit=200)

if history:
    history_df = pd.DataFrame(
        [
            {
                "simulation_time": h["simulation_time"],
                "queue_length": h["features"]["queue_length"],
                "average_speed": h["features"]["average_speed"],
                "waiting_time": h["features"]["waiting_time"],
            }
            for h in history
        ]
    ).set_index("simulation_time")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.caption("Queue Length Over Time")
        st.line_chart(history_df["queue_length"])

    with chart_col2:
        st.caption("Average Speed Over Time")
        st.line_chart(history_df["average_speed"])

    st.caption("Waiting Time Over Time")
    st.line_chart(history_df["waiting_time"])
else:
    st.info("Not enough history yet for trend charts.")


# ==============================================================
# Prediction Visualization
# ==============================================================

st.subheader("Prediction Visualization (LSTM)")

if prediction is None:
    st.info(
        "Prediction not yet available — needs enough observation history "
        "to fill the model's lookback window, or no trained model exists."
    )
elif "error" in prediction:
    st.error(f"Prediction error: {prediction['error']}")
else:
    target_options = list(prediction["targets"].keys())
    selected_target = st.selectbox("Predicted variable", target_options)

    predicted_values = prediction["targets"][selected_target]
    horizon = prediction["horizon_steps"]

    current_time = latest["simulation_time"]
    future_times = [current_time + step for step in range(1, horizon + 1)]

    prediction_df = pd.DataFrame(
        {"simulation_time": future_times, "predicted": predicted_values}
    ).set_index("simulation_time")

    st.caption(f"Forecast for '{selected_target}' — next {horizon} steps")
    st.line_chart(prediction_df["predicted"])

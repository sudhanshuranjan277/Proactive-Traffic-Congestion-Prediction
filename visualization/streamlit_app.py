"""
Streamlit Dashboard

Two tabs:
- Live Monitoring: connects to the FastAPI backend (api/main.py) for
  real-time SUMO simulation status, predictions, and metrics.
- Performance Report: Fixed vs Proactive controller comparison, built
  from REAL evaluation output files (evaluation/evaluate_controller.py
  + evaluation/compare_models.py) — never hard-coded numbers.

Run with:
    streamlit run visualization/streamlit_app.py

Requires api/main.py to already be running (separately) for the Live
Monitoring tab:
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

from visualization import report_charts


st.set_page_config(
    page_title="Proactive Traffic Control — Dashboard",
    layout="wide",
)

st.title(f"{PROJECT_NAME} — Dashboard")

live_tab, report_tab = st.tabs(["Live Monitoring", "Performance Report"])


# ==============================================================
# TAB 1 — Live Monitoring
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


with live_tab:

    st_autorefresh(interval=FRONTEND_REFRESH_INTERVAL_MS, key="dashboard_refresh")

    status, connection_error = get_status()

    control_col1, control_col2, control_col3 = st.columns([1, 1, 4])

    with control_col1:
        if st.button("Start Simulation", type="primary", width="stretch"):
            start_simulation()

    with control_col2:
        if st.button("Stop Simulation", width="stretch"):
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

    # NOTE: intentionally NOT using st.stop() here — that halts the
    # ENTIRE script, including the Performance Report tab below,
    # which doesn't depend on the live backend at all. A down
    # backend should only disable this tab's content, not the app.
    if not connection_error:

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
        else:

            selected_junction = st.selectbox("Junction", junctions)

            latest = status["latest"].get(selected_junction)

            if latest is None:
                st.info("No data yet for this junction.")
            else:

                features = latest["features"]
                metrics = latest["cumulative_metrics"]
                prediction = latest.get("prediction")

                event_label = latest["traffic_event_label"]
                event_severity = {
                    "Normal Traffic": st.success,
                    "Accident": st.error,
                    "Road Maintenance": st.warning,
                    "Emergency Vehicle": st.error,
                    "Special Event": st.warning,
                }.get(event_label, st.info)
                event_severity(f"Traffic Event: {event_label}")

                st.subheader("Real-Time Monitoring")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Queue Length", features["queue_length"])
                m2.metric("Average Speed", f"{features['average_speed']:.1f}")
                m3.metric("Waiting Time", f"{features['waiting_time']:.1f}s")
                m4.metric("Vehicle Count", features["vehicle_count"])
                m5.metric("Signal Phase", features["current_signal_phase"])

                st.subheader("Performance Report (Cumulative)")
                r1, r2, r3, r4, r5 = st.columns(5)
                r1.metric("Avg Queue Length", metrics["avg_queue_length"])
                r2.metric("Avg Waiting Time", metrics["avg_waiting_time"])
                r3.metric("Avg Speed", metrics["avg_speed"])
                r4.metric("Total Vehicles", metrics["total_vehicles_observed"])
                r5.metric("Congestion Ratio", f"{metrics['congestion_ratio'] * 100:.1f}%")

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

                st.subheader("Prediction Visualization (LSTM)")

                if prediction is None:
                    st.info(
                        "Prediction not yet available — needs enough observation "
                        "history to fill the model's lookback window, or no "
                        "trained model exists."
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


# ==============================================================
# TAB 2 — Performance Report (Fixed vs Proactive, real data only)
# ==============================================================

with report_tab:

    comparison_df = report_charts.load_comparison()

    if comparison_df is None:
        st.info(
            "No evaluation results yet. Run, in order:\n\n"
            "1. `python evaluation/evaluate_controller.py --mode fixed`\n"
            "2. `python evaluation/evaluate_controller.py --mode proactive`\n"
            "3. `python evaluation/compare_models.py`\n\n"
            "Then reload this page."
        )
    else:

        st.subheader("Summary Table")
        st.dataframe(
            comparison_df[
                ["metric_label", "fixed_value", "proactive_value", "improvement_percent", "result"]
            ].rename(columns={
                "metric_label": "Metric",
                "fixed_value": "Fixed",
                "proactive_value": "Proactive",
                "improvement_percent": "Improvement %",
                "result": "Result",
            }),
            width="stretch",
            hide_index=True,
        )

        st.subheader("Per-Metric Comparison")
        metric_cols = st.columns(2)
        for i, (_, row) in enumerate(comparison_df.iterrows()):
            with metric_cols[i % 2]:
                fig = report_charts.build_metric_bar_figure(row)
                st.pyplot(fig)

        fixed_df, proactive_df = report_charts.load_timeseries()

        if fixed_df is not None and proactive_df is not None:
            st.subheader("Trends Over Simulation Time")
            for column, label in [
                ("queue_length", "Average Queue Length"),
                ("waiting_time", "Average Waiting Time"),
                ("average_speed", "Average Speed"),
            ]:
                if column in fixed_df.columns and column in proactive_df.columns:
                    fig = report_charts.build_timeseries_figure(fixed_df, proactive_df, column, label)
                    st.pyplot(fig)
        else:
            st.caption(
                "Per-timestep trend charts unavailable — "
                "fixed_evaluation.csv / proactive_evaluation.csv not found."
            )

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.subheader("Radar Comparison")
            st.pyplot(report_charts.build_radar_figure(comparison_df))
        with chart_col2:
            st.subheader("Overall Improvement")
            st.pyplot(report_charts.build_improvement_figure(comparison_df))

        st.subheader("Download Full Report")
        if st.button("Generate PDF Report"):
            output_path = os.path.join(PROJECT_ROOT, "outputs", "results", "performance_report.pdf")
            try:
                report_charts.build_pdf_report(output_path)
                with open(output_path, "rb") as pdf_file:
                    st.download_button(
                        "Download PDF",
                        data=pdf_file.read(),
                        file_name="Traffic_Controller_Performance_Report.pdf",
                        mime="application/pdf",
                    )
            except FileNotFoundError as exc:
                st.error(str(exc))

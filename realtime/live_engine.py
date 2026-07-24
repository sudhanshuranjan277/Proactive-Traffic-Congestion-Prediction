"""
Live Traffic Engine

Runs a continuous SUMO simulation in a background thread (traci is
blocking/synchronous, so it can't run directly inside FastAPI's async
event loop), producing live feature observations, LSTM predictions,
and cumulative performance metrics that api/main.py exposes over
REST + WebSocket.

Controller mode: FIXED-TIME only for now (matches
evaluation/evaluate_controller.py's run_fixed_controller — the
default SUMO signal program runs untouched, no apply_action() calls).
DDQN will be swapped in once it's trained and validated; see
config.py comments / project roadmap.
"""

import os
import sys
import time
import threading
import traceback
from collections import deque
from datetime import datetime, timezone

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    DEFAULT_LOCATION_ID,
    OBSERVATION_WINDOW,
    SIMULATION_TIME,
    LOOKBACK,
    REALTIME_STEP_DELAY_SECONDS,
    REALTIME_HISTORY_LENGTH,
    TRAFFIC_EVENT_TYPES,
)

from environment.sumo_env import SumoEnvironment
from integration.collector import TrafficCollector
from integration.pipeline import TrafficPipeline

from prediction.predictor import TrafficPredictor


class LiveTrafficEngine:
    """
    Singleton-style engine — one live SUMO run at a time. Call start()
    to begin, stop() to end, get_status() at any point (thread-safe)
    to read the latest state.
    """

    def __init__(self, location_id=DEFAULT_LOCATION_ID):

        self.location_id = location_id

        self._lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()

        self.is_running = False
        self.is_connected = False
        self.error = None
        self.simulation_time = 0.0
        self.sequence_number = 0  # increments on every new snapshot

        self.controller_mode = "fixed-time"

        # Per-junction rolling buffers of raw feature rows, used to
        # build the LSTM input window once enough history exists.
        self._feature_buffers = {}  # junction_id -> deque[dict]

        # Per-junction recent history for frontend trend charts.
        self.history = {}  # junction_id -> deque[dict]

        # Per-junction latest snapshot (features + prediction).
        self.latest = {}  # junction_id -> dict

        # Per-junction cumulative running stats for the performance report.
        self._cumulative = {}  # junction_id -> dict

        # Predictor is optional — engine still runs (fixed-time,
        # feature collection only) if no trained model exists yet.
        self.predictor = None
        self.predictor_error = None
        self._load_predictor()

    # ------------------------------------------------------------
    # Predictor loading (non-fatal if missing)
    # ------------------------------------------------------------

    def _load_predictor(self):

        try:
            self.predictor = TrafficPredictor()
        except FileNotFoundError as exc:
            self.predictor = None
            self.predictor_error = str(exc)

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------

    def start(self):

        with self._lock:

            if self.is_running:
                return False

            self._stop_event.clear()
            self.error = None
            self.sequence_number = 0
            self._feature_buffers = {}
            self.history = {}
            self.latest = {}
            self._cumulative = {}

            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
            )
            self.is_running = True
            self._thread.start()

            return True

    def stop(self):

        with self._lock:

            if not self.is_running:
                return False

            self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=10)

        with self._lock:
            self.is_running = False

        return True

    # ------------------------------------------------------------
    # Background simulation loop
    # ------------------------------------------------------------

    def _run_loop(self):

        sumo_env = SumoEnvironment()
        collector = TrafficCollector()
        pipeline = TrafficPipeline(
            collector=collector,
            location_id=self.location_id,
            observation_window=OBSERVATION_WINDOW,
        )

        try:
            if not sumo_env.connect():
                raise RuntimeError("SUMO connection failed.")

            with self._lock:
                self.is_connected = True

            for _ in range(SIMULATION_TIME):

                if self._stop_event.is_set():
                    break

                sumo_env.simulation_step()

                rows = pipeline.process_step()

                if rows:
                    self._handle_new_rows(rows)

                if REALTIME_STEP_DELAY_SECONDS > 0:
                    time.sleep(REALTIME_STEP_DELAY_SECONDS)

        except Exception as exc:  # noqa: BLE001 — surface any failure to the API
            with self._lock:
                self.error = f"{exc}\n{traceback.format_exc()}"

        finally:
            try:
                sumo_env.disconnect()
            except Exception:
                pass

            with self._lock:
                self.is_connected = False
                self.is_running = False

    # ------------------------------------------------------------
    # Per-observation-window processing
    # ------------------------------------------------------------

    def _handle_new_rows(self, rows):

        timestamp = datetime.now(timezone.utc).isoformat()

        with self._lock:

            for row in rows:

                junction_id = row["junction_id"]
                self.simulation_time = row["simulation_time"]

                if junction_id not in self._feature_buffers:
                    self._feature_buffers[junction_id] = deque(maxlen=LOOKBACK)
                    self.history[junction_id] = deque(maxlen=REALTIME_HISTORY_LENGTH)
                    self._cumulative[junction_id] = {
                        "queue_length_sum": 0.0,
                        "waiting_time_sum": 0.0,
                        "average_speed_sum": 0.0,
                        "vehicle_count_sum": 0,
                        "observation_count": 0,
                        "congested_windows": 0,
                    }

                self._feature_buffers[junction_id].append(row)

                stats = self._cumulative[junction_id]
                stats["queue_length_sum"] += row["queue_length"]
                stats["waiting_time_sum"] += row["waiting_time"]
                stats["average_speed_sum"] += row["average_speed"]
                stats["vehicle_count_sum"] += row["vehicle_count"]
                stats["observation_count"] += 1
                if row["queue_length"] > 0:
                    stats["congested_windows"] += 1

                prediction = self._predict_for_junction(junction_id)

                event_type_code = row["traffic_event_type"]

                snapshot = {
                    "junction_id": junction_id,
                    "timestamp": timestamp,
                    "simulation_time": row["simulation_time"],
                    "features": row,
                    "traffic_event_label": TRAFFIC_EVENT_TYPES.get(
                        event_type_code, f"Unknown ({event_type_code})"
                    ),
                    "prediction": prediction,
                    "cumulative_metrics": self._compute_report(junction_id),
                }

                self.latest[junction_id] = snapshot
                self.history[junction_id].append(snapshot)

            self.sequence_number += 1

    def _predict_for_junction(self, junction_id):
        """
        Returns None until enough history (LOOKBACK rows) has
        accumulated for this junction, or if no trained model exists.
        """

        if self.predictor is None:
            return None

        buffer = self._feature_buffers[junction_id]

        if len(buffer) < LOOKBACK:
            return None

        feature_columns = self.predictor.feature_columns
        if not feature_columns:
            return None

        try:
            sequence = np.array(
                [[row[col] for col in feature_columns] for row in buffer],
                dtype=np.float32,
            )
            prediction = self.predictor.predict(sequence)  # (horizon, n_targets)

            target_columns = self.predictor.target_columns or []
            return {
                "horizon_steps": prediction.shape[0],
                "targets": {
                    target: prediction[:, i].tolist()
                    for i, target in enumerate(target_columns)
                },
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    def _compute_report(self, junction_id):

        stats = self._cumulative[junction_id]
        count = max(stats["observation_count"], 1)

        return {
            "avg_queue_length": round(stats["queue_length_sum"] / count, 3),
            "avg_waiting_time": round(stats["waiting_time_sum"] / count, 3),
            "avg_speed": round(stats["average_speed_sum"] / count, 3),
            "total_vehicles_observed": stats["vehicle_count_sum"],
            "congestion_ratio": round(stats["congested_windows"] / count, 3),
            "observation_count": stats["observation_count"],
        }

    # ------------------------------------------------------------
    # Thread-safe read access for the API layer
    # ------------------------------------------------------------

    def get_status(self):

        with self._lock:

            return {
                "is_running": self.is_running,
                "is_connected": self.is_connected,
                "controller_mode": self.controller_mode,
                "simulation_time": self.simulation_time,
                "sequence_number": self.sequence_number,
                "error": self.error,
                "predictor_available": self.predictor is not None,
                "predictor_error": self.predictor_error,
                "junctions": list(self.latest.keys()),
                "latest": dict(self.latest),
            }

    def get_history(self, junction_id, limit=REALTIME_HISTORY_LENGTH):

        with self._lock:

            if junction_id not in self.history:
                return []

            return list(self.history[junction_id])[-limit:]

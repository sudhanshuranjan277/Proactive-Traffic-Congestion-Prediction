"""
Reinforcement learning environment for a single SUMO junction.
"""

import numpy as np

from integration.pipeline import TrafficPipeline
from integration.controller import (
    TrafficSignalAction,
    TrafficSignalController,
)
from rl.reward import compute_reward
from prediction.predictor import TrafficPredictor


class JunctionTrafficEnvironment:

    def __init__(
        self,
        sumo_env,
        collector,
        predictor,
        junction_id,
        location_id="location_1",
        lookback=30,
        observation_window=60,
        max_steps=None,
        extension_seconds=5,
    ):
        self.sumo_env = sumo_env
        self.collector = collector
        self.predictor = predictor
        self.location_id = location_id
        self.junction_id = junction_id
        self.lookback = lookback
        self.observation_window = observation_window
        self.max_steps = max_steps
        self.extension_seconds = extension_seconds
        self.controller = TrafficSignalController(
            extension_seconds=extension_seconds,
        )
        self.pipeline = TrafficPipeline(
            collector=self.collector,
            location_id=self.location_id,
            observation_window=self.observation_window,
        )
        self.history = []
        self.previous_row = None
        self.step_index = 0
        self.action_space = range(3)
        self.observation_space = {
            "shape": (9,),
            "dtype": np.float32,
        }

    def reset(self):
        if not self.collector.initialized:
            self.collector.initialize_junctions()

        self.history = []
        self.previous_row = None
        self.step_index = 0
        self.pipeline = TrafficPipeline(
            collector=self.collector,
            location_id=self.location_id,
            observation_window=self.observation_window,
        )

        while len(self.history) < self.lookback:
            self.sumo_env.simulation_step()
            rows = self.pipeline.process_step()
            self._append_rows(rows)

        self.previous_row = self.history[-1]

        return self._build_state(
            self.previous_row,
            self.history[-self.lookback :],
        )

    def _append_rows(self, rows):
        for row in rows:
            if row["junction_id"] == self.junction_id:
                self.history.append(row)

    def _build_state(self, current_row, history_rows):
        features = [
            float(current_row["traffic_flow"]),
            float(current_row["traffic_event_type"]),
            float(current_row["remaining_green_time"]),
            float(current_row["downstream_occupancy"]),
            float(current_row["queue_length"]),
        ]

        predictor_input = np.asarray(
            [
                [
                    float(row["traffic_flow"]),
                    float(row["traffic_event_type"]),
                    float(row["remaining_green_time"]),
                    float(row["downstream_occupancy"]),
                    float(row["queue_length"]),
                ]
                for row in history_rows[-self.lookback :]
            ],
            dtype=np.float32,
        )

        predictions = self.predictor.predict_queue_length(
            predictor_input
        )

        predicted_mean = float(predictions.mean())
        predicted_max = float(predictions.max())
        predicted_trend = float(
            predictions[-1] - predictions[0]
        )

        saturation_score = min(
            1.0,
            max(
                0.0,
                predicted_max / 20.0,
            ),
        )

        state = np.asarray(
            [
                *features,
                predicted_mean,
                predicted_max,
                predicted_trend,
                saturation_score,
            ],
            dtype=np.float32,
        )

        return state

    def step(self, action):
        action_valid = self.controller.apply_action(
            self.junction_id,
            action,
        )

        for _ in range(self.observation_window):
            self.sumo_env.simulation_step()
            rows = self.pipeline.process_step()
            self._append_rows(rows)

        self.step_index += 1

        if not self.history:
            raise RuntimeError(
                "No observation available after action execution."
            )

        current_row = self.history[-1]

        reward = compute_reward(
            self.previous_row,
            current_row,
            action,
            action_executed=action_valid,
        )

        done = (
            self.max_steps is not None
            and self.step_index >= self.max_steps
        )

        state = self._build_state(
            current_row,
            self.history[-self.lookback :],
        )

        self.previous_row = current_row

        return state, reward, done, {
            "action_valid": action_valid,
            "junction_id": self.junction_id,
            "simulation_time": current_row["simulation_time"],
        }

    def render(self, mode="human"):
        return {
            "junction_id": self.junction_id,
            "step_index": self.step_index,
            "last_observation": self.history[-1] if self.history else None,
        }

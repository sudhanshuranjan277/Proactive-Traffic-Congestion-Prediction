"""
Reinforcement learning environment for a single SUMO junction.

The environment connects:

SUMO
    ->
Traffic Collector
    ->
Traffic Pipeline
    ->
LSTM Future Traffic Prediction
    ->
DDQN State
    ->
Traffic Signal Controller
    ->
Reward
"""

import numpy as np

from integration.pipeline import TrafficPipeline
from integration.controller import TrafficSignalController
from rl.reward import compute_reward


class JunctionTrafficEnvironment:

    # Default feature configuration. All of these can be overridden via the
    # constructor so nothing about the state/prediction shape is fixed in
    # the method bodies below.
    DEFAULT_INPUT_FEATURES = [
        "traffic_flow",
        "traffic_event_type",
        "remaining_green_time",
        "downstream_occupancy",
        "queue_length",
        "average_speed",
        "waiting_time",
        "current_signal_phase",
        "downstream_queue_length",
        "travel_time",
        "arrival_rate",
        "departure_rate",
    ]

    DEFAULT_TARGET_FEATURES = [
        "queue_length",
        "downstream_occupancy",
        "average_speed",
        "waiting_time",
    ]

    # Features pulled straight from the current row into the state vector.
    DEFAULT_CURRENT_STATE_FEATURES = [
        "traffic_flow",
        "remaining_green_time",
        "downstream_occupancy",
        "queue_length",
        "average_speed",
        "waiting_time",
        "downstream_queue_length",
        "arrival_rate",
        "departure_rate",
    ]

    # For each target/predicted feature, which aggregations over the
    # prediction horizon get appended to the state. Supported stats:
    # "mean", "max", "min", "delta" (last - first).
    DEFAULT_FUTURE_FEATURE_STATS = {
        "queue_length": ["mean", "max", "delta"],
        "downstream_occupancy": ["mean", "max"],
        "average_speed": ["mean"],
        "waiting_time": ["mean", "max"],
    }

    _STAT_FUNCS = {
        "mean": lambda arr: float(np.mean(arr)),
        "max": lambda arr: float(np.max(arr)),
        "min": lambda arr: float(np.min(arr)),
        "delta": lambda arr: float(arr[-1] - arr[0]),
    }

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
        num_actions=3,
        prediction_horizon=10,
        input_features=None,
        target_features=None,
        current_state_features=None,
        future_feature_stats=None,
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

        self.prediction_horizon = prediction_horizon

        self.input_features = (
            input_features
            if input_features is not None
            else list(self.DEFAULT_INPUT_FEATURES)
        )

        self.target_features = (
            target_features
            if target_features is not None
            else list(self.DEFAULT_TARGET_FEATURES)
        )

        self.current_state_features = (
            current_state_features
            if current_state_features is not None
            else list(self.DEFAULT_CURRENT_STATE_FEATURES)
        )

        self.future_feature_stats = (
            future_feature_stats
            if future_feature_stats is not None
            else {
                feature: list(stats)
                for feature, stats in self.DEFAULT_FUTURE_FEATURE_STATS.items()
            }
        )

        self._validate_future_feature_stats()

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

        self.action_space = range(num_actions)

        state_dim = len(self.current_state_features) + sum(
            len(stats) for stats in self.future_feature_stats.values()
        )

        self.observation_space = {
            "shape": (state_dim,),
            "dtype": np.float32,
        }

    def _validate_future_feature_stats(self):

        for feature, stats in self.future_feature_stats.items():

            if feature not in self.target_features:

                raise ValueError(
                    f"future_feature_stats references '{feature}', "
                    "which is not in target_features."
                )

            for stat in stats:

                if stat not in self._STAT_FUNCS:

                    raise ValueError(
                        f"Unsupported stat '{stat}' for feature "
                        f"'{feature}'. Supported stats: "
                        f"{sorted(self._STAT_FUNCS)}."
                    )

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

    def _build_predictor_input(
        self,
        history_rows,
    ):

        if len(history_rows) < self.lookback:

            raise ValueError(
                f"Expected at least {self.lookback} "
                f"historical observations, "
                f"received {len(history_rows)}."
            )

        predictor_input = np.asarray(
            [
                [
                    float(row[feature])
                    for feature in self.input_features
                ]
                for row in history_rows[-self.lookback :]
            ],
            dtype=np.float32,
        )

        expected_shape = (
            self.lookback,
            len(self.input_features),
        )

        if predictor_input.shape != expected_shape:

            raise ValueError(
                "Invalid predictor input shape. "
                f"Expected {expected_shape}, "
                f"received {predictor_input.shape}."
            )

        return predictor_input

    def _build_state(
        self,
        current_row,
        history_rows,
    ):

        current_features = np.asarray(
            [
                float(current_row[feature])
                for feature in self.current_state_features
            ],
            dtype=np.float32,
        )

        predictor_input = self._build_predictor_input(
            history_rows
        )

        predictions = self.predictor.predict(
            predictor_input
        )

        predictions = np.asarray(
            predictions,
            dtype=np.float32,
        )

        expected_prediction_shape = (
            self.prediction_horizon,
            len(self.target_features),
        )

        if predictions.shape != expected_prediction_shape:

            raise ValueError(
                "Invalid LSTM prediction shape. "
                f"Expected {expected_prediction_shape}, "
                f"received {predictions.shape}."
            )

        future_values = []

        for feature, stats in self.future_feature_stats.items():

            column_index = self.target_features.index(feature)

            predicted_series = predictions[:, column_index]

            for stat in stats:

                future_values.append(
                    self._STAT_FUNCS[stat](predicted_series)
                )

        future_features = np.asarray(
            future_values,
            dtype=np.float32,
        )

        state = np.concatenate(
            [
                current_features,
                future_features,
            ]
        ).astype(np.float32)

        if state.shape != self.observation_space["shape"]:

            raise ValueError(
                "Invalid DDQN state shape. "
                f"Expected {self.observation_space['shape']}, "
                f"received {state.shape}."
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
                "No observation available "
                "after action execution."
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
            "simulation_time": current_row[
                "simulation_time"
            ],
        }

    def render(self, mode="human"):

        return {
            "junction_id": self.junction_id,
            "step_index": self.step_index,
            "last_observation": (
                self.history[-1]
                if self.history
                else None
            ),
        }
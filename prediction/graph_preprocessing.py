"""
Graph-Aligned Traffic Preprocessing

Unlike prediction/preprocessing.py's TrafficPreprocessor (which
builds independent per-junction sequences for the LSTM), this module
aligns ALL junctions to shared simulation_time steps, so each
training sample is a full graph snapshot: every junction's
lookback window and target horizon, at the same points in time.

Reuses TrafficPreprocessor's column definitions and StandardScaler
so both pipelines stay consistent with each other.
"""

import numpy as np

from prediction.preprocessing import TrafficPreprocessor


class GraphTrafficPreprocessor:

    FEATURE_COLUMNS = TrafficPreprocessor.FEATURE_COLUMNS
    TARGET_COLUMNS = TrafficPreprocessor.TARGET_COLUMNS
    REQUIRED_COLUMNS = TrafficPreprocessor.REQUIRED_COLUMNS

    def __init__(self, lookback=30, prediction_horizon=10):
        if lookback <= 0:
            raise ValueError("Lookback must be greater than zero.")
        if prediction_horizon <= 0:
            raise ValueError(
                "Prediction horizon must be greater than zero."
            )
        self.lookback = lookback
        self.prediction_horizon = prediction_horizon
        self._base = TrafficPreprocessor(lookback, prediction_horizon)

    def load_dataset(self, dataset_path):
        return self._base.load_dataset(dataset_path)

    def prepare_dataset(self, dataset):
        return self._base.prepare_dataset(dataset)

    def align_junctions(self, dataset):
        """
        Returns (junction_ids, common_times, per_junction_frames).

        junction_ids is sorted for a stable, reproducible node
        ordering — this ordering IS the graph's node index mapping,
        and must be saved alongside the trained model so inference
        uses the same order.

        common_times is the intersection of simulation_time values
        present for every junction — only steps where the full graph
        snapshot exists are usable.
        """

        junction_ids = sorted(dataset["junction_id"].unique().tolist())

        if len(junction_ids) < 2:
            raise ValueError(
                "Graph model needs at least 2 junctions to model "
                f"cross-junction relationships; found {len(junction_ids)}. "
                "Use the LSTM model (prediction/lstm.py) for "
                "single-junction data instead."
            )

        per_junction_frames = {}
        common_times = None

        for junction_id in junction_ids:
            duplicates = dataset[
                dataset["junction_id"] == junction_id
            ]["simulation_time"].duplicated().sum()
            if duplicates > 0:
                print(
                    f"Warning: {duplicates} duplicate timestamps "
                    f"removed for junction {junction_id}"
                )

            frame = (
                dataset[dataset["junction_id"] == junction_id]
                .drop_duplicates(subset="simulation_time")
                .set_index("simulation_time")
                .sort_index()
            )
            per_junction_frames[junction_id] = frame

            times = set(frame.index)
            common_times = times if common_times is None else (common_times & times)

        common_times = sorted(common_times)

        if not common_times:
            raise ValueError(
                "No simulation_time values are shared across all "
                "junctions — cannot build aligned graph snapshots. "
                "Junctions may be running on different/offset clocks."
            )

        return junction_ids, common_times, per_junction_frames

    def create_sequences(self, dataset):

        junction_ids, common_times, per_junction_frames = self.align_junctions(
            dataset
        )

        minimum_steps = self.lookback + self.prediction_horizon

        if len(common_times) < minimum_steps:
            raise ValueError(
                f"Only {len(common_times)} shared timesteps across "
                f"junctions, need at least {minimum_steps} "
                "(lookback + prediction_horizon)."
            )

        feature_stack = []
        target_stack = []

        for junction_id in junction_ids:
            frame = per_junction_frames[junction_id].loc[common_times]

            if frame.empty:
                raise ValueError(
                    f"No aligned data found for {junction_id}"
                )

            if np.isnan(
                frame[self.FEATURE_COLUMNS]
            ).any():
                raise ValueError(
                    f"NaN values detected in junction {junction_id}"
                )

            feature_stack.append(
                frame[self.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            )
            target_stack.append(
                frame[self.TARGET_COLUMNS].to_numpy(dtype=np.float32)
            )

        # (num_junctions, T, n_features) / (num_junctions, T, n_targets)
        feature_stack = np.stack(feature_stack, axis=0)
        target_stack = np.stack(target_stack, axis=0)

        num_junctions, total_steps, _ = feature_stack.shape
        max_start = total_steps - self.lookback - self.prediction_horizon + 1

        X, y, metadata = [], [], []

        for start in range(max_start):
            input_end = start + self.lookback
            target_end = input_end + self.prediction_horizon

            X.append(feature_stack[:, start:input_end, :])
            y.append(target_stack[:, input_end:target_end, :])

            metadata.append(
                {
                    "sample_index": start,
                    "junction_ids": list(junction_ids),
                    "num_junctions": len(junction_ids),
                    "input_start_time": float(common_times[start]),
                    "input_end_time": float(common_times[input_end - 1]),
                    "prediction_start_time": float(common_times[input_end]),
                    "target_end_time": float(common_times[target_end - 1]),
                }
            )

        if not X:
            raise ValueError(
                "No graph sequences generated — dataset may be too "
                "small for the configured lookback/prediction_horizon."
            )

        # (samples, num_junctions, lookback, n_features)
        X = np.asarray(X, dtype=np.float32)
        # (samples, num_junctions, horizon, n_targets)
        y = np.asarray(y, dtype=np.float32)

        return X, y, metadata, junction_ids

    def process(self, dataset_path):
        dataset = self.load_dataset(dataset_path)
        dataset = self.prepare_dataset(dataset)
        return self.create_sequences(dataset)
"""
Traffic Dataset Preprocessing

Responsible for:
- Loading traffic datasets
- Validating dataset columns
- Sorting observations chronologically
- Creating junction-wise LSTM sequences
- Preventing sequence mixing between locations/junctions
- Providing reusable feature and target scaling
"""

import pandas as pd
import numpy as np


class StandardScaler:

    """Standardize numeric values using a mean and standard deviation."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, values):
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            raise ValueError(
                "Cannot fit StandardScaler on empty values."
            )
        self.mean = np.mean(values, axis=0)
        self.std = np.std(values, axis=0)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        return self

    def transform(self, values):
        self._validate_fitted()
        values = np.asarray(values, dtype=np.float32)
        return (values - self.mean) / self.std

    def inverse_transform(self, values):
        self._validate_fitted()
        values = np.asarray(values, dtype=np.float32)
        return (values * self.std) + self.mean

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def _validate_fitted(self):
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler is not fitted.")


class TrafficPreprocessor:

    FEATURE_COLUMNS = [
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

    TARGET_COLUMNS = [
        "queue_length",
        "downstream_occupancy",
        "average_speed",
        "waiting_time",
    ]

    REQUIRED_COLUMNS = [
        "location_id",
        "junction_id",
        "simulation_time",
        "vehicle_count",
        *FEATURE_COLUMNS,
    ]

    def __init__(self, lookback=30, prediction_horizon=10):
        if lookback <= 0:
            raise ValueError("Lookback must be greater than zero.")
        if prediction_horizon <= 0:
            raise ValueError(
                "Prediction horizon must be greater than zero."
            )
        self.lookback = lookback
        self.prediction_horizon = prediction_horizon

    def load_dataset(self, dataset_path):
        dataset = pd.read_csv(dataset_path)
        if dataset.empty:
            raise ValueError("Traffic dataset is empty.")
        self.validate_columns(dataset)
        return dataset

    def validate_columns(self, dataset):
        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in dataset.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing dataset columns: {missing_columns}"
            )

    def prepare_dataset(self, dataset):
        dataset = dataset.copy()
        numeric_columns = [
            "simulation_time",
            "vehicle_count",
            *self.FEATURE_COLUMNS,
        ]
        for column in numeric_columns:
            dataset[column] = pd.to_numeric(
                dataset[column],
                errors="raise",
            )

        if dataset[numeric_columns].isnull().any().any():
            raise ValueError(
                "Dataset contains missing numeric values."
            )

        numeric_values = dataset[numeric_columns].to_numpy(dtype=np.float32)
        if not np.isfinite(numeric_values).all():
            raise ValueError(
                "Dataset contains infinite numeric values."
            )

        dataset = dataset.sort_values(
            by=["location_id", "junction_id", "simulation_time"],
        ).reset_index(drop=True)
        return dataset

    def create_sequences(self, dataset):
        input_sequences = []
        target_sequences = []
        sequence_metadata = []

        grouped_dataset = dataset.groupby(
            ["location_id", "junction_id"],
            sort=False,
        )

        minimum_rows = self.lookback + self.prediction_horizon

        for (location_id, junction_id), junction_data in grouped_dataset:
            junction_data = junction_data.sort_values(
                "simulation_time"
            ).reset_index(drop=True)
            if len(junction_data) < minimum_rows:
                continue

            feature_values = junction_data[self.FEATURE_COLUMNS].to_numpy(
                dtype=np.float32
            )
            target_values = junction_data[self.TARGET_COLUMNS].to_numpy(
                dtype=np.float32
            )

            max_start = len(junction_data) - self.lookback - self.prediction_horizon + 1
            for start_index in range(max_start):
                input_end = start_index + self.lookback
                target_end = input_end + self.prediction_horizon
                input_sequence = feature_values[start_index:input_end]
                target_sequence = target_values[input_end:target_end]
                input_sequences.append(input_sequence)
                target_sequences.append(target_sequence)
                sequence_metadata.append({
                    "location_id": str(location_id),
                    "junction_id": str(junction_id),
                    "input_start_time": float(
                        junction_data.iloc[start_index]["simulation_time"]
                    ),
                    "input_end_time": float(
                        junction_data.iloc[input_end - 1]["simulation_time"]
                    ),
                    "target_end_time": float(
                        junction_data.iloc[target_end - 1]["simulation_time"]
                    ),
                })

        if not input_sequences:
            raise ValueError(
                "No LSTM sequences generated. "
                "Dataset may be too small for the configured lookback "
                "and prediction horizon."
            )

        X = np.asarray(input_sequences, dtype=np.float32)
        y = np.asarray(target_sequences, dtype=np.float32)
        return X, y, sequence_metadata

    def process(self, dataset_path):
        dataset = self.load_dataset(dataset_path)
        dataset = self.prepare_dataset(dataset)
        return self.create_sequences(dataset)

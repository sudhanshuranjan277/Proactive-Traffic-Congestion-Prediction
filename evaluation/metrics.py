"""
Traffic Performance Evaluation Metrics

Focused evaluation metrics for the proactive
traffic congestion prediction and signal control
system.

Evaluation scope:
- Average Queue Length
- Average Waiting Time
- Average Speed
- Average Travel Time
- Total Throughput
- Congestion Ratio
- Spillback Events

These metrics are aligned with the current project
workflow:

Traffic Parameters
    ->
LSTM Future Traffic Prediction
    ->
Proactive DDQN Signal Control
    ->
Traffic Congestion Evaluation
"""

import math

import pandas as pd


REQUIRED_COLUMNS = {
    "queue_length",
    "waiting_time",
    "average_speed",
    "travel_time",
    "departure_rate",
    "traffic_event_type",
    "downstream_occupancy",
    "downstream_queue_length",
}


def load_evaluation_dataset(dataset):
    """
    Load and validate the evaluation dataset.

    Parameters
    ----------
    dataset:
        pandas DataFrame or CSV file path.

    Returns
    -------
    pandas.DataFrame
        Validated traffic dataset.
    """

    if isinstance(dataset, str):
        dataset = pd.read_csv(dataset)

    if not isinstance(dataset, pd.DataFrame):
        raise TypeError(
            "Dataset must be a pandas DataFrame "
            "or a CSV file path."
        )

    if dataset.empty:
        raise ValueError(
            "Evaluation dataset is empty."
        )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataset.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return dataset


def safe_mean(series):
    """
    Calculate a finite mean value.
    """

    value = float(series.mean())

    if not math.isfinite(value):
        return 0.0

    return value


def safe_sum(series):
    """
    Calculate a finite sum value.
    """

    value = float(series.sum())

    if not math.isfinite(value):
        return 0.0

    return value


def compute_average_queue_length(dataset):
    """
    Compute average traffic queue length.
    """

    return safe_mean(
        dataset["queue_length"]
    )


def compute_average_waiting_time(dataset):
    """
    Compute average vehicle waiting time.
    """

    return safe_mean(
        dataset["waiting_time"]
    )


def compute_average_speed(dataset):
    """
    Compute average traffic speed.
    """

    return safe_mean(
        dataset["average_speed"]
    )


def compute_average_travel_time(dataset):
    """
    Compute average vehicle travel time.
    """

    return safe_mean(
        dataset["travel_time"]
    )


def compute_total_throughput(dataset):
    """
    Compute total observed vehicle departures.

    departure_rate represents vehicles leaving
    the observed traffic region during an
    observation window.
    """

    return safe_sum(
        dataset["departure_rate"]
    )


def compute_congestion_ratio(dataset):
    """
    Compute the ratio of observations classified
    as traffic-event or congestion states.

    Current event encoding:

    0     -> Normal traffic
    non-0 -> Traffic event / congestion state
    """

    congestion_mask = (
        dataset["traffic_event_type"] != 0
    )

    return float(
        congestion_mask.mean()
    )


def compute_spillback_events(
    dataset,
    occupancy_threshold=0.10,
    queue_threshold=1,
):
    """
    Count potential downstream spillback events.

    A spillback condition is identified when:

    downstream occupancy >= occupancy threshold

    AND

    downstream queue length >= queue threshold.

    Thresholds remain configurable because the
    detector occupancy scale depends on the SUMO
    network and detector configuration.
    """

    if not 0.0 <= occupancy_threshold <= 1.0:
        raise ValueError(
            "Occupancy threshold must be "
            "between 0.0 and 1.0."
        )

    if queue_threshold < 0:
        raise ValueError(
            "Queue threshold cannot be negative."
        )

    spillback_mask = (
        (
            dataset["downstream_occupancy"]
            >= occupancy_threshold
        )
        & (
            dataset["downstream_queue_length"]
            >= queue_threshold
        )
    )

    return int(
        spillback_mask.sum()
    )


def summarize_metrics(
    dataset,
    occupancy_threshold=0.10,
    queue_threshold=1,
):
    """
    Generate the complete traffic performance
    metric summary.
    """

    dataset = load_evaluation_dataset(
        dataset
    )

    return {
        "average_queue_length": (
            compute_average_queue_length(
                dataset
            )
        ),
        "average_waiting_time": (
            compute_average_waiting_time(
                dataset
            )
        ),
        "average_speed": (
            compute_average_speed(
                dataset
            )
        ),
        "average_travel_time": (
            compute_average_travel_time(
                dataset
            )
        ),
        "total_throughput": (
            compute_total_throughput(
                dataset
            )
        ),
        "congestion_ratio": (
            compute_congestion_ratio(
                dataset
            )
        ),
        "spillback_events": (
            compute_spillback_events(
                dataset,
                occupancy_threshold=(
                    occupancy_threshold
                ),
                queue_threshold=(
                    queue_threshold
                ),
            )
        ),
    }
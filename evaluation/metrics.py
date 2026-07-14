"""
Traffic performance evaluation metrics.
"""

import pandas as pd


def compute_average_queue_length(dataset):
    return float(dataset["queue_length"].mean())


def compute_average_waiting_time(dataset):
    return float(dataset["waiting_time"].mean())


def compute_average_speed(dataset):
    return float(dataset["average_speed"].mean())


def compute_total_throughput(dataset):
    return float(dataset["departure_rate"].sum())


def compute_spillback_events(dataset, occupancy_threshold=0.9):
    return int((dataset["downstream_occupancy"] >= occupancy_threshold).sum())


def compute_congestion_ratio(dataset):
    return float((dataset["traffic_event_type"] == 2).sum() / len(dataset))


def summarize_metrics(dataset):
    if isinstance(dataset, str):
        dataset = pd.read_csv(dataset)

    return {
        "average_queue_length": compute_average_queue_length(dataset),
        "average_waiting_time": compute_average_waiting_time(dataset),
        "average_speed": compute_average_speed(dataset),
        "total_throughput": compute_total_throughput(dataset),
        "spillback_events": compute_spillback_events(dataset),
        "congestion_ratio": compute_congestion_ratio(dataset),
    }

"""
Plot basic traffic dataset metrics.
"""

import os
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PROCESSED_DATA_DIR, DATASET_FILENAME
from evaluation.metrics import summarize_metrics


def main():
    dataset_path = os.path.join(PROCESSED_DATA_DIR, DATASET_FILENAME)
    metrics = summarize_metrics(dataset_path)

    labels = [
        "Average Queue Length",
        "Average Waiting Time",
        "Average Speed",
        "Total Throughput",
        "Spillback Events",
        "Congestion Ratio",
    ]

    values = [
        metrics["average_queue_length"],
        metrics["average_waiting_time"],
        metrics["average_speed"],
        metrics["total_throughput"],
        metrics["spillback_events"],
        metrics["congestion_ratio"],
    ]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color="tab:blue")
    plt.xticks(rotation=45, ha="right")
    plt.title("Traffic Dataset Summary Metrics")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

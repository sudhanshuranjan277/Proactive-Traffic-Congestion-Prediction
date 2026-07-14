"""
Location-Wise Traffic Dataset Generator

Runs SUMO simulation and generates
dataset-ready traffic observations using:

SUMO -> TraCI -> Collector -> Pipeline -> CSV
"""

import os
import sys
import csv

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    PROCESSED_DATA_DIR,
    DEFAULT_LOCATION_ID,
    DATASET_FILENAME,
    OBSERVATION_WINDOW,
)
from environment.sumo_env import SumoEnvironment
from integration.collector import TrafficCollector
from integration.pipeline import TrafficPipeline

LOCATION_ID = DEFAULT_LOCATION_ID
SIMULATION_STEPS = 3600


def save_dataset(rows, output_file):
    """
    Save dataset rows to CSV.
    """

    if not rows:
        raise ValueError(
            "No dataset rows were generated."
        )

    fieldnames = [
        "location_id",
        "junction_id",
        "simulation_time",
        "vehicle_count",
        "traffic_flow",
        "arrival_rate",
        "departure_rate",
        "traffic_event_type",
        "remaining_green_time",
        "current_signal_phase",
        "downstream_occupancy",
        "downstream_queue_length",
        "average_speed",
        "waiting_time",
        "travel_time",
        "queue_length",
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def main():

    env = SumoEnvironment()

    collector = TrafficCollector()

    pipeline = TrafficPipeline(
        collector=collector,
        location_id=LOCATION_ID,
        observation_window=OBSERVATION_WINDOW,
    )

    dataset_rows = []

    env.show_configuration()

    if not env.connect():
        raise RuntimeError(
            "SUMO connection failed."
        )

    print(
        f"\nGenerating dataset for "
        f"{LOCATION_ID}...\n"
    )

    try:

        for _ in range(SIMULATION_STEPS):

            env.simulation_step()

            rows = pipeline.process_step()

            if rows:

                dataset_rows.extend(rows)

                print(
                    f"Simulation Time: "
                    f"{rows[0]['simulation_time']:.0f}s | "
                    f"Rows Generated: {len(rows)} | "
                    f"Total Rows: {len(dataset_rows)}"
                )

    finally:

        env.disconnect()

    output_file = os.path.join(
        PROCESSED_DATA_DIR,
        DATASET_FILENAME,
    )

    save_dataset(
        dataset_rows,
        output_file
    )

    print("\nDataset Generation Completed.")
    print(f"Dataset Rows : {len(dataset_rows)}")
    print(f"Dataset File : {output_file}")


if __name__ == "__main__":
    main()
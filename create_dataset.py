"""
Location-Wise Traffic Dataset Generator

Runs SUMO simulation and generates
dataset-ready traffic observations using:

SUMO -> TraCI -> Collector -> Pipeline -> CSV

Supports generating multiple independent episodes (different demand
each time, via scripts/generate_traffic_demand.py --seed) into
separate output files, which scripts/merge_datasets.py can then
combine into one larger training dataset.
"""

import os
import sys
import csv
import argparse

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
    SIMULATION_TIME,
)
from environment.sumo_env import SumoEnvironment
from integration.collector import TrafficCollector
from integration.pipeline import TrafficPipeline

SIMULATION_STEPS = SIMULATION_TIME


def parse_args():

    parser = argparse.ArgumentParser(
        description="Generate a traffic dataset from a SUMO simulation run."
    )

    parser.add_argument(
        "--location-id",
        default=DEFAULT_LOCATION_ID,
        help=(
            "Location ID written into every row (default: "
            f"'{DEFAULT_LOCATION_ID}' from config.py). Use a different "
            "value for each independent episode so preprocessing.py "
            "never mixes rows from different runs into one sequence."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path. Defaults to "
            f"'{DATASET_FILENAME}' under PROCESSED_DATA_DIR, but with "
            "--location-id set, defaults to '<location-id>_dataset.csv' "
            "instead, so repeated runs don't overwrite each other."
        ),
    )

    return parser.parse_args()


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

    args = parse_args()

    location_id = args.location_id

    if args.output:
        output_file = args.output
    elif location_id != DEFAULT_LOCATION_ID:
        output_file = os.path.join(
            PROCESSED_DATA_DIR,
            f"{location_id}_dataset.csv",
        )
    else:
        output_file = os.path.join(
            PROCESSED_DATA_DIR,
            DATASET_FILENAME,
        )

    env = SumoEnvironment()

    collector = TrafficCollector()

    pipeline = TrafficPipeline(
        collector=collector,
        location_id=location_id,
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
        f"{location_id}...\n"
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

    save_dataset(
        dataset_rows,
        output_file
    )

    print("\nDataset Generation Completed.")
    print(f"Dataset Rows : {len(dataset_rows)}")
    print(f"Dataset File : {output_file}")


if __name__ == "__main__":
    main()
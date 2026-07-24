"""
Traffic Controller Evaluation Runner

Evaluates traffic-control strategies using the
same SUMO network, traffic demand, collector,
pipeline, and evaluation metrics.

Supported modes:

1. fixed
   SUMO default traffic signal control.

2. proactive
   Trained LSTM future traffic predictor combined
   with the trained DDQN traffic signal controller.

Evaluation workflow:

SUMO
    ->
Traffic Collector
    ->
Traffic Pipeline
    ->
Traffic Control Strategy
    ->
Evaluation Observations
    ->
Traffic Performance Metrics

Important:
Reactive DDQN is not included here because the
current DDQN state contains LSTM future prediction
features. A genuine reactive DDQN requires a
separate current-state-only environment and model.
"""

import csv
import os
import sys


# ======================================
# Project Path
# ======================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


# ======================================
# Project Imports
# ======================================

from config import (
    DEFAULT_LOCATION_ID,
    DDQN_MODEL_FILENAME,
    LOOKBACK,
    LSTM_MODEL_FILENAME,
    LSTM_SCALER_FILENAME,
    MODEL_DIR,
    OBSERVATION_WINDOW,
    RL_EXTENSION_SECONDS,
    RL_GAMMA,
    RL_LEARNING_RATE,
    RL_MEMORY_CAPACITY,
    RL_NUM_ACTIONS,
    RL_TARGET_UPDATE_FREQUENCY,
    SIMULATION_TIME,
)

from environment.sumo_env import (
    SumoEnvironment,
)

from evaluation.metrics import (
    summarize_metrics,
)

from integration.collector import (
    TrafficCollector,
)

from integration.pipeline import (
    TrafficPipeline,
)

from prediction.predictor import (
    TrafficPredictor,
)

from rl.agent import (
    TrafficSignalAgent,
)

from rl.environment import (
    JunctionTrafficEnvironment,
)


# ======================================
# Model Paths
# ======================================

LSTM_MODEL_PATH = os.path.join(
    MODEL_DIR,
    LSTM_MODEL_FILENAME,
)

LSTM_SCALER_PATH = os.path.join(
    MODEL_DIR,
    LSTM_SCALER_FILENAME,
)

DDQN_MODEL_PATH = os.path.join(
    MODEL_DIR,
    DDQN_MODEL_FILENAME,
)


# ======================================
# Evaluation Output Paths
# ======================================

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "results",
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True,
)


# ======================================
# Dataset Schema
# ======================================

DATASET_FIELDS = [
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


# ======================================
# Validation
# ======================================

def validate_mode(mode):
    """
    Validate evaluation controller mode.
    """

    supported_modes = {
        "fixed",
        "proactive",
    }

    if mode not in supported_modes:
        raise ValueError(
            "Unsupported evaluation mode. "
            f"Expected one of "
            f"{sorted(supported_modes)}, "
            f"received '{mode}'."
        )


def validate_proactive_model_files():
    """
    Validate trained model artifacts required
    for proactive controller evaluation.
    """

    required_files = {
        "LSTM model": LSTM_MODEL_PATH,
        "LSTM scaler": LSTM_SCALER_PATH,
        "DDQN model": DDQN_MODEL_PATH,
    }

    for (
        artifact_name,
        artifact_path,
    ) in required_files.items():

        if not os.path.exists(
            artifact_path
        ):
            raise FileNotFoundError(
                f"{artifact_name} file "
                f"not found: "
                f"{artifact_path}"
            )


# ======================================
# Output Helpers
# ======================================

def save_evaluation_rows(
    rows,
    output_path,
):
    """
    Save collected traffic observations.
    """

    if not rows:
        raise ValueError(
            "No evaluation observations "
            "were collected."
        )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=DATASET_FIELDS,
        )

        writer.writeheader()

        for row in rows:

            output_row = {
                field: row[field]
                for field in DATASET_FIELDS
            }

            writer.writerow(
                output_row
            )


def save_metric_summary(
    metrics,
    output_path,
):
    """
    Save traffic metric summary.
    """

    if not metrics:
        raise ValueError(
            "Metric summary is empty."
        )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                metrics.keys()
            ),
        )

        writer.writeheader()

        writer.writerow(
            metrics
        )


# ======================================
# Fixed-Time Controller
# ======================================

def run_fixed_controller():
    """
    Evaluate the default SUMO traffic signal
    program without RL intervention.

    This acts as the fixed-time baseline.
    """

    sumo_env = SumoEnvironment()

    collector = TrafficCollector()

    pipeline = TrafficPipeline(
        collector=collector,
        location_id=DEFAULT_LOCATION_ID,
        observation_window=(
            OBSERVATION_WINDOW
        ),
    )

    evaluation_rows = []

    print(
        "\nStarting fixed-time "
        "baseline evaluation..."
    )

    if not sumo_env.connect():

        raise RuntimeError(
            "SUMO connection failed."
        )

    try:

        for _ in range(
            SIMULATION_TIME
        ):

            sumo_env.simulation_step()

            rows = pipeline.process_step()

            if rows:

                evaluation_rows.extend(
                    rows
                )

                print(
                    f"Simulation Time: "
                    f"{rows[0]['simulation_time']:.0f}s | "
                    f"Rows: "
                    f"{len(evaluation_rows)}"
                )

    finally:

        sumo_env.disconnect()

    return evaluation_rows


# ======================================
# Proactive DDQN Agent
# ======================================

def create_proactive_agent(
    state_dim,
):
    """
    Create and load the trained proactive
    DDQN traffic signal agent.
    """

    agent = TrafficSignalAgent(
        state_dim=state_dim,
        action_dim=RL_NUM_ACTIONS,
        memory_capacity=(
            RL_MEMORY_CAPACITY
        ),
        gamma=RL_GAMMA,
        learning_rate=(
            RL_LEARNING_RATE
        ),
        target_update_frequency=(
            RL_TARGET_UPDATE_FREQUENCY
        ),
    )

    agent.load(
        DDQN_MODEL_PATH
    )

    return agent


# ======================================
# Proactive Controller
# ======================================

def run_proactive_controller():
    """
    Evaluate proactive LSTM + DDQN traffic
    signal control.

    LSTM future predictions are embedded in the
    environment state.

    DDQN selects actions greedily using:

    epsilon = 0.0

    No DDQN learning is performed during
    evaluation.
    """

    validate_proactive_model_files()

    sumo_env = SumoEnvironment()

    collector = TrafficCollector()

    predictor = TrafficPredictor(
        model_path=LSTM_MODEL_PATH,
        scaler_path=LSTM_SCALER_PATH,
    )

    print(
        "\nStarting proactive "
        "LSTM + DDQN evaluation..."
    )

    if not sumo_env.connect():

        raise RuntimeError(
            "SUMO connection failed."
        )

    try:

        if not collector.initialized:

            collector.initialize_junctions()

        junction_ids = list(
            collector.junction_lanes.keys()
        )

        if not junction_ids:

            raise RuntimeError(
                "No signalized junctions "
                "were discovered."
            )

        junction_id = junction_ids[0]

        print(
            f"Evaluation Junction: "
            f"{junction_id}"
        )

        environment = (
            JunctionTrafficEnvironment(
                sumo_env=sumo_env,
                collector=collector,
                predictor=predictor,
                junction_id=junction_id,
                location_id=(
                    DEFAULT_LOCATION_ID
                ),
                lookback=LOOKBACK,
                observation_window=(
                    OBSERVATION_WINDOW
                ),
                max_steps=None,
                extension_seconds=(
                    RL_EXTENSION_SECONDS
                ),
            )
        )

        print(
            "Initializing LSTM "
            "traffic history..."
        )

        state = environment.reset()

        state_dim = int(
            state.shape[0]
        )

        print(
            f"State Dimension: "
            f"{state_dim}"
        )

        print(
            f"Control Start Time: "
            f"{environment.previous_row['simulation_time']:.0f}s"
        )

        agent = create_proactive_agent(
            state_dim=state_dim
        )

        action_counts = {
            action: 0
            for action in range(
                RL_NUM_ACTIONS
            )
        }

        control_steps = 0

        while True:

            current_time = float(
                environment.previous_row[
                    "simulation_time"
                ]
            )

            next_control_time = (
                current_time
                + OBSERVATION_WINDOW
            )

            if (
                next_control_time
                > SIMULATION_TIME
            ):
                break

            action = agent.select_action(
                state,
                epsilon=0.0,
            )

            (
                next_state,
                reward,
                done,
                info,
            ) = environment.step(
                action
            )

            action_counts[action] += 1

            control_steps += 1

            state = next_state

            print(
                f"Control Step: "
                f"{control_steps} | "
                f"Action: "
                f"{action} | "
                f"Executed: "
                f"{info['action_valid']} | "
                f"Reward: "
                f"{reward:.4f} | "
                f"Time: "
                f"{info['simulation_time']:.0f}s"
            )

            if done:
                break

        evaluation_rows = list(
            environment.history
        )

        print(
            "\nProactive Action "
            "Distribution:"
        )

        for (
            action,
            count,
        ) in action_counts.items():

            print(
                f"Action {action}: "
                f"{count}"
            )

    finally:

        sumo_env.disconnect()

    return evaluation_rows


# ======================================
# Controller Evaluation
# ======================================

def evaluate_controller(
    mode,
):
    """
    Execute a traffic controller experiment and
    calculate performance metrics.
    """

    validate_mode(
        mode
    )

    print(
        "=" * 60
    )

    print(
        "Traffic Controller Evaluation"
    )

    print(
        "=" * 60
    )

    print(
        f"Evaluation Mode    : "
        f"{mode}"
    )

    print(
        f"Simulation Time    : "
        f"{SIMULATION_TIME}s"
    )

    print(
        f"Observation Window : "
        f"{OBSERVATION_WINDOW}s"
    )

    print(
        "=" * 60
    )

    if mode == "fixed":

        evaluation_rows = (
            run_fixed_controller()
        )

    elif mode == "proactive":

        evaluation_rows = (
            run_proactive_controller()
        )

    else:

        raise RuntimeError(
            "Controller mode dispatch failed."
        )

    dataset_path = os.path.join(
        RESULTS_DIR,
        f"{mode}_evaluation.csv",
    )

    metrics_path = os.path.join(
        RESULTS_DIR,
        f"{mode}_metrics.csv",
    )

    save_evaluation_rows(
        evaluation_rows,
        dataset_path,
    )

    metrics = summarize_metrics(
        dataset_path
    )

    save_metric_summary(
        metrics,
        metrics_path,
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "Evaluation Completed"
    )

    print(
        "=" * 60
    )

    print(
        f"Controller Mode : "
        f"{mode}"
    )

    print(
        f"Observations    : "
        f"{len(evaluation_rows)}"
    )

    print(
        "\nPerformance Metrics"
    )

    print(
        "-" * 60
    )

    for (
        metric_name,
        metric_value,
    ) in metrics.items():

        print(
            f"{metric_name:<30}"
            f"{metric_value:.6f}"
        )

    print(
        "\nEvaluation Dataset:"
    )

    print(
        dataset_path
    )

    print(
        "\nMetric Summary:"
    )

    print(
        metrics_path
    )

    return metrics


# ======================================
# Main
# ======================================

def main():

    if len(sys.argv) != 2:

        raise SystemExit(
            "\nUsage:\n\n"
            "python evaluation\\"
            "evaluate_controller.py "
            "<fixed|proactive>\n\n"
            "Examples:\n\n"
            "python evaluation\\"
            "evaluate_controller.py fixed\n\n"
            "python evaluation\\"
            "evaluate_controller.py proactive"
        )

    mode = (
        sys.argv[1]
        .strip()
        .lower()
    )

    evaluate_controller(
        mode
    )


if __name__ == "__main__":
    main()
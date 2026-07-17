"""
Traffic Controller Performance Comparison

Compares traffic performance metrics between:

1. Fixed-Time Traffic Signal Control
2. Proactive LSTM + DDQN Traffic Signal Control

The comparison uses evaluation metric files generated
by evaluation/evaluate_controller.py.

Comparison workflow:

Fixed Metrics
    +
Proactive Metrics
    ->
Metric Comparison
    ->
Percentage Improvement
    ->
Comparison CSV

No traffic performance result is hard-coded.
All comparison values are derived from actual
evaluation outputs.
"""

import csv
import math
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
# Result Paths
# ======================================

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "results",
)

FIXED_METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "fixed_metrics.csv",
)

PROACTIVE_METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "proactive_metrics.csv",
)

COMPARISON_PATH = os.path.join(
    RESULTS_DIR,
    "controller_comparison.csv",
)


# ======================================
# Metric Configuration
# ======================================

METRIC_DIRECTIONS = {
    "average_queue_length": "lower",
    "average_waiting_time": "lower",
    "average_speed": "higher",
    "average_travel_time": "lower",
    "total_throughput": "higher",
    "congestion_ratio": "lower",
    "spillback_events": "lower",
}


METRIC_LABELS = {
    "average_queue_length": (
        "Average Queue Length"
    ),
    "average_waiting_time": (
        "Average Waiting Time"
    ),
    "average_speed": (
        "Average Speed"
    ),
    "average_travel_time": (
        "Average Travel Time"
    ),
    "total_throughput": (
        "Total Throughput"
    ),
    "congestion_ratio": (
        "Congestion Ratio"
    ),
    "spillback_events": (
        "Spillback Events"
    ),
}


# ======================================
# Metric Loading
# ======================================

def load_metrics(file_path):
    """
    Load a single-row evaluation metric CSV.
    """

    if not os.path.exists(
        file_path
    ):
        raise FileNotFoundError(
            "Metric file not found: "
            f"{file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8",
        newline="",
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )

        rows = list(
            reader
        )

    if len(rows) != 1:
        raise ValueError(
            "Metric file must contain exactly "
            "one metric row: "
            f"{file_path}"
        )

    raw_metrics = rows[0]

    missing_metrics = (
        set(METRIC_DIRECTIONS)
        - set(raw_metrics)
    )

    if missing_metrics:
        raise ValueError(
            "Metric file is missing required "
            "metrics: "
            f"{sorted(missing_metrics)}"
        )

    metrics = {}

    for metric_name in (
        METRIC_DIRECTIONS
    ):

        try:

            metric_value = float(
                raw_metrics[
                    metric_name
                ]
            )

        except (
            TypeError,
            ValueError,
        ) as error:

            raise ValueError(
                "Invalid numeric value for "
                f"metric '{metric_name}' "
                f"in file: {file_path}"
            ) from error

        if not math.isfinite(
            metric_value
        ):
            raise ValueError(
                "Non-finite value for "
                f"metric '{metric_name}' "
                f"in file: {file_path}"
            )

        metrics[
            metric_name
        ] = metric_value

    return metrics


# ======================================
# Improvement Calculation
# ======================================

def calculate_improvement(
    fixed_value,
    proactive_value,
    direction,
):
    """
    Calculate proactive controller improvement
    relative to the fixed-time baseline.

    For lower-is-better metrics:

        improvement =
        (fixed - proactive) / fixed * 100

    For higher-is-better metrics:

        improvement =
        (proactive - fixed) / fixed * 100

    Positive value:
        proactive controller improved.

    Negative value:
        proactive controller degraded.

    None:
        percentage improvement cannot be defined
        because the fixed baseline is zero.
    """

    if fixed_value == 0.0:

        return None

    if direction == "lower":

        improvement = (
            (
                fixed_value
                - proactive_value
            )
            / abs(
                fixed_value
            )
            * 100.0
        )

    elif direction == "higher":

        improvement = (
            (
                proactive_value
                - fixed_value
            )
            / abs(
                fixed_value
            )
            * 100.0
        )

    else:

        raise ValueError(
            "Unsupported metric direction: "
            f"{direction}"
        )

    return float(
        improvement
    )


def determine_result(
    fixed_value,
    proactive_value,
    direction,
):
    """
    Determine whether proactive control improved,
    degraded, or matched the fixed baseline.
    """

    if math.isclose(
        fixed_value,
        proactive_value,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        return "UNCHANGED"

    if direction == "lower":

        if proactive_value < fixed_value:
            return "IMPROVED"

        return "DEGRADED"

    if direction == "higher":

        if proactive_value > fixed_value:
            return "IMPROVED"

        return "DEGRADED"

    raise ValueError(
        "Unsupported metric direction: "
        f"{direction}"
    )


# ======================================
# Controller Comparison
# ======================================

def compare_controllers(
    fixed_metrics,
    proactive_metrics,
):
    """
    Compare fixed and proactive controller metrics.
    """

    comparison_rows = []

    for (
        metric_name,
        direction,
    ) in METRIC_DIRECTIONS.items():

        fixed_value = float(
            fixed_metrics[
                metric_name
            ]
        )

        proactive_value = float(
            proactive_metrics[
                metric_name
            ]
        )

        improvement = (
            calculate_improvement(
                fixed_value,
                proactive_value,
                direction,
            )
        )

        result = determine_result(
            fixed_value,
            proactive_value,
            direction,
        )

        comparison_rows.append(
            {
                "metric": metric_name,
                "metric_label": (
                    METRIC_LABELS[
                        metric_name
                    ]
                ),
                "better_direction": (
                    direction
                ),
                "fixed_value": (
                    fixed_value
                ),
                "proactive_value": (
                    proactive_value
                ),
                "improvement_percent": (
                    improvement
                ),
                "result": result,
            }
        )

    return comparison_rows


# ======================================
# Save Comparison
# ======================================

def save_comparison(
    comparison_rows,
    output_path,
):
    """
    Save controller comparison results.
    """

    if not comparison_rows:
        raise ValueError(
            "Controller comparison is empty."
        )

    fieldnames = [
        "metric",
        "metric_label",
        "better_direction",
        "fixed_value",
        "proactive_value",
        "improvement_percent",
        "result",
    ]

    with open(
        output_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            comparison_rows
        )


# ======================================
# Console Summary
# ======================================

def print_comparison(
    comparison_rows,
):
    """
    Display controller comparison summary.
    """

    print(
        "\n"
        + "=" * 100
    )

    print(
        "Fixed-Time vs Proactive "
        "LSTM + DDQN Comparison"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Metric':<28}"
        f"{'Fixed':>14}"
        f"{'Proactive':>14}"
        f"{'Improvement':>16}"
        f"{'Result':>16}"
    )

    print(
        "-" * 100
    )

    for row in comparison_rows:

        improvement = row[
            "improvement_percent"
        ]

        if improvement is None:

            improvement_text = "N/A"

        else:

            improvement_text = (
                f"{improvement:.2f}%"
            )

        print(
            f"{row['metric_label']:<28}"
            f"{row['fixed_value']:>14.6f}"
            f"{row['proactive_value']:>14.6f}"
            f"{improvement_text:>16}"
            f"{row['result']:>16}"
        )

    print(
        "=" * 100
    )


def print_result_summary(
    comparison_rows,
):
    """
    Display aggregate comparison counts.
    """

    improved_count = sum(
        row["result"] == "IMPROVED"
        for row in comparison_rows
    )

    degraded_count = sum(
        row["result"] == "DEGRADED"
        for row in comparison_rows
    )

    unchanged_count = sum(
        row["result"] == "UNCHANGED"
        for row in comparison_rows
    )

    print(
        "\nComparison Summary"
    )

    print(
        "-" * 60
    )

    print(
        f"Improved Metrics  : "
        f"{improved_count}"
    )

    print(
        f"Degraded Metrics  : "
        f"{degraded_count}"
    )

    print(
        f"Unchanged Metrics : "
        f"{unchanged_count}"
    )

    print(
        f"Total Metrics     : "
        f"{len(comparison_rows)}"
    )


# ======================================
# Main
# ======================================

def main():

    print(
        "=" * 60
    )

    print(
        "Traffic Controller Model Comparison"
    )

    print(
        "=" * 60
    )

    print(
        "\nFixed Metric File:"
    )

    print(
        FIXED_METRICS_PATH
    )

    print(
        "\nProactive Metric File:"
    )

    print(
        PROACTIVE_METRICS_PATH
    )

    fixed_metrics = load_metrics(
        FIXED_METRICS_PATH
    )

    proactive_metrics = load_metrics(
        PROACTIVE_METRICS_PATH
    )

    comparison_rows = (
        compare_controllers(
            fixed_metrics,
            proactive_metrics,
        )
    )

    print_comparison(
        comparison_rows
    )

    print_result_summary(
        comparison_rows
    )

    save_comparison(
        comparison_rows,
        COMPARISON_PATH,
    )

    print(
        "\nComparison File:"
    )

    print(
        COMPARISON_PATH
    )


if __name__ == "__main__":
    main()
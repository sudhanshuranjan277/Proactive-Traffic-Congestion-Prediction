"""
Performance Report Charts

Builds Fixed vs Proactive controller comparison visualizations from
REAL evaluation output files — never hard-coded values, per the
principle already stated in evaluation/compare_models.py's docstring.

Data sources (all produced by running, in order:
evaluation/evaluate_controller.py --mode fixed,
evaluation/evaluate_controller.py --mode proactive,
evaluation/compare_models.py):

  outputs/results/controller_comparison.csv  — summary metrics + improvement %
  outputs/results/fixed_evaluation.csv        — per-timestep raw observations (fixed)
  outputs/results/proactive_evaluation.csv    — per-timestep raw observations (proactive)

If these don't exist yet, every loader below returns None so the
caller can show a clear "run evaluation first" message instead of
fabricating placeholder numbers.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import EVALUATION_RESULTS_DIR

# Reuse the SAME metric definitions compare_models.py already uses,
# instead of redefining/hardcoding them a second time here.
from evaluation.compare_models import METRIC_DIRECTIONS, METRIC_LABELS


COMPARISON_PATH = os.path.join(EVALUATION_RESULTS_DIR, "controller_comparison.csv")
FIXED_EVAL_PATH = os.path.join(EVALUATION_RESULTS_DIR, "fixed_evaluation.csv")
PROACTIVE_EVAL_PATH = os.path.join(EVALUATION_RESULTS_DIR, "proactive_evaluation.csv")


# ==============================================================
# Loaders — return None (not fabricated data) if files don't exist
# ==============================================================

def load_comparison():
    if not os.path.exists(COMPARISON_PATH):
        return None
    df = pd.read_csv(COMPARISON_PATH)
    for col in ("fixed_value", "proactive_value", "improvement_percent"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_timeseries():
    if not (os.path.exists(FIXED_EVAL_PATH) and os.path.exists(PROACTIVE_EVAL_PATH)):
        return None, None
    fixed_df = pd.read_csv(FIXED_EVAL_PATH)
    proactive_df = pd.read_csv(PROACTIVE_EVAL_PATH)
    return fixed_df, proactive_df


# ==============================================================
# Normalization helper for the radar chart (2 series: Fixed, Proactive)
# ==============================================================

def _normalize_pair(fixed_val, proactive_val, lower_is_better):
    values = [fixed_val, proactive_val]
    lo, hi = min(values), max(values)
    if hi == lo:
        return 0.5, 0.5
    if lower_is_better:
        return (hi - fixed_val) / (hi - lo), (hi - proactive_val) / (hi - lo)
    return (fixed_val - lo) / (hi - lo), (proactive_val - lo) / (hi - lo)


# ==============================================================
# Figure builders — each returns a matplotlib Figure
# ==============================================================

def build_metric_bar_figure(row):

    fig, ax = plt.subplots(figsize=(6, 4.5))

    labels = ["Fixed", "Proactive"]
    values = [row["fixed_value"], row["proactive_value"]]
    colors = ["#4C72B0", "#55A868"]

    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="black", linewidth=1.0)

    ax.set_title(row["metric_label"], fontsize=14, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    ymax = max(values) * 1.2 if max(values) > 0 else 1
    ax.set_ylim(0, ymax)

    for bar, val in zip(bars, values):
        txt = f"{val:.3f}" if val < 1 else f"{val:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.02,
                 txt, ha="center", fontsize=10, weight="bold")

    fig.tight_layout()
    return fig


def build_timeseries_figure(fixed_df, proactive_df, column, ylabel):

    fixed_trend = fixed_df.groupby("simulation_time")[column].mean()
    proactive_trend = proactive_df.groupby("simulation_time")[column].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fixed_trend.index, fixed_trend.values, label="Fixed", linewidth=2)
    ax.plot(proactive_trend.index, proactive_trend.values, label="Proactive", linewidth=2)

    ax.set_title(f"{ylabel} vs Simulation Time", fontsize=14, weight="bold")
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    return fig


def build_radar_figure(comparison_df):

    labels, fixed_scores, proactive_scores = [], [], []

    for _, row in comparison_df.iterrows():
        lower_is_better = METRIC_DIRECTIONS.get(row["metric"], "lower") == "lower"
        f_score, p_score = _normalize_pair(row["fixed_value"], row["proactive_value"], lower_is_better)
        labels.append(row["metric_label"])
        fixed_scores.append(f_score)
        proactive_scores.append(p_score)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    fixed_scores += fixed_scores[:1]
    proactive_scores += proactive_scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    ax.plot(angles, fixed_scores, linewidth=2, label="Fixed", color="#C44E52", marker="s")
    ax.fill(angles, fixed_scores, "#C44E52", alpha=0.1)

    ax.plot(angles, proactive_scores, linewidth=2, label="Proactive", color="#55A868", marker="^")
    ax.fill(angles, proactive_scores, "#55A868", alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1 (Best)"], color="gray", size=8)

    ax.set_title("Controller Performance (Normalized)", size=15, y=1.1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.tight_layout()
    return fig


def build_improvement_figure(comparison_df):

    fig, ax = plt.subplots(figsize=(9, 6))

    names = comparison_df["metric_label"].tolist()
    values = comparison_df["improvement_percent"].tolist()
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))

    bars = ax.barh(names, values, color=colors, edgecolor="black")

    ax.set_title("Overall Improvement (%) — Proactive vs Fixed", fontsize=15, weight="bold")
    ax.set_xlabel("Improvement (%)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, values):
        ax.text(val + (1 if val >= 0 else -1), bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}%", va="center", fontsize=10, weight="bold",
                 ha="left" if val >= 0 else "right")

    fig.tight_layout()
    return fig


# ==============================================================
# Full PDF report (mirrors what was generated standalone before,
# now driven by real data + reusable from the dashboard)
# ==============================================================

def build_pdf_report(output_path):

    comparison_df = load_comparison()
    if comparison_df is None:
        raise FileNotFoundError(
            f"No comparison data at {COMPARISON_PATH}. "
            "Run evaluate_controller.py (fixed + proactive) and "
            "compare_models.py first."
        )

    fixed_df, proactive_df = load_timeseries()

    with PdfPages(output_path) as pdf:

        for _, row in comparison_df.iterrows():
            fig = build_metric_bar_figure(row)
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

        if fixed_df is not None and proactive_df is not None:
            for column, label in [
                ("queue_length", "Average Queue Length"),
                ("waiting_time", "Average Waiting Time"),
                ("average_speed", "Average Speed"),
            ]:
                if column in fixed_df.columns and column in proactive_df.columns:
                    fig = build_timeseries_figure(fixed_df, proactive_df, column, label)
                    pdf.savefig(fig, dpi=300)
                    plt.close(fig)

        fig = build_radar_figure(comparison_df)
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

        fig = build_improvement_figure(comparison_df)
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

    return output_path

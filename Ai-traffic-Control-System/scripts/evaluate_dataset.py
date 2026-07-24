"""
Dataset evaluation script.

Loads a processed traffic dataset and computes summary metrics.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PROCESSED_DATA_DIR, DATASET_FILENAME
from evaluation.metrics import summarize_metrics


def main():
    dataset_path = os.path.join(PROCESSED_DATA_DIR, DATASET_FILENAME)
    metrics = summarize_metrics(dataset_path)

    print("=" * 60)
    print("Dataset Evaluation")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"{key.replace('_', ' ').title()}: {value:.4f}")


if __name__ == "__main__":
    main()

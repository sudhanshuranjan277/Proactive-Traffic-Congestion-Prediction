"""
window_generator.py

Author: Sudhanshu Ranjan
Project: TransGTR Baseline (KDD 2023)

Description
-----------
Generates sliding windows for traffic forecasting.

Responsibilities
----------------
- Generate input sequences
- Generate target sequences
- Support configurable input/output lengths
- Convert data into NumPy arrays

This module DOES NOT:
- Load datasets
- Normalize data
- Build graphs
"""

from typing import Tuple

import numpy as np
import pandas as pd


class WindowGenerator:
    """
    Sliding window generator for time-series forecasting.
    """

    def __init__(
        self,
        input_length: int = 12,
        prediction_length: int = 12,
        stride: int = 1,
    ) -> None:

        self.input_length = input_length
        self.prediction_length = prediction_length
        self.stride = stride

    def generate(
        self,
        dataframe: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate input-output sliding windows.

        Returns
        -------
        X : np.ndarray
            Input sequences

        Y : np.ndarray
            Target sequences
        """

        data = dataframe.values

        x_windows = []
        y_windows = []

        total_length = (
            self.input_length +
            self.prediction_length
        )

        for start in range(
            0,
            len(data) - total_length + 1,
            self.stride,
        ):

            end_input = start + self.input_length

            end_target = (
                end_input +
                self.prediction_length
            )

            x = data[start:end_input]

            y = data[end_input:end_target]

            x_windows.append(x)

            y_windows.append(y)

        return (
            np.array(x_windows),
            np.array(y_windows),
        )

    def summary(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Display generated window information.
        """

        print("=" * 50)
        print("Sliding Window Summary")
        print("=" * 50)

        print(f"Input Shape : {x.shape}")
        print(f"Target Shape: {y.shape}")

        print("=" * 50)
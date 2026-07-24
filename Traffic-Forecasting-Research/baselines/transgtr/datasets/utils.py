"""
utils.py

Dataset helper functions.
"""

import numpy as np


def train_val_test_split(
    num_samples: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
):
    """
    Generate train/validation/test indices.
    """

    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)

    indices = np.arange(num_samples)

    return (
        indices[:train_end],
        indices[train_end:val_end],
        indices[val_end:],
    )


def check_nan(array):
    """
    Check whether array contains NaN values.
    """

    return np.isnan(array).any()
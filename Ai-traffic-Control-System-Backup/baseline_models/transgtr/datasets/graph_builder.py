"""
graph_builder.py

Author: Sudhanshu Ranjan
Project: TransGTR Baseline (KDD 2023)

Description
-----------
Builds graph structures for graph-based traffic forecasting.

Responsibilities
----------------
- Load adjacency matrix
- Add self-loops
- Normalize adjacency matrix
- Validate graph
- Return graph for downstream models

This module DOES NOT:
- Load datasets
- Normalize features
- Generate windows
"""

from pathlib import Path

import numpy as np


class GraphBuilder:
    """
    Graph construction utility.
    """

    def __init__(self, adjacency_path: str):

        self.adjacency_path = Path(adjacency_path)

    def load_adjacency(self) -> np.ndarray:
        """
        Load adjacency matrix.
        """

        if not self.adjacency_path.exists():
            raise FileNotFoundError(
                f"Adjacency file not found: {self.adjacency_path}"
            )

        adjacency = np.load(self.adjacency_path)

        return adjacency

    def add_self_loops(
        self,
        adjacency: np.ndarray,
    ) -> np.ndarray:
        """
        Add self loops.
        """

        adjacency = adjacency.copy()

        adjacency += np.eye(adjacency.shape[0])

        return adjacency

    def normalize(
        self,
        adjacency: np.ndarray,
    ) -> np.ndarray:
        """
        Symmetric normalization.

        A_hat = D^(-1/2) A D^(-1/2)
        """

        degree = np.sum(adjacency, axis=1)

        degree[degree == 0] = 1

        d_inv_sqrt = np.diag(
            np.power(degree, -0.5)
        )

        normalized = (
            d_inv_sqrt
            @ adjacency
            @ d_inv_sqrt
        )

        return normalized

    def validate(
        self,
        adjacency: np.ndarray,
    ) -> None:
        """
        Validate adjacency matrix.
        """

        if adjacency.ndim != 2:
            raise ValueError(
                "Adjacency matrix must be 2-dimensional."
            )

        rows, cols = adjacency.shape

        if rows != cols:
            raise ValueError(
                "Adjacency matrix must be square."
            )

    def build(self) -> np.ndarray:
        """
        Complete graph construction pipeline.
        """

        adjacency = self.load_adjacency()

        self.validate(adjacency)

        adjacency = self.add_self_loops(adjacency)

        adjacency = self.normalize(adjacency)

        return adjacency

    def summary(
        self,
        adjacency: np.ndarray,
    ) -> None:

        print("=" * 50)
        print("Graph Summary")
        print("=" * 50)
        print(f"Nodes : {adjacency.shape[0]}")
        print(f"Shape : {adjacency.shape}")
        print("=" * 50)
"""
dataset.py

Author: Sudhanshu Ranjan
Project: TransGTR Baseline (KDD 2023)

Description
-----------
PyTorch Dataset implementation for TransGTR.

Responsibilities
----------------
- Load raw dataset
- Preprocess dataset
- Generate sliding windows
- Build graph
- Provide samples to DataLoader

This module integrates the complete dataset pipeline.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from datasets.dataset_loader import DatasetLoader
from datasets.preprocessor import DataPreprocessor
from datasets.window_generator import WindowGenerator
from datasets.graph_builder import GraphBuilder


class TrafficDataset(Dataset):
    """
    Traffic forecasting dataset for TransGTR.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: str,
        adjacency_path: str,
        input_length: int = 12,
        prediction_length: int = 12,
        timestamp_column: str | None = None,
    ):

        # -----------------------------
        # Load Dataset
        # -----------------------------
        loader = DatasetLoader(
            dataset_name=dataset_name,
            dataset_path=Path(dataset_path),
        )

        dataframe = loader.load()

        # -----------------------------
        # Preprocess Dataset
        # -----------------------------
        preprocessor = DataPreprocessor()

        dataframe = preprocessor.preprocess(
            dataframe,
            timestamp_column=timestamp_column,
        )

        # -----------------------------
        # Sliding Window Generation
        # -----------------------------
        window_generator = WindowGenerator(
            input_length=input_length,
            prediction_length=prediction_length,
        )

        self.inputs, self.targets = (
            window_generator.generate(dataframe)
        )

        # -----------------------------
        # Graph Construction
        # -----------------------------
        graph_builder = GraphBuilder(
            adjacency_path=adjacency_path,
        )

        self.adjacency = graph_builder.build()

    def __len__(self):
        """
        Number of training samples.
        """
        return len(self.inputs)

    def __getitem__(self, index):
        """
        Return one sample.
        """

        x = torch.tensor(
            self.inputs[index],
            dtype=torch.float32,
        )

        y = torch.tensor(
            self.targets[index],
            dtype=torch.float32,
        )

        adjacency = torch.tensor(
            self.adjacency,
            dtype=torch.float32,
        )

        return {
            "inputs": x,
            "targets": y,
            "adjacency": adjacency,
        }
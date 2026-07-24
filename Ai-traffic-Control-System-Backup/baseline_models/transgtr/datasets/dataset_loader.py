"""
dataset_loader.py

Author: Sudhanshu Ranjan
Project: TransGTR Baseline (KDD 2023)

Description:
------------
Loads raw traffic datasets for TransGTR.

Supported datasets:
    - METR-LA
    - PEMS-BAY
    - PEMSD7M
    - SUMO (Future)

Responsibilities:
    - Validate dataset path
    - Load raw dataset
    - Return pandas DataFrame
    - Basic dataset information

This module DOES NOT:
    - Normalize data
    - Generate windows
    - Build graphs
Those responsibilities belong to their respective modules.
"""

from pathlib import Path
from typing import Union

import pandas as pd


class DatasetLoader:
    """
    Dataset loader for traffic forecasting datasets.
    """

    SUPPORTED_DATASETS = (
        "METR-LA",
        "PEMS-BAY",
        "PEMSD7M",
        "SUMO",
    )

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Union[str, Path],
    ) -> None:

        self.dataset_name = dataset_name.upper()
        self.dataset_path = Path(dataset_path)

        self._validate_dataset()

    def _validate_dataset(self) -> None:
        """
        Validate dataset configuration.
        """

        if self.dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {self.dataset_name}"
            )

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

    def load(self) -> pd.DataFrame:
        """
        Load dataset.

        Returns
        -------
        pandas.DataFrame
            Raw dataset.
        """

        suffix = self.dataset_path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(self.dataset_path)

        raise ValueError(
            f"Unsupported file format: {suffix}"
        )

    def dataset_info(self, dataframe: pd.DataFrame) -> None:
        """
        Print dataset summary.
        """

        print("=" * 50)
        print(f"Dataset : {self.dataset_name}")
        print(f"Rows    : {len(dataframe)}")
        print(f"Columns : {len(dataframe.columns)}")
        print("=" * 50)
        print(dataframe.info())
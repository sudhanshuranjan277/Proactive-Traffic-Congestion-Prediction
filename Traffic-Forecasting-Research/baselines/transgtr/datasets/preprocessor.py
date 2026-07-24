"""
preprocessor.py

Author: Sudhanshu Ranjan
Project: TransGTR Baseline (KDD 2023)

Description
-----------
Preprocesses raw traffic datasets before model training.

Responsibilities
----------------
- Handle missing values
- Remove duplicate records
- Sort by timestamp
- Normalize features
- Validate processed data

This module DOES NOT:
- Load datasets
- Generate sliding windows
- Build graphs
"""

from typing import Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """
    Data preprocessing class.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def remove_duplicates(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows.
        """
        return dataframe.drop_duplicates()

    def fill_missing_values(
        self,
        dataframe: pd.DataFrame,
        method: str = "forward_fill",
    ) -> pd.DataFrame:
        """
        Fill missing values.

        Supported methods:
            - forward_fill
            - backward_fill
            - zero
        """

        if method == "forward_fill":
            return dataframe.ffill()

        if method == "backward_fill":
            return dataframe.bfill()

        if method == "zero":
            return dataframe.fillna(0)

        raise ValueError(f"Unknown fill method: {method}")

    def sort_by_timestamp(
        self,
        dataframe: pd.DataFrame,
        timestamp_column: str,
    ) -> pd.DataFrame:
        """
        Sort dataframe by timestamp.
        """

        if timestamp_column not in dataframe.columns:
            raise ValueError(
                f"{timestamp_column} not found."
            )

        return dataframe.sort_values(timestamp_column)

    def normalize(
        self,
        dataframe: pd.DataFrame,
        columns: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Normalize numeric columns.
        """

        dataframe = dataframe.copy()

        if columns is None:
            columns = dataframe.select_dtypes(
                include="number"
            ).columns.tolist()

        dataframe[columns] = self.scaler.fit_transform(
            dataframe[columns]
        )

        return dataframe

    def preprocess(
        self,
        dataframe: pd.DataFrame,
        timestamp_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        """

        dataframe = self.remove_duplicates(dataframe)

        dataframe = self.fill_missing_values(
            dataframe,
            method="forward_fill",
        )

        if timestamp_column is not None:
            dataframe = self.sort_by_timestamp(
                dataframe,
                timestamp_column,
            )

        dataframe = self.normalize(dataframe)

        return dataframe
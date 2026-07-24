"""
embedding.py

Author: Sudhanshu Ranjan

Description
-----------
Embedding module for TransGTR.

Responsibilities
----------------
1. Project input features to embedding dimension.
2. Learn temporal representations.
3. Add positional encoding.
4. Return embeddings for transformer encoder.
"""

from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding.
    """

    def __init__(
        self,
        max_length: int,
        embed_dim: int,
    ) -> None:

        super().__init__()

        self.position = nn.Parameter(
            torch.randn(
                1,
                max_length,
                embed_dim,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        length = x.size(1)

        return x + self.position[:, :length]


class DataEmbedding(nn.Module):
    """
    Data Embedding Module.

    Input
    -----
    (Batch, Sequence Length, Features)

    Output
    ------
    (Batch, Sequence Length, Embed Dimension)
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        max_length: int = 512,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.value_embedding = nn.Linear(
            input_dim,
            embed_dim,
        )

        self.position_embedding = PositionalEncoding(
            max_length=max_length,
            embed_dim=embed_dim,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.value_embedding(x)

        x = self.position_embedding(x)

        x = self.dropout(x)

        return x
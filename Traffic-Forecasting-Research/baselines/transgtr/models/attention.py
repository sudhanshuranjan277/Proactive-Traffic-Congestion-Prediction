"""
attention.py

Multi-Head Self Attention module.
"""

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):

        output, attention_weights = self.attention(
            x,
            x,
            x,
            need_weights=True,
        )

        return output, attention_weights
    
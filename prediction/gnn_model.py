"""
Graph Transformer for multi-junction traffic prediction.

Unlike prediction/lstm.py (which forecasts one junction at a time,
with zero awareness of other junctions), this model forecasts ALL
junctions jointly, letting them attend to each other — a genuine
spatial-modeling upgrade, not just an architecture swap.

Design (deliberately dependency-light — no torch_geometric):
- Each junction is treated as a token.
- A GRU encodes each junction's own recent history independently
  (temporal modeling, same idea as the LSTM baseline).
- A Transformer encoder then lets junction-tokens attend to each
  other (spatial modeling). With no adjacency mask, this is
  self-attention over a fully-connected graph — i.e. the model
  learns which junctions matter to each other, rather than relying
  on a hand-specified topology (matches the "Graph Transformer"
  framing from the reference papers, adapted to data we actually
  have: no per-lane occupancy or multi-city data required).
- A per-junction output head produces the multi-step, multi-target
  forecast, exactly matching prediction/lstm.py's I/O contract.
"""

import torch
import torch.nn as nn
import math


class TrafficGraphTransformer(nn.Module):

    def __init__(
        self,
        input_size,
        target_size,
        hidden_size=64,
        num_layers=2,
        prediction_horizon=10,
        num_heads=4,
        num_attention_layers=2,
        dropout=0.2,
    ):

        super().__init__()

        self.hidden_size = hidden_size
        self.prediction_horizon = prediction_horizon
        self.target_size = target_size

        temporal_dropout = dropout if num_layers > 1 else 0.0

        # Per-junction temporal encoder (shared weights across junctions).
        self.temporal_encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=temporal_dropout,
        )

        # Cross-junction spatial encoder (self-attention = graph attention
        # over a fully-connected graph of junctions).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.graph_attention = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_attention_layers,
        )

        self.output_head = nn.Linear(
            hidden_size,
            prediction_horizon * target_size,
        )

        # Learnable positional embedding for junction tokens
        self.position_embedding = nn.Parameter(
            torch.randn(1, 512, hidden_size) * 0.02
        )

        self._initialize_weights()

    def forward(self, x):
        """
        x: (batch, num_junctions, lookback, input_size)
        returns: (batch, num_junctions, prediction_horizon, target_size)
        """

        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D input, got {x.shape}"
            )

        x = x.float()

        batch_size, num_junctions, lookback, input_size = x.shape

        # Encode each junction's own history independently.
        x = x.contiguous().view(batch_size * num_junctions, lookback, input_size)
        _, h_n = self.temporal_encoder(x)
        node_embeddings = h_n[-1]  # (batch*num_junctions, hidden_size)
        node_embeddings = node_embeddings.contiguous().view(
            batch_size, num_junctions, self.hidden_size
        )

        # Add positional information before attention
        node_embeddings = (
            node_embeddings
            + self.position_embedding[:, :num_junctions, :]
        )

        # Let junctions attend to each other.
        attended = self.graph_attention(node_embeddings)

        predictions = self.output_head(attended)
        predictions = predictions.contiguous().view(
            batch_size,
            num_junctions,
            self.prediction_horizon,
            self.target_size,
        )

        return predictions

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.GRU):

                for name, param in module.named_parameters():

                    if "weight" in name:
                        nn.init.xavier_uniform_(param)

                    elif "bias" in name:
                        nn.init.zeros_(param)
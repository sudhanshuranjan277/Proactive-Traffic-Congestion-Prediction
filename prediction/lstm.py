"""
LSTM Traffic Prediction Model

Predicts future queue-length sequence
from historical junction traffic features.
"""

import torch
import torch.nn as nn


class TrafficLSTM(nn.Module):

    def __init__(
        self,
        input_size,
        target_size,
        hidden_size=64,
        num_layers=2,
        prediction_horizon=10,
        dropout=0.2,
    ):

        super().__init__()

        self.prediction_horizon = prediction_horizon
        self.target_size = target_size

        lstm_dropout = (
            dropout if num_layers > 1 else 0.0
        )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.output_layer = nn.Linear(
            hidden_size,
            prediction_horizon * target_size,
        )

    def forward(self, x):
       

        lstm_output, _ = self.lstm(x)

        final_hidden_state = lstm_output[:, -1, :]

        predictions = self.output_layer(
            final_hidden_state
        )

        predictions = predictions.view(
            x.size(0),
            self.prediction_horizon,
            self.target_size,
        )

        return predictions


def create_lstm_model(input_size, target_size, **kwargs):
  

    return TrafficLSTM(
        input_size=input_size,
        target_size=target_size,
        **kwargs,
    )
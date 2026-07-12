"""
Traffic Queue Length Predictor

Loads trained LSTM model and scalers
to predict future queue-length sequences.
"""

import pickle

import numpy as np
import torch

from prediction.lstm import create_lstm_model


class TrafficPredictor:

    def __init__(
        self,
        model_path,
        scaler_path,
    ):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        )

        with open(
            scaler_path,
            "rb",
        ) as scaler_file:

            scalers = pickle.load(
                scaler_file
            )

        self.feature_scaler = scalers[
            "feature_scaler"
        ]

        self.target_scaler = scalers[
            "target_scaler"
        ]

        self.lookback = checkpoint[
            "lookback"
        ]

        self.feature_columns = checkpoint[
            "feature_columns"
        ]

        self.target_columns = checkpoint[
            "target_columns"
        ]

        self.model = create_lstm_model(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            num_layers=checkpoint["num_layers"],
            prediction_horizon=checkpoint[
                "prediction_horizon"
            ],
            target_size=checkpoint["target_size"],
            dropout=checkpoint["dropout"],
        ).to(self.device)

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

    def validate_sequence(
        self,
        sequence,
    ):
        """
        Validate predictor input shape.
        """

        sequence = np.asarray(
            sequence,
            dtype=np.float32,
        )

        expected_shape = (
            self.lookback,
            len(self.feature_columns),
        )

        if sequence.shape != expected_shape:

            raise ValueError(
                f"Invalid input sequence shape. "
                f"Expected {expected_shape}, "
                f"received {sequence.shape}"
            )

        return sequence

    def predict(
        self,
        sequence,
    ):
        """
        Predict future queue-length sequence.
        """

        sequence = self.validate_sequence(
            sequence
        )

        scaled_sequence = (
            self.feature_scaler.transform(
                sequence
            )
            .astype(np.float32)
        )

        input_tensor = torch.from_numpy(
            scaled_sequence
        ).unsqueeze(0)

        input_tensor = input_tensor.to(
            self.device
        )

        with torch.no_grad():

            scaled_prediction = self.model(
                input_tensor
            )

        scaled_prediction = (
            scaled_prediction
            .cpu()
            .numpy()[0]
        )

        prediction = (
            self.target_scaler.inverse_transform(
                scaled_prediction
            )
        )

        prediction = np.maximum(
            prediction,
            0.0,
        )

        return prediction

    def predict_queue_length(
        self,
        sequence,
    ):
        """
        Return flattened future queue prediction.
        """

        prediction = self.predict(
            sequence
        )

        return prediction[:, 0]
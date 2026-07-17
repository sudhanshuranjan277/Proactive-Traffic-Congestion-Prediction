"""
Traffic Queue Length Predictor

Loads the trained LSTM model and scalers to perform
real-time traffic queue length prediction.
"""

import joblib
import numpy as np
import torch

from prediction.lstm import TrafficLSTM


class TrafficPredictor:

    def __init__(
        self,
        model_path="models/lstm_model.pth",
        scaler_path="models/lstm_scalers.pkl",
        input_size=12,
        hidden_size=64,
        num_layers=2,
        prediction_horizon=10,
        target_size=1,
        dropout=0.2,
    ):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Using Device : {self.device}")

        # --------------------------------------------------
        # Load Scalers
        # --------------------------------------------------

        scalers = joblib.load(scaler_path)

        self.feature_scaler = scalers["feature_scaler"]
        self.target_scaler = scalers["target_scaler"]

        # --------------------------------------------------
        # Build Model
        # --------------------------------------------------

        self.model = TrafficLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            prediction_horizon=prediction_horizon,
            target_size=target_size,
            dropout=dropout,
        ).to(self.device)

        # --------------------------------------------------
        # Load Weights
        # --------------------------------------------------

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device,
            )
        )

        self.model.eval()

        self.lookback = prediction_horizon

        self.input_size = input_size

        print("LSTM Model Loaded Successfully.")

    # ----------------------------------------------------------
    # Validate Input
    # ----------------------------------------------------------

    def validate_input(self, sequence):

        sequence = np.asarray(sequence, dtype=np.float32)

        expected_shape = (
            self.lookback,
            self.input_size,
        )

        if sequence.shape != expected_shape:

            raise ValueError(

                f"Expected input shape {expected_shape}, "
                f"received {sequence.shape}"

            )

        return sequence

    # ----------------------------------------------------------
    # Scale Input
    # ----------------------------------------------------------

    def preprocess(self, sequence):

        sequence = self.validate_input(sequence)

        scaled = self.feature_scaler.transform(sequence)

        scaled = scaled.astype(np.float32)

        scaled = np.expand_dims(scaled, axis=0)

        return torch.tensor(
            scaled,
            dtype=torch.float32,
            device=self.device,
        )

    # ----------------------------------------------------------
    # Predict
    # ----------------------------------------------------------

    def predict(self, sequence):

        input_tensor = self.preprocess(sequence)

        with torch.no_grad():

            prediction = self.model(input_tensor)

        prediction = prediction.squeeze(0).cpu().numpy()

        prediction = self.target_scaler.inverse_transform(
            prediction
        )

        prediction = np.maximum(
            prediction,
            0
        )

        return prediction

    # ----------------------------------------------------------
    # Queue Length Prediction
    # ----------------------------------------------------------

    def predict_queue_length(self, sequence):

        prediction = self.predict(sequence)

        return prediction[:, 0]

    # ----------------------------------------------------------
    # Future Targets
    # ----------------------------------------------------------

    def predict_future_targets(self, sequence):

        return self.predict(sequence)


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    predictor = TrafficPredictor()

    dummy_sequence = np.random.rand(
        10,
        12,
    ).astype(np.float32)

    prediction = predictor.predict_queue_length(
        dummy_sequence
    )

    print("\nPredicted Queue Length")

    print(prediction)
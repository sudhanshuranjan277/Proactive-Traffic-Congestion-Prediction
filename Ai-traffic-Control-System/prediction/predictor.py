"""
Traffic Predictor

Loads the trained LSTM model checkpoint produced by
scripts/train_model.py and its scalers (produced by the same script)
to perform traffic prediction (queue_length, downstream_occupancy,
average_speed, waiting_time).

IMPORTANT: this loads the checkpoint format saved by
scripts/train_model.py — a dict containing 'model_state_dict' plus
architecture metadata (input_size, hidden_size, num_layers,
prediction_horizon, target_size, dropout, lookback, feature_columns,
target_columns) — NOT a raw state_dict. All architecture parameters
are read from the checkpoint itself rather than re-guessed/hardcoded,
so this predictor can never drift out of sync with however the model
was actually trained.
"""

import os
import sys
import pickle

import numpy as np
import torch

# Make the project root importable, whether this file is run directly
# (python prediction/predictor.py) or imported as a module
# (python -m prediction.predictor, or from another script) — matches
# the same bootstrap pattern used in scripts/train_model.py.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prediction.lstm import TrafficLSTM

from config import (
    MODEL_DIR,
    LSTM_MODEL_FILENAME,
    LSTM_SCALER_FILENAME,
)


DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, LSTM_MODEL_FILENAME)
DEFAULT_SCALER_PATH = os.path.join(MODEL_DIR, LSTM_SCALER_FILENAME)


class TrafficPredictor:

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        scaler_path=DEFAULT_SCALER_PATH,
        device=None,
    ):

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Using Device : {self.device}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. "
                f"Run scripts/train_model.py first."
            )
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler file not found at '{scaler_path}'. "
                f"Run scripts/train_model.py first."
            )

        # --------------------------------------------------
        # Load Scalers
        # (custom StandardScaler objects from
        # prediction/preprocessing.py, saved with pickle by
        # scripts/train_model.py — use pickle to load, not joblib,
        # to match exactly how they were written.)
        # --------------------------------------------------

        with open(scaler_path, "rb") as scaler_file:
            scalers = pickle.load(scaler_file)

        self.feature_scaler = scalers["feature_scaler"]
        self.target_scaler = scalers["target_scaler"]

        # --------------------------------------------------
        # Load Checkpoint
        # (dict format from scripts/train_model.py, not a raw
        # state_dict — see module docstring.)
        # --------------------------------------------------

        checkpoint = torch.load(model_path, map_location=self.device)

        required_keys = {
            "model_state_dict", "input_size", "hidden_size",
            "num_layers", "prediction_horizon", "target_size", "dropout",
        }
        missing_keys = required_keys - set(checkpoint.keys())
        if missing_keys:
            raise ValueError(
                f"Checkpoint at '{model_path}' is missing expected keys "
                f"{missing_keys}. It may have been saved by a different "
                f"script than scripts/train_model.py."
            )

        # Architecture is rebuilt from what was ACTUALLY trained/saved,
        # never hardcoded here, so this can't silently drift out of
        # sync with the model that produced the checkpoint.
        self.input_size = checkpoint["input_size"]
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.prediction_horizon = checkpoint["prediction_horizon"]
        self.target_size = checkpoint["target_size"]
        self.dropout = checkpoint["dropout"]

        # Lookback (input sequence length) — NOT the same thing as
        # prediction_horizon. Falls back to prediction_horizon only if
        # an older checkpoint didn't record it, with a loud warning
        # since that fallback is very likely wrong.
        if "lookback" in checkpoint:
            self.lookback = checkpoint["lookback"]
        else:
            print(
                "WARNING: checkpoint has no 'lookback' key — falling back "
                "to prediction_horizon, which is very likely INCORRECT. "
                "Re-train with the current scripts/train_model.py to fix this."
            )
            self.lookback = self.prediction_horizon

        self.feature_columns = checkpoint.get("feature_columns")
        self.target_columns = checkpoint.get("target_columns")

        self.model = TrafficLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            prediction_horizon=self.prediction_horizon,
            target_size=self.target_size,
            dropout=self.dropout,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        print("LSTM Model Loaded Successfully.")
        print(f"  Lookback (input timesteps)    : {self.lookback}")
        print(f"  Prediction Horizon            : {self.prediction_horizon}")
        print(f"  Input Features                : {self.input_size}")
        print(f"  Target Variables               : {self.target_size}"
              f"{' (' + ', '.join(self.target_columns) + ')' if self.target_columns else ''}")

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
        """
        Parameters
        ----------
        sequence : array-like, shape (lookback, input_size)
            Raw (unscaled) feature values, oldest row first.

        Returns
        -------
        numpy.ndarray, shape (prediction_horizon, target_size)
            Predicted target values in original (unscaled) units,
            clipped at 0 (queue length / waiting time can't be negative).
        """

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
        """
        Returns just the queue_length column of the prediction.
        Assumes 'queue_length' is target index 0, which matches
        prediction/preprocessing.py's TARGET_COLUMNS ordering.
        """

        prediction = self.predict(sequence)

        return prediction[:, 0]

    # ----------------------------------------------------------
    # Future Targets
    # ----------------------------------------------------------

    def predict_future_targets(self, sequence):

        return self.predict(sequence)


# ==========================================================
# Self-test
#
# Uses the checkpoint's own recorded shape (lookback, input_size)
# rather than hardcoded numbers, so this stays correct even if
# config.py's LOOKBACK/FEATURE_COLUMNS change later.
# ==========================================================

if __name__ == "__main__":

    predictor = TrafficPredictor()

    dummy_sequence = np.random.rand(
        predictor.lookback,
        predictor.input_size,
    ).astype(np.float32)

    prediction = predictor.predict_future_targets(
        dummy_sequence
    )

    print(f"\nPrediction shape : {prediction.shape}")
    print("Predicted Queue Length (first target column):")
    print(prediction[:, 0])
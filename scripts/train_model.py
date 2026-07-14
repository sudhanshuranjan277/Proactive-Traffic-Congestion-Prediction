"""
LSTM Traffic Model Training

Responsible for:
- Loading traffic dataset
- Creating chronological LSTM sequences
- Scaling features and targets
- Chronological train-validation split
- Training TrafficLSTM
- Saving best model and scalers
"""

import os
import sys
import pickle

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from prediction.preprocessing import (
    TrafficPreprocessor,
    StandardScaler,
)

from prediction.lstm import create_lstm_model

from config import (
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    DATASET_FILENAME,
    LOOKBACK,
    PREDICTION_HORIZON,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT,
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_LEARNING_RATE,
    TRAIN_RATIO,
    RANDOM_SEED,
)


DATASET_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    DATASET_FILENAME,
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_model.pth",
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "lstm_scalers.pkl",
)

HIDDEN_SIZE = LSTM_HIDDEN_SIZE
NUM_LAYERS = LSTM_NUM_LAYERS
DROPOUT = LSTM_DROPOUT

BATCH_SIZE = LSTM_BATCH_SIZE
EPOCHS = LSTM_EPOCHS
LEARNING_RATE = LSTM_LEARNING_RATE


def set_seed():
    """
    Set random seeds for reproducibility.
    """

    np.random.seed(RANDOM_SEED)

    torch.manual_seed(RANDOM_SEED)


def chronological_split(
    X,
    y,
    train_ratio,
):
    """
    Split time-series sequences chronologically.

    Random splitting is intentionally avoided.
    """

    split_index = int(
        len(X) * train_ratio
    )

    if (
        split_index <= 0
        or split_index >= len(X)
    ):
        raise ValueError(
            "Invalid chronological split."
        )

    X_train = X[:split_index]

    X_validation = X[split_index:]

    y_train = y[:split_index]

    y_validation = y[split_index:]

    return (
        X_train,
        X_validation,
        y_train,
        y_validation,
    )


def scale_sequences(
    X_train,
    X_validation,
    y_train,
    y_validation,
):
    """
    Fit scalers only on training data
    and transform train and validation sequences.
    """

    feature_scaler = StandardScaler()

    target_scaler = StandardScaler()

    feature_count = X_train.shape[-1]

    target_count = y_train.shape[-1]

    feature_scaler.fit(
        X_train.reshape(
            -1,
            feature_count,
        )
    )

    target_scaler.fit(
        y_train.reshape(
            -1,
            target_count,
        )
    )

    X_train = feature_scaler.transform(
        X_train
    ).astype(np.float32)

    X_validation = feature_scaler.transform(
        X_validation
    ).astype(np.float32)

    y_train = target_scaler.transform(
        y_train
    ).astype(np.float32)

    y_validation = target_scaler.transform(
        y_validation
    ).astype(np.float32)

    return (
        X_train,
        X_validation,
        y_train,
        y_validation,
        feature_scaler,
        target_scaler,
    )


def create_data_loader(
    X,
    y,
    shuffle,
):
    """
    Create PyTorch DataLoader.
    """

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


def evaluate_model(
    model,
    data_loader,
    loss_function,
    device,
):
    """
    Calculate average validation loss.
    """

    model.eval()

    total_loss = 0.0

    batch_count = 0

    with torch.no_grad():

        for features, targets in data_loader:

            features = features.to(device)

            targets = targets.to(device)

            predictions = model(features)

            loss = loss_function(
                predictions,
                targets,
            )

            total_loss += loss.item()

            batch_count += 1

    if batch_count == 0:

        raise ValueError(
            "Validation DataLoader is empty."
        )

    return total_loss / batch_count


def main():

    set_seed()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)

    print(
        "LSTM Traffic Prediction Training"
    )

    print("=" * 60)

    print(f"Device  : {device}")

    print(f"Dataset : {DATASET_PATH}")

    preprocessor = TrafficPreprocessor(
        lookback=LOOKBACK,
        prediction_horizon=PREDICTION_HORIZON,
    )

    X, y, metadata = preprocessor.process(
        DATASET_PATH
    )

    print(f"Sequences : {len(metadata)}")

    print(f"X Shape   : {X.shape}")

    print(f"Y Shape   : {y.shape}")

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = chronological_split(
        X,
        y,
        TRAIN_RATIO,
    )

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
        feature_scaler,
        target_scaler,
    ) = scale_sequences(
        X_train,
        X_validation,
        y_train,
        y_validation,
    )

    print(
        f"Train Sequences      : "
        f"{len(X_train)}"
    )

    print(
        f"Validation Sequences : "
        f"{len(X_validation)}"
    )

    train_loader = create_data_loader(
        X_train,
        y_train,
        shuffle=False,
    )

    validation_loader = create_data_loader(
        X_validation,
        y_validation,
        shuffle=False,
    )

    model = create_lstm_model(
        input_size=X.shape[-1],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        prediction_horizon=PREDICTION_HORIZON,
        target_size=y.shape[-1],
        dropout=DROPOUT,
    ).to(device)

    loss_function = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_validation_loss = float("inf")

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    print("\nTraining Started...\n")

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        total_train_loss = 0.0

        batch_count = 0

        for features, targets in train_loader:

            features = features.to(device)

            targets = targets.to(device)

            optimizer.zero_grad()

            predictions = model(features)

            loss = loss_function(
                predictions,
                targets,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_train_loss += loss.item()

            batch_count += 1

        if batch_count == 0:

            raise ValueError(
                "Training DataLoader is empty."
            )

        average_train_loss = (
            total_train_loss / batch_count
        )

        validation_loss = evaluate_model(
            model,
            validation_loader,
            loss_function,
            device,
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: "
            f"{average_train_loss:.6f} | "
            f"Validation Loss: "
            f"{validation_loss:.6f}"
        )

        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            checkpoint = {
                "model_state_dict": (
                    model.state_dict()
                ),
                "input_size": X.shape[-1],
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "prediction_horizon": (
                    PREDICTION_HORIZON
                ),
                "target_size": y.shape[-1],
                "dropout": DROPOUT,
                "lookback": LOOKBACK,
                "feature_columns": list(
                    preprocessor.FEATURE_COLUMNS
                ),
                "target_columns": list(
                    preprocessor.TARGET_COLUMNS
                ),
                "best_validation_loss": (
                    best_validation_loss
                ),
            }

            torch.save(
                checkpoint,
                MODEL_PATH,
            )

    scalers = {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
    }

    with open(
        SCALER_PATH,
        "wb",
    ) as scaler_file:

        pickle.dump(
            scalers,
            scaler_file,
        )

    print("\nTraining Completed.")

    print(
        f"Best Validation Loss : "
        f"{best_validation_loss:.6f}"
    )

    print(
        f"Model File  : {MODEL_PATH}"
    )

    print(
        f"Scaler File : {SCALER_PATH}"
    )


if __name__ == "__main__":
    main()
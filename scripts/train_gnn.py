"""
Graph Transformer Training Pipeline

Trains prediction/gnn_model.py's TrafficGraphTransformer on ALL
junctions jointly, using prediction/graph_preprocessing.py to build
aligned graph snapshots.

Deliberately reuses diagnostics, chronological splitting, scaling,
metrics, and the persistence baseline from scripts/train_model.py
(the LSTM pipeline) rather than reimplementing them — this is what
makes the two models genuinely comparable in
evaluation/compare_models.py: same data, same metrics, same
baseline, only the architecture differs.

Run with:
    python scripts/train_gnn.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    DATASET_FILENAME,
    LOOKBACK,
    PREDICTION_HORIZON,
    GNN_HIDDEN_SIZE,
    GNN_NUM_LAYERS,
    GNN_NUM_HEADS,
    GNN_NUM_ATTENTION_LAYERS,
    GNN_DROPOUT,
    GNN_BATCH_SIZE,
    GNN_EPOCHS,
    GNN_LEARNING_RATE,
    GNN_WEIGHT_DECAY,
    GNN_MODEL_FILENAME,
    GNN_SCALER_FILENAME,
    TRAIN_RATIO,
    RANDOM_SEED,
    METRICS_DIR,
    LOGS_DIR,
)

from prediction.graph_preprocessing import GraphTrafficPreprocessor
from prediction.gnn_model import TrafficGraphTransformer

# Reused, not reimplemented — see module docstring.
from scripts.train_model import (
    diagnose_targets,
    chronological_split,
    scale_sequences,
    compute_metrics,
    print_metrics_comparison,
    evaluate_model,
)

import pickle


DATASET_PATH = os.path.join(PROCESSED_DATA_DIR, DATASET_FILENAME)
MODEL_PATH = os.path.join(MODEL_DIR, GNN_MODEL_FILENAME)
SCALER_PATH = os.path.join(MODEL_DIR, GNN_SCALER_FILENAME)
METRICS_PATH = os.path.join(METRICS_DIR, "gnn_metrics.csv")
TRAINING_LOG_PATH = os.path.join(LOGS_DIR, "gnn_training.log")


def set_seed():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


def compute_graph_persistence_baseline(X_raw, y_raw, feature_columns, target_columns):
    """
    Same idea as compute_persistence_baseline in train_model.py
    ("each target stays at its last observed value"), adapted for
    the extra junction axis:
    X_raw: (samples, num_junctions, lookback, n_features)
    y_raw: (samples, num_junctions, horizon, n_targets)
    """

    feature_indices = [feature_columns.index(t) for t in target_columns]

    last_observed = X_raw[:, :, -1, feature_indices]  # (samples, junctions, n_targets)

    horizon = y_raw.shape[2]
    baseline_pred = np.repeat(
        last_observed[:, :, np.newaxis, :], horizon, axis=2
    )

    return baseline_pred  # (samples, junctions, horizon, n_targets)


def create_data_loader(X, y, shuffle):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=GNN_BATCH_SIZE, shuffle=shuffle)


def main():

    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device : {device}")

    print("=" * 60)
    print("Graph Transformer Training (multi-junction joint prediction)")
    print("=" * 60)
    print(f"Dataset : {DATASET_PATH}")

    preprocessor = GraphTrafficPreprocessor(
        lookback=LOOKBACK,
        prediction_horizon=PREDICTION_HORIZON,
    )

    raw_dataset = preprocessor.load_dataset(DATASET_PATH)
    raw_dataset = preprocessor.prepare_dataset(raw_dataset)

    diagnose_targets(
        raw_dataset,
        preprocessor.FEATURE_COLUMNS,
        preprocessor.TARGET_COLUMNS,
    )

    X, y, metadata, junction_ids = preprocessor.create_sequences(raw_dataset)

    print(f"\nJunctions   : {junction_ids}")
    print(f"Sequences   : {len(metadata)}")
    print(f"X Shape     : {X.shape}  (samples, junctions, lookback, features)")
    print(f"Y Shape     : {y.shape}  (samples, junctions, horizon, targets)")

    X_train, X_validation, y_train, y_validation = chronological_split(
        X, y, TRAIN_RATIO
    )

    X_validation_raw = X_validation.copy()
    y_validation_raw = y_validation.copy()

    print(f"Train Sequences      : {len(X_train)}")
    print(f"Validation Sequences : {len(X_validation)}")

    (
        X_train, X_validation, y_train, y_validation,
        feature_scaler, target_scaler,
    ) = scale_sequences(X_train, X_validation, y_train, y_validation)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(SCALER_PATH, "wb") as scaler_file:
        pickle.dump(
            {"feature_scaler": feature_scaler, "target_scaler": target_scaler},
            scaler_file,
        )

    train_loader = create_data_loader(X_train, y_train, shuffle=True)
    validation_loader = create_data_loader(X_validation, y_validation, shuffle=False)

    num_junctions = X.shape[1]

    model = TrafficGraphTransformer(
        input_size=X.shape[-1],
        target_size=y.shape[-1],
        hidden_size=GNN_HIDDEN_SIZE,
        num_layers=GNN_NUM_LAYERS,
        prediction_horizon=PREDICTION_HORIZON,
        num_heads=GNN_NUM_HEADS,
        num_attention_layers=GNN_NUM_ATTENTION_LAYERS,
        dropout=GNN_DROPOUT,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=GNN_LEARNING_RATE, weight_decay=GNN_WEIGHT_DECAY
    )

    best_validation_loss = float("inf")

    print("\nTraining Started...\n")

    for epoch in range(GNN_EPOCHS):

        model.train()
        running_loss = 0.0
        batch_count = 0

        for features, targets in train_loader:

            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        train_loss = running_loss / max(batch_count, 1)
        validation_loss = evaluate_model(model, validation_loader, criterion, device)

        print(
            f"Epoch {epoch+1:03d}/{GNN_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Validation Loss: {validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "input_size": X.shape[-1],
                "hidden_size": GNN_HIDDEN_SIZE,
                "num_layers": GNN_NUM_LAYERS,
                "num_heads": GNN_NUM_HEADS,
                "num_attention_layers": GNN_NUM_ATTENTION_LAYERS,
                "prediction_horizon": PREDICTION_HORIZON,
                "target_size": y.shape[-1],
                "dropout": GNN_DROPOUT,
                "lookback": LOOKBACK,
                "num_junctions": num_junctions,
                "junction_ids": junction_ids,
                "feature_columns": list(preprocessor.FEATURE_COLUMNS),
                "target_columns": list(preprocessor.TARGET_COLUMNS),
                "best_validation_loss": best_validation_loss,
            }

            torch.save(checkpoint, MODEL_PATH)

    print("\nTraining Completed.")
    print(f"Best Validation Loss : {best_validation_loss:.6f}")
    print(f"Model File  : {MODEL_PATH}")
    print(f"Scaler File : {SCALER_PATH}")

    # ------------------------------------------------------------
    # FINAL EVALUATION — best checkpoint, real units, vs. baseline
    # ------------------------------------------------------------

    print("\nEvaluating best checkpoint on validation set...")

    best_checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        X_val_tensor = torch.from_numpy(X_validation).to(device)
        predictions_scaled = model(X_val_tensor).cpu().numpy()

    # Flatten (samples, junctions, horizon, targets) -> (samples*junctions, horizon, targets)
    # for inverse_transform and metric reuse (compute_metrics expects a 3D array).
    samples, junctions, horizon, n_targets = predictions_scaled.shape

    predictions_flat = target_scaler.inverse_transform(
        predictions_scaled.reshape(-1, n_targets)
    ).reshape(samples, junctions, horizon, n_targets)

    predictions_flat = predictions_flat.reshape(samples * junctions, horizon, n_targets)
    actual_flat = y_validation_raw.reshape(samples * junctions, horizon, n_targets)

    model_metrics = compute_metrics(
        actual_flat, predictions_flat, preprocessor.TARGET_COLUMNS
    )

    baseline_predictions = compute_graph_persistence_baseline(
        X_validation_raw, y_validation_raw,
        preprocessor.FEATURE_COLUMNS, preprocessor.TARGET_COLUMNS,
    ).reshape(samples * junctions, horizon, n_targets)

    baseline_metrics = compute_metrics(
        actual_flat, baseline_predictions, preprocessor.TARGET_COLUMNS
    )

    print_metrics_comparison(
        model_metrics, baseline_metrics, preprocessor.TARGET_COLUMNS
    )

    os.makedirs(METRICS_DIR, exist_ok=True)
    import pandas as pd
    rows = []
    for target in preprocessor.TARGET_COLUMNS:
        rows.append({"target": target, "model": "graph_transformer", **model_metrics[target]})
        rows.append({"target": target, "model": "persistence_baseline", **baseline_metrics[target]})
    pd.DataFrame(rows).to_csv(METRICS_PATH, index=False)
    print(f"\nMetrics CSV : {METRICS_PATH}")


if __name__ == "__main__":
    main()

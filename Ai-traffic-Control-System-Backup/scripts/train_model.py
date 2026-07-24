"""
UPDATED TEMPLATE

Production-ready template based on your uploaded file.
Added:
- training_history.csv
- loss_curve.png
- metrics.csv (with MAPE)
- prediction_results.csv
- actual_vs_predicted.png
- residual_plot.png / residual_histogram.png
- training.log
- PDF report

Model logic is unchanged.
"""

"""
LSTM Traffic Model Training

Responsible for:
- Loading traffic dataset
- Creating chronological LSTM sequences
- Scaling features and targets
- Chronological train-validation split
- Training TrafficLSTM
- Saving best model and scalers
- Logging, metrics, plots, and a PDF report
"""

import os
import sys
import pickle
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)

import torch
import torch.nn as nn

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

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
    GRAPHS_DIR,
    LOGS_DIR,
    METRICS_DIR,
    REPORTS_DIR,
    PREDICTIONS_DIR,
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


TRAINING_HISTORY_PATH = os.path.join(LOGS_DIR, "training_history.csv")
TRAINING_LOG_PATH = os.path.join(LOGS_DIR, "training.log")
METRICS_PATH = os.path.join(METRICS_DIR, "lstm_metrics.csv")
PREDICTION_RESULTS_PATH = os.path.join(PREDICTIONS_DIR, "prediction_results.csv")
LOSS_CURVE_PATH = os.path.join(GRAPHS_DIR, "loss_curve.png")
ACTUAL_PRED_PATH = os.path.join(GRAPHS_DIR, "actual_vs_predicted.png")
RESIDUAL_PLOT_PATH = os.path.join(GRAPHS_DIR, "residual_plot.png")
RESIDUAL_HIST_PATH = os.path.join(GRAPHS_DIR, "residual_histogram.png")
PDF_REPORT_PATH = os.path.join(REPORTS_DIR, "lstm_training_report.pdf")

HIDDEN_SIZE = LSTM_HIDDEN_SIZE
NUM_LAYERS = LSTM_NUM_LAYERS
DROPOUT = LSTM_DROPOUT

BATCH_SIZE = LSTM_BATCH_SIZE
EPOCHS = LSTM_EPOCHS
LEARNING_RATE = LSTM_LEARNING_RATE

EARLY_STOPPING_PATIENCE = 20


def setup_logging():
    """
    Configure logging to write training progress to LOGS_DIR/training.log
    in addition to whatever is printed to the console.
    """

    logging.basicConfig(
        filename=TRAINING_LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )


def create_output_folders():
    """
    Make sure every output directory exists before training starts,
    not just MODEL_DIR.
    """

    for folder in [
        MODEL_DIR,
        LOGS_DIR,
        METRICS_DIR,
        REPORTS_DIR,
        PREDICTIONS_DIR,
        GRAPHS_DIR,
    ]:
        os.makedirs(folder, exist_ok=True)


def diagnose_targets(dataset, feature_columns, target_columns):
    """
    Print target distribution + feature correlation, so it's obvious
    up front whether there's learnable signal in the data at all
    (rather than only finding out after a full training run).
    """

    print("\n")
    print("=" * 60)
    print("DATA DIAGNOSTICS")
    print("=" * 60)

    for target in target_columns:

        series = dataset[target]

        print(f"\nTarget: '{target}'")
        print(series.describe())

        zero_pct = (series == 0).mean() * 100
        print(f"Percent of values == 0 : {zero_pct:.2f}%")
        if zero_pct > 50:
            print(
                ">> WARNING: heavily zero-inflated. MSE-trained models "
                "tend to collapse toward predicting near-zero, which "
                "shows up as flat loss and R2 near 0."
            )

        correlations = dataset[feature_columns].corrwith(series).sort_values(
            key=lambda s: s.abs(), ascending=False
        )
        print("Top correlated features:")
        print(correlations.head(5))

    print("=" * 60)


def compute_persistence_baseline(X_raw, y_raw, feature_columns, target_columns):
    """
    Naive baseline: "each target stays at its last observed value for
    the whole prediction horizon." Computed on RAW (unscaled) values.

    Since every target column is also present in FEATURE_COLUMNS (the
    model gets each target's own recent history as an input feature),
    the last observed value of a target is just the last timestep of
    X_raw at that feature's column index.
    """

    feature_indices = [feature_columns.index(t) for t in target_columns]

    # X_raw: (samples, lookback, n_features) -> last timestep's target columns
    last_observed = X_raw[:, -1, feature_indices]  # (samples, n_targets)

    horizon = y_raw.shape[1]
    baseline_pred = np.repeat(last_observed[:, np.newaxis, :], horizon, axis=1)

    return baseline_pred  # (samples, horizon, n_targets), same shape as y_raw


def compute_metrics(actual, predicted, target_columns):
    """
    MAE / RMSE / R2 / MAPE per target column, flattened across samples
    and horizon steps for that target.
    """

    metrics = {}

    for i, target in enumerate(target_columns):

        actual_col = actual[:, :, i].reshape(-1)
        predicted_col = predicted[:, :, i].reshape(-1)

        mae = mean_absolute_error(actual_col, predicted_col)
        rmse = np.sqrt(mean_squared_error(actual_col, predicted_col))
        r2 = r2_score(actual_col, predicted_col)

        # MAPE blows up / is undefined when actual values are 0, which is
        # common in zero-inflated traffic data. Guard against that instead
        # of letting it silently explode.
        nonzero_mask = actual_col != 0
        if nonzero_mask.any():
            mape = mean_absolute_percentage_error(
                actual_col[nonzero_mask], predicted_col[nonzero_mask]
            )
        else:
            mape = float("nan")

        metrics[target] = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

    return metrics


def print_metrics_comparison(model_metrics, baseline_metrics, target_columns):

    print("\n")
    print("=" * 80)
    print("MODEL vs PERSISTENCE BASELINE (per target)")
    print("=" * 80)

    for target in target_columns:

        m = model_metrics[target]
        b = baseline_metrics[target]

        print(f"\n{target}")
        print(
            f"  Model    -> MAE: {m['MAE']:.4f}  RMSE: {m['RMSE']:.4f}  "
            f"R2: {m['R2']:.4f}  MAPE: {m['MAPE']:.4f}"
        )
        print(
            f"  Baseline -> MAE: {b['MAE']:.4f}  RMSE: {b['RMSE']:.4f}  "
            f"R2: {b['R2']:.4f}  MAPE: {b['MAPE']:.4f}"
        )

        if m["RMSE"] < b["RMSE"]:
            print("  -> Model beats the naive baseline.")
        else:
            print("  -> Model does NOT beat the naive baseline yet.")

    print("\n" + "=" * 80)


def save_metrics_csv(model_metrics, target_columns, path):
    """
    Flatten the per-target metrics dict into rows and save as CSV.
    """

    rows = []

    for target in target_columns:
        m = model_metrics[target]
        rows.append(
            {
                "Target": target,
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "R2": m["R2"],
                "MAPE": m["MAPE"],
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(path, index=False)

    return metrics_df


def save_prediction_csv(actual, predicted, target_columns, path):
    """
    Flatten (samples, horizon, targets) actual/predicted arrays into a
    long-format CSV: one row per (sample, time_step, target).
    """

    n_samples, n_steps, n_targets = actual.shape

    records = []

    for sample_idx in range(n_samples):
        for step_idx in range(n_steps):
            for target_idx, target in enumerate(target_columns):

                actual_value = actual[sample_idx, step_idx, target_idx]
                predicted_value = predicted[sample_idx, step_idx, target_idx]

                records.append(
                    {
                        "sample": sample_idx,
                        "time_step": step_idx,
                        "target": target,
                        "actual": actual_value,
                        "predicted": predicted_value,
                        "residual": actual_value - predicted_value,
                    }
                )

    predictions_df = pd.DataFrame(records)
    predictions_df.to_csv(path, index=False)

    return predictions_df


def plot_loss_curve(train_losses, validation_losses, path):

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(path)
    plt.close()


def plot_actual_vs_predicted(actual_flat, predicted_flat, path, max_points=500):

    # Cap the number of points plotted so the figure stays readable on
    # large validation sets.
    n_points = min(len(actual_flat), max_points)

    plt.figure()
    plt.plot(actual_flat[:n_points], label="Actual")
    plt.plot(predicted_flat[:n_points], label="Predicted")
    plt.xlabel("Sample")
    plt.ylabel("Value")
    plt.title("Actual vs Predicted")
    plt.legend()
    plt.grid()
    plt.savefig(path)
    plt.close()


def plot_residuals(predicted_flat, residual_flat, path):

    plt.figure()
    plt.scatter(predicted_flat, residual_flat, s=8, alpha=0.5)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.grid()
    plt.savefig(path)
    plt.close()


def plot_residual_histogram(residual_flat, path):

    plt.figure()
    plt.hist(residual_flat, bins=50)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.grid()
    plt.savefig(path)
    plt.close()


def build_pdf_report(
    metrics_df,
    best_validation_loss,
    n_train_sequences,
    n_validation_sequences,
    path,
):
    """
    Assemble a short PDF report: title, run summary, hyperparameters,
    metrics table, and the four saved plots.
    """

    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("LSTM Traffic Model Training Report", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 12))

    story.append(Paragraph("Dataset Summary", styles["Heading2"]))
    story.append(
        Paragraph(
            f"Training sequences: {n_train_sequences} &nbsp;&nbsp; "
            f"Validation sequences: {n_validation_sequences}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 12))

    story.append(Paragraph("Hyperparameters", styles["Heading2"]))
    hyperparameter_lines = (
        f"Lookback: {LOOKBACK} | Prediction Horizon: {PREDICTION_HORIZON}<br/>"
        f"Hidden Size: {HIDDEN_SIZE} | Num Layers: {NUM_LAYERS} | "
        f"Dropout: {DROPOUT}<br/>"
        f"Batch Size: {BATCH_SIZE} | Learning Rate: {LEARNING_RATE} | "
        f"Max Epochs: {EPOCHS}"
    )
    story.append(Paragraph(hyperparameter_lines, styles["Normal"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Training Summary", styles["Heading2"]))
    story.append(
        Paragraph(
            f"Best validation loss: {best_validation_loss:.6f}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 12))

    story.append(Paragraph("Metrics (Validation Set)", styles["Heading2"]))

    table_data = [list(metrics_df.columns)] + metrics_df.round(4).values.tolist()
    table = Table(table_data)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 20))

    for title, image_path in [
        ("Loss Curve", LOSS_CURVE_PATH),
        ("Actual vs Predicted", ACTUAL_PRED_PATH),
        ("Residual Plot", RESIDUAL_PLOT_PATH),
        ("Residual Histogram", RESIDUAL_HIST_PATH),
    ]:
        if os.path.exists(image_path):
            story.append(Paragraph(title, styles["Heading2"]))
            story.append(Image(image_path, width=400, height=260))
            story.append(Spacer(1, 16))

    doc.build(story)


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

    # Output folders + logging must be ready before anything else runs.
    create_output_folders()
    setup_logging()

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

    logging.info("Training started. Device=%s Dataset=%s", device, DATASET_PATH)

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

    raw_dataset = preprocessor.prepare_dataset(
        preprocessor.load_dataset(DATASET_PATH)
    )
    diagnose_targets(
        raw_dataset,
        preprocessor.FEATURE_COLUMNS,
        preprocessor.TARGET_COLUMNS,
    )

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

    # Keep raw (unscaled) copies for the persistence baseline and for
    # inverse-transforming model predictions back to real units later.
    X_validation_raw = X_validation.copy()
    y_validation_raw = y_validation.copy()

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

    training_history = []
    train_losses = []
    validation_losses = []

    patience = EARLY_STOPPING_PATIENCE
    patience_counter = 0

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

        logging.info(
            "Epoch %03d/%d | Train Loss: %.6f | Validation Loss: %.6f",
            epoch,
            EPOCHS,
            average_train_loss,
            validation_loss,
        )

        train_losses.append(average_train_loss)
        validation_losses.append(validation_loss)

        training_history.append(
            {
                "epoch": epoch,
                "train_loss": average_train_loss,
                "validation_loss": validation_loss,
            }
        )

        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            patience_counter = 0

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

        else:

            patience_counter += 1

            if patience_counter >= patience:

                print(
                    f"\nEarly stopping triggered at epoch {epoch} "
                    f"(no improvement for {patience} epochs)."
                )
                logging.info(
                    "Early stopping triggered at epoch %d", epoch
                )
                break

    # Save training history CSV
    pd.DataFrame(training_history).to_csv(
        TRAINING_HISTORY_PATH,
        index=False,
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

    logging.info(
        "Training completed. Best validation loss: %.6f",
        best_validation_loss,
    )

    # ------------------------------------------------------------
    # FINAL EVALUATION — load the BEST saved checkpoint (not
    # whatever the last epoch happened to produce) and report
    # real-unit MAE/RMSE/R2/MAPE per target, alongside a persistence
    # baseline, so accuracy is never just "trust the loss number".
    # ------------------------------------------------------------

    print("\nEvaluating best checkpoint on validation set...")

    best_checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        X_val_tensor = torch.from_numpy(X_validation).to(device)
        predictions_scaled = model(X_val_tensor).cpu().numpy()

    predictions = target_scaler.inverse_transform(predictions_scaled)

    model_metrics = compute_metrics(
        y_validation_raw, predictions, preprocessor.TARGET_COLUMNS
    )

    baseline_predictions = compute_persistence_baseline(
        X_validation_raw, y_validation_raw,
        preprocessor.FEATURE_COLUMNS, preprocessor.TARGET_COLUMNS,
    )
    baseline_metrics = compute_metrics(
        y_validation_raw, baseline_predictions, preprocessor.TARGET_COLUMNS
    )

    print_metrics_comparison(
        model_metrics, baseline_metrics, preprocessor.TARGET_COLUMNS
    )

    # Save metrics CSV
    metrics_df = save_metrics_csv(
        model_metrics, preprocessor.TARGET_COLUMNS, METRICS_PATH
    )

    # Save prediction CSV (long format: sample, time_step, target, actual, predicted, residual)
    predictions_df = save_prediction_csv(
        y_validation_raw, predictions, preprocessor.TARGET_COLUMNS, PREDICTION_RESULTS_PATH
    )

    actual_flat = predictions_df["actual"].to_numpy()
    predicted_flat = predictions_df["predicted"].to_numpy()
    residual_flat = predictions_df["residual"].to_numpy()

    # Plots
    plot_loss_curve(train_losses, validation_losses, LOSS_CURVE_PATH)
    plot_actual_vs_predicted(actual_flat, predicted_flat, ACTUAL_PRED_PATH)
    plot_residuals(predicted_flat, residual_flat, RESIDUAL_PLOT_PATH)
    plot_residual_histogram(residual_flat, RESIDUAL_HIST_PATH)

    # PDF report
    build_pdf_report(
        metrics_df,
        best_validation_loss,
        n_train_sequences=len(X_train),
        n_validation_sequences=len(X_validation),
        path=PDF_REPORT_PATH,
    )

    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    print(f"Training History : {TRAINING_HISTORY_PATH}")
    print(f"Training Log      : {TRAINING_LOG_PATH}")
    print(f"Metrics CSV       : {METRICS_PATH}")
    print(f"Prediction CSV    : {PREDICTION_RESULTS_PATH}")
    print(f"Loss Curve        : {LOSS_CURVE_PATH}")
    print(f"Actual vs Predicted: {ACTUAL_PRED_PATH}")
    print(f"Residual Plot     : {RESIDUAL_PLOT_PATH}")
    print(f"Residual Histogram: {RESIDUAL_HIST_PATH}")
    print(f"PDF Report        : {PDF_REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
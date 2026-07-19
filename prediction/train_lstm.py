# ============================================================
# IMPORTS
# ============================================================

import os
import sys
import random
import joblib
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# IMPORT EXISTING MODEL
# ============================================================

from lstm import TrafficLSTM

# ============================================================
# FEATURE LIST
# (defined before Config so INPUT_SIZE can be derived from it
#  instead of being hardcoded / able to drift out of sync)
# ============================================================

TARGET_COLUMN = "queue_length"

FEATURE_COLUMNS = [
    "traffic_flow",
    "arrival_rate",
    "departure_rate",
    "traffic_event_type",
    "remaining_green_time",
    "current_signal_phase",
    "downstream_occupancy",
    "downstream_queue_length",
    "vehicle_count",
    "average_speed",
    "waiting_time",
    "travel_time",
    # The target's own recent history is one of the strongest predictors
    # of a queue-length-style series. Only *past* values are ever placed
    # in the input window (see create_sequences), so this is a legitimate
    # autoregressive feature, not leakage of the future.
    TARGET_COLUMN,
]


# ============================================================
# CONFIGURATION
# Every setting can be overridden via an environment variable,
# so nothing here is truly hardcoded — these are just defaults.
# e.g.  DATA_PATH=data/location_2.csv EPOCHS=50 python train_lstm.py
# ============================================================

def _env_str(name, default):
    return os.getenv(name, default)


def _env_int(name, default):
    return int(os.getenv(name, default))


def _env_float(name, default):
    return float(os.getenv(name, default))


def _env_bool(name, default):
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Config:

    # Dataset
    DATA_PATH = _env_str("DATA_PATH", "data/processed/location_1_dataset.csv")

    # Save Paths
    MODEL_SAVE_PATH = _env_str("MODEL_SAVE_PATH", "models/lstm_model.pth")
    SCALER_SAVE_PATH = _env_str("SCALER_SAVE_PATH", "models/lstm_scalers.pkl")

    TRAIN_HISTORY = _env_str("TRAIN_HISTORY_PATH", "outputs/training_history.csv")
    METRICS_PATH = _env_str("METRICS_PATH", "outputs/metrics.csv")
    PREDICTIONS_PATH = _env_str("PREDICTIONS_PATH", "outputs/predictions.csv")

    LOSS_GRAPH = _env_str("LOSS_GRAPH_PATH", "outputs/loss_curve.png")
    PREDICTION_GRAPH = _env_str("PREDICTION_GRAPH_PATH", "outputs/prediction_vs_actual.png")
    METRICS_BAR_GRAPH = _env_str("METRICS_BAR_GRAPH_PATH", "outputs/metrics_bar_chart.png")

    # Sequence Parameters
    SEQUENCE_LENGTH = _env_int("SEQUENCE_LENGTH", 10)
    PREDICTION_HORIZON = _env_int("PREDICTION_HORIZON", 10)

    # Model Parameters
    # INPUT_SIZE is derived from FEATURE_COLUMNS, not hardcoded,
    # so it can never get out of sync with the actual feature list.
    INPUT_SIZE = len(FEATURE_COLUMNS)
    HIDDEN_SIZE = _env_int("HIDDEN_SIZE", 64)
    NUM_LAYERS = _env_int("NUM_LAYERS", 2)
    DROPOUT = _env_float("DROPOUT", 0.20)

    # Training
    BATCH_SIZE = _env_int("BATCH_SIZE", 32)
    LEARNING_RATE = _env_float("LEARNING_RATE", 0.001)
    EPOCHS = _env_int("EPOCHS", 100)
    PATIENCE = _env_int("PATIENCE", 10)
    ENABLE_EARLY_STOPPING = _env_bool("ENABLE_EARLY_STOPPING", True)
    SEED = _env_int("SEED", 42)
    TEST_SIZE = _env_float("TEST_SIZE", 0.20)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# FIX RANDOM SEED
# ============================================================

def set_seed(seed=Config.SEED):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed()


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

def create_directories():

    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(Config.TRAIN_HISTORY) or ".", exist_ok=True)


create_directories()


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(path):

    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found : {path}")

    df = pd.read_csv(path)

    print(f"Dataset Shape : {df.shape}")
    print(df.head())

    return df


# ============================================================
# DATA CLEANING
# ============================================================

def clean_dataset(df):

    print("\nCleaning Dataset...")

    df = df.copy()

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Replace Infinite
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill Missing Values
    # NOTE: fillna(method="ffill"/"bfill") is removed in pandas >= 2.2
    # and raises a TypeError. Use .ffill()/.bfill() instead.
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)

    print("Missing Values Removed.")
    print(f"New Shape : {df.shape}")

    return df


# ============================================================
# TIME-AWARE TRAIN / TEST SPLIT (on the raw dataframe)
#
# Splitting BEFORE scaling and BEFORE sequence creation avoids two
# leakage sources: (1) the scaler seeing test-set values, and
# (2) a sequence window spanning across the train/test boundary.
# ============================================================

def split_dataframe(df):

    print("\nSplitting Dataset (time-ordered, no shuffle)...")

    n = len(df)
    test_len = int(n * Config.TEST_SIZE)

    train_df = df.iloc[: n - test_len].reset_index(drop=True)
    test_df = df.iloc[n - test_len:].reset_index(drop=True)

    print(f"Training Rows : {len(train_df)}")
    print(f"Testing Rows  : {len(test_df)}")

    return train_df, test_df


# ============================================================
# FIT SCALERS (on the training split ONLY, to avoid leakage)
# ============================================================

def fit_scalers(train_df):

    print("\nFitting Scalers on training data only...")

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    feature_scaler.fit(train_df[FEATURE_COLUMNS])
    target_scaler.fit(train_df[[TARGET_COLUMN]])

    scaler_dict = {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
    }

    joblib.dump(scaler_dict, Config.SCALER_SAVE_PATH)

    print("Scaler Saved.")

    return feature_scaler, target_scaler


# ============================================================
# TRANSFORM A SPLIT USING ALREADY-FITTED SCALERS
# ============================================================

def transform_split(df, feature_scaler, target_scaler):

    X_scaled = feature_scaler.transform(df[FEATURE_COLUMNS])
    y_scaled = target_scaler.transform(df[[TARGET_COLUMN]])

    return X_scaled, y_scaled


# ============================================================
# SEQUENCE GENERATION
# ============================================================

def create_sequences(features, target):

    print("\nCreating Sequences...")

    X = []
    y = []

    seq_len = Config.SEQUENCE_LENGTH
    horizon = Config.PREDICTION_HORIZON

    total = len(features)

    for i in range(total - seq_len - horizon):
        X.append(features[i:i + seq_len])
        y.append(target[i + seq_len:i + seq_len + horizon])

    X = np.array(X)
    y = np.array(y)

    print(f"Input Shape : {X.shape}")
    print(f"Target Shape : {y.shape}")

    return X, y


# ============================================================
# CUSTOM DATASET
# ============================================================

class TrafficDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# ============================================================
# BUILD DATALOADERS
# ============================================================

def create_dataloaders(X_train, X_test, y_train, y_test):

    train_dataset = TrafficDataset(X_train, y_train)
    test_dataset = TrafficDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    print("\nDataLoaders Created")
    print(f"Train Batches : {len(train_loader)}")
    print(f"Test Batches  : {len(test_loader)}")

    return train_loader, test_loader


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():

    print("\nBuilding LSTM Model...")

    model = TrafficLSTM(
        input_size=Config.INPUT_SIZE,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        prediction_horizon=Config.PREDICTION_HORIZON,
        target_size=1,
        dropout=Config.DROPOUT,
    )

    model = model.to(Config.DEVICE)

    print(model)

    return model


# ============================================================
# LOSS FUNCTION
# ============================================================

def build_loss():
    criterion = nn.MSELoss()
    print("\nLoss Function : MSELoss")
    return criterion


# ============================================================
# OPTIMIZER
# ============================================================

def build_optimizer(model):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=1e-5,
    )

    print("Optimizer : Adam")

    return optimizer


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

def build_scheduler(optimizer):

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    print("Scheduler : ReduceLROnPlateau")

    return scheduler


# ============================================================
# PERSISTENCE BASELINE
#
# Naive forecast: "the queue length stays at its last observed
# value for the whole horizon." A model that can't beat this is
# not learning anything useful. Computed on raw (unscaled) values
# so it's directly comparable to the model's inverse-transformed
# metrics.
# ============================================================

def compute_baseline_metrics(test_df):

    seq_len = Config.SEQUENCE_LENGTH
    horizon = Config.PREDICTION_HORIZON

    values = test_df[TARGET_COLUMN].values
    total = len(values)

    baseline_preds = []
    actual = []

    for i in range(total - seq_len - horizon):
        last_seen_value = values[i + seq_len - 1]
        future_values = values[i + seq_len: i + seq_len + horizon]

        baseline_preds.extend([last_seen_value] * horizon)
        actual.extend(future_values)

    baseline_preds = np.array(baseline_preds)
    actual = np.array(actual)

    mae = mean_absolute_error(actual, baseline_preds)
    rmse = np.sqrt(mean_squared_error(actual, baseline_preds))
    r2 = r2_score(actual, baseline_preds)

    metrics = {
        "Baseline MAE": mae,
        "Baseline RMSE": rmse,
        "Baseline R2 Score": r2,
    }

    print("\n")
    print("=" * 60)
    print("PERSISTENCE BASELINE (naive 'last value repeated')")
    print("=" * 60)
    print(f"Baseline MAE      : {mae:.6f}")
    print(f"Baseline RMSE     : {rmse:.6f}")
    print(f"Baseline R2 Score : {r2:.6f}")

    return metrics


# ============================================================
# DIAGNOSTICS — run before training to sanity-check the data
#
# If validation loss is flat from epoch 1, it usually means either
# (a) the target is degenerate/skewed (e.g. mostly zeros), so the
#     model just learns to predict the average, or
# (b) the features have little/no correlation with the target, so
#     there's nothing learnable for the LSTM to pick up on.
# This prints both, so it's obvious which one you're dealing with.
# ============================================================

def diagnose_data(df):

    print("\n")
    print("=" * 60)
    print("DATA DIAGNOSTICS")
    print("=" * 60)

    target = df[TARGET_COLUMN]

    print(f"\nTarget column: '{TARGET_COLUMN}'")
    print(target.describe())

    zero_pct = (target == 0).mean() * 100
    print(f"\nPercent of target values == 0 : {zero_pct:.2f}%")

    if zero_pct > 50:
        print(">> WARNING: target is heavily zero-inflated. "
              "MSE-trained models tend to collapse toward predicting "
              "near-zero for everything, which explains flat loss and "
              "R2 near 0. Consider a different loss (e.g. weighted MSE) "
              "or reframing as classification+regression (occurs vs. size).")

    print("\nCorrelation of each feature with the target:")
    correlations = df[FEATURE_COLUMNS].corrwith(target).sort_values(
        key=lambda s: s.abs(), ascending=False
    )
    print(correlations)

    weak = correlations[correlations.abs() < 0.05]
    if len(weak) == len(correlations):
        print("\n>> WARNING: every feature has near-zero linear correlation "
              "with the target. The LSTM may have very little learnable "
              "signal to work with — this often means the true predictive "
              "features aren't in the dataset, or the relationship is too "
              "nonlinear/noisy for the model to find quickly.")

    print("=" * 60)


# ============================================================
# INITIALIZE COMPLETE TRAINING COMPONENTS
# ============================================================

def initialize_training():

    print("\nInitializing Training Pipeline...")

    df = load_dataset(Config.DATA_PATH)
    df = clean_dataset(df)

    diagnose_data(df)

    train_df, test_df = split_dataframe(df)

    feature_scaler, target_scaler = fit_scalers(train_df)

    X_train_scaled, y_train_scaled = transform_split(train_df, feature_scaler, target_scaler)
    X_test_scaled, y_test_scaled = transform_split(test_df, feature_scaler, target_scaler)

    print("\nCreating training sequences...")
    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled)

    print("\nCreating testing sequences...")
    X_test, y_test = create_sequences(X_test_scaled, y_test_scaled)

    train_loader, test_loader = create_dataloaders(
        X_train, X_test, y_train, y_test
    )

    model = build_model()
    criterion = build_loss()
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    baseline_metrics = compute_baseline_metrics(test_df)

    print("\nTraining Pipeline Ready.")

    return model, train_loader, test_loader, criterion, optimizer, scheduler, baseline_metrics


# ============================================================
# VALIDATE MODEL
# ============================================================

def validate_model(model, test_loader, criterion):
    """
    Validate trained LSTM model.

    Returns
    -------
    validation_loss, metrics, predictions, actual_values
    """

    model.eval()

    running_loss = 0.0
    predictions = []
    actual_values = []

    scaler_dict = joblib.load(Config.SCALER_SAVE_PATH)
    target_scaler = scaler_dict["target_scaler"]

    with torch.no_grad():

        for inputs, targets in test_loader:

            inputs = inputs.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

            outputs = outputs.cpu().numpy().reshape(-1, 1)
            targets = targets.cpu().numpy().reshape(-1, 1)

            outputs = target_scaler.inverse_transform(outputs)
            targets = target_scaler.inverse_transform(targets)

            predictions.extend(outputs.flatten())
            actual_values.extend(targets.flatten())

    validation_loss = running_loss / max(len(test_loader), 1)

    predictions = np.array(predictions)
    actual_values = np.array(actual_values)

    mae = mean_absolute_error(actual_values, predictions)
    rmse = np.sqrt(mean_squared_error(actual_values, predictions))
    r2 = r2_score(actual_values, predictions)

    metrics = {
        "Validation Loss": validation_loss,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2,
    }

    print("\n")
    print("=" * 60)
    print("VALIDATION RESULT")
    print("=" * 60)
    print(f"Validation Loss : {validation_loss:.6f}")
    print(f"MAE             : {mae:.6f}")
    print(f"RMSE            : {rmse:.6f}")
    print(f"R2 Score        : {r2:.6f}")

    return validation_loss, metrics, predictions, actual_values


# ============================================================
# COMPLETE TRAINING LOOP
# ============================================================

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler):

    print("\n")
    print("=" * 80)
    print("STARTING MODEL TRAINING")
    print("=" * 80)

    history = {
        "epoch": [],
        "train_loss": [],
        "validation_loss": [],
        "learning_rate": [],
    }

    best_validation_loss = float("inf")
    early_stop_counter = 0

    # Ensure these exist even if training exits after epoch 0 for some reason
    predictions, actual_values, metrics = None, None, None

    for epoch in range(Config.EPOCHS):

        ####################################################
        # TRAINING
        ####################################################

        model.train()

        running_train_loss = 0.0
        total_batches = len(train_loader)

        for inputs, targets in train_loader:

            inputs = inputs.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()

            # Prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_train_loss += loss.item()

        average_train_loss = running_train_loss / max(total_batches, 1)

        ####################################################
        # VALIDATION
        ####################################################

        validation_loss, metrics, predictions, actual_values = validate_model(
            model, test_loader, criterion
        )

        ####################################################
        # LR Scheduler
        ####################################################

        scheduler.step(validation_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        ####################################################
        # Save History
        ####################################################

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(average_train_loss)
        history["validation_loss"].append(validation_loss)
        history["learning_rate"].append(current_lr)

        ####################################################
        # Epoch Summary
        ####################################################

        print("\n")
        print("-" * 80)
        print(f"Epoch : {epoch+1}/{Config.EPOCHS}")
        print(f"Training Loss : {average_train_loss:.6f}")
        print(f"Validation Loss : {validation_loss:.6f}")
        print(f"MAE : {metrics['MAE']:.6f}")
        print(f"RMSE : {metrics['RMSE']:.6f}")
        print(f"R2 Score : {metrics['R2 Score']:.6f}")
        print(f"Learning Rate : {current_lr:.8f}")

        ####################################################
        # BEST MODEL
        ####################################################

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss
            early_stop_counter = 0

            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

            print("\nBest model updated.")

        else:

            early_stop_counter += 1
            print(f"\nNo improvement ({early_stop_counter}/{Config.PATIENCE})")

        ####################################################
        # EARLY STOPPING
        ####################################################

        if Config.ENABLE_EARLY_STOPPING and early_stop_counter >= Config.PATIENCE:
            print("\n")
            print("=" * 80)
            print("EARLY STOPPING TRIGGERED")
            print("=" * 80)
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(Config.TRAIN_HISTORY, index=False)

    print("\nTraining history saved.")

    return history_df, predictions, actual_values, metrics


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(metrics):

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(Config.METRICS_PATH, index=False)

    print("Metrics saved successfully.")


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(predictions, actual_values):

    prediction_df = pd.DataFrame({
        "Actual": actual_values,
        "Predicted": predictions,
    })

    prediction_df.to_csv(Config.PREDICTIONS_PATH, index=False)

    print("Predictions saved successfully.")


# ============================================================
# PLOT TRAINING LOSS
# ============================================================

def plot_loss(history):

    plt.figure(figsize=(10, 5))

    plt.plot(history["epoch"], history["train_loss"], marker="o", label="Train Loss")
    plt.plot(history["epoch"], history["validation_loss"], marker="o", label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(Config.LOSS_GRAPH, dpi=300)
    plt.close()

    print("Loss graph saved.")


# ============================================================
# PLOT PREDICTIONS
# ============================================================

def plot_predictions(actual, predicted):

    plt.figure(figsize=(12, 6))

    plt.plot(actual, label="Actual")
    plt.plot(predicted, label="Predicted")

    plt.xlabel("Samples")
    plt.ylabel("Queue Length")
    plt.title("Actual vs Predicted Queue Length")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(Config.PREDICTION_GRAPH, dpi=300)
    plt.close()

    print("Prediction graph saved.")


# ============================================================
# METRICS BAR CHART (Model vs Baseline)
# ============================================================

def plot_metrics_bar(metrics):
    """
    Bar chart comparing the model's MAE / RMSE / R2 against the
    persistence baseline's MAE / RMSE / R2, side by side.
    """

    error_labels = ["MAE", "RMSE"]
    model_errors = [metrics["MAE"], metrics["RMSE"]]
    baseline_errors = [metrics["Baseline MAE"], metrics["Baseline RMSE"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left panel: error metrics (lower is better) ---
    x = np.arange(len(error_labels))
    width = 0.35

    axes[0].bar(x - width / 2, model_errors, width, label="Model")
    axes[0].bar(x + width / 2, baseline_errors, width, label="Baseline")

    for i, (m, b) in enumerate(zip(model_errors, baseline_errors)):
        axes[0].text(i - width / 2, m, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
        axes[0].text(i + width / 2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=9)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(error_labels)
    axes[0].set_ylabel("Error (lower is better)")
    axes[0].set_title("Model vs Baseline — Error Metrics")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    # --- Right panel: R2 score (higher is better, can be negative) ---
    r2_labels = ["Model R2", "Baseline R2"]
    r2_values = [metrics["R2 Score"], metrics["Baseline R2 Score"]]
    colors = ["#4C72B0", "#DD8452"]

    bars = axes[1].bar(r2_labels, r2_values, color=colors)

    for bar, val in zip(bars, r2_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.3f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9,
        )

    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("R2 Score (higher is better)")
    axes[1].set_title("Model vs Baseline — R2 Score")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(Config.METRICS_BAR_GRAPH, dpi=300)
    plt.close()

    print("Metrics bar chart saved.")


# ============================================================
# COMPLETE TRAINING PIPELINE
# ============================================================

def run_complete_training():

    print("\n")
    print("=" * 80)
    print("INITIALIZING TRAINING PIPELINE")
    print("=" * 80)
    print(f"Working directory : {os.getcwd()}")
    print(f"Dataset path       : {os.path.abspath(Config.DATA_PATH)}")
    print(f"Device             : {Config.DEVICE}")
    sys.stdout.flush()

    (
        model, train_loader, test_loader,
        criterion, optimizer, scheduler,
        baseline_metrics,
    ) = initialize_training()

    history, predictions, actual_values, metrics = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler
    )

    metrics.update(baseline_metrics)

    save_metrics(metrics)
    save_predictions(predictions, actual_values)

    plot_loss(history)
    plot_predictions(actual_values, predictions)
    plot_metrics_bar(metrics)

    print("\n")
    print("=" * 80)
    print("MODEL vs BASELINE")
    print("=" * 80)
    print(f"Model RMSE    : {metrics['RMSE']:.6f}   |   Baseline RMSE    : {metrics['Baseline RMSE']:.6f}")
    print(f"Model MAE     : {metrics['MAE']:.6f}   |   Baseline MAE     : {metrics['Baseline MAE']:.6f}")
    print(f"Model R2      : {metrics['R2 Score']:.6f}   |   Baseline R2      : {metrics['Baseline R2 Score']:.6f}")
    if metrics["RMSE"] < metrics["Baseline RMSE"]:
        print("\n-> Model is beating the naive baseline.")
    else:
        print("\n-> Model is NOT beating the naive baseline yet. Consider more features, "
              "more data, or a longer/shorter prediction horizon.")

    print("\n")
    print("=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Model Saved      : {Config.MODEL_SAVE_PATH}")
    print(f"Scaler Saved     : {Config.SCALER_SAVE_PATH}")
    print(f"Metrics Saved    : {Config.METRICS_PATH}")
    print(f"Training History : {Config.TRAIN_HISTORY}")
    print(f"Loss Graph       : {Config.LOSS_GRAPH}")
    print(f"Prediction Graph : {Config.PREDICTION_GRAPH}")
    print(f"Metrics Bar Chart: {Config.METRICS_BAR_GRAPH}")
    print("=" * 80)
# MAIN
if __name__ == "__main__":
    try:
        run_complete_training()
    except Exception:
        import traceback
        print("\n")
        print("=" * 80)
        print("TRAINING FAILED — full error below")
        print("=" * 80)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)
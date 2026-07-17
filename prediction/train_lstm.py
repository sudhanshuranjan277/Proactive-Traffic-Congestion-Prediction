# ============================================================
# IMPORTS
# ============================================================

import os
import random
import joblib
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
# CONFIGURATION
# ============================================================

class Config:

    # Dataset
    DATA_PATH = "data/processed/location_1_dataset.csv"

    # Save Paths
    MODEL_SAVE_PATH = "models/lstm_model.pth"
    SCALER_SAVE_PATH = "models/lstm_scalers.pkl"

    TRAIN_HISTORY = "outputs/training_history.csv"
    METRICS_PATH = "outputs/metrics.csv"

    LOSS_GRAPH = "outputs/loss_curve.png"
    PREDICTION_GRAPH = "outputs/prediction_vs_actual.png"

    # Sequence Parameters
    SEQUENCE_LENGTH = 10
    PREDICTION_HORIZON = 10

    # Model Parameters
    INPUT_SIZE = 12
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    DROPOUT = 0.20

    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 100
    PATIENCE = 10

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# ============================================================
# FIX RANDOM SEED
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed()


# ============================================================
# FEATURE LIST
# ============================================================

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

    "travel_time"

]

TARGET_COLUMN = "queue_length"


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

def create_directories():

    os.makedirs("models", exist_ok=True)

    os.makedirs("outputs", exist_ok=True)


create_directories()


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(path):

    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Dataset not found : {path}"
        )

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

    df.fillna(method="ffill", inplace=True)

    df.fillna(method="bfill", inplace=True)

    df.fillna(0, inplace=True)

    print("Missing Values Removed.")

    print(f"New Shape : {df.shape}")

    return df


# ============================================================
# FEATURE PREPROCESSING
# ============================================================

def preprocess_features(df):

    print("\nPreparing Features...")

    X = df[FEATURE_COLUMNS]

    y = df[TARGET_COLUMN]

    feature_scaler = MinMaxScaler()

    target_scaler = MinMaxScaler()

    X_scaled = feature_scaler.fit_transform(X)

    y_scaled = target_scaler.fit_transform(
        y.values.reshape(-1, 1)
    )

    scaler_dict = {

        "feature_scaler": feature_scaler,

        "target_scaler": target_scaler

    }

    joblib.dump(
        scaler_dict,
        Config.SCALER_SAVE_PATH
    )

    print("Scaler Saved.")

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

        X.append(

            features[
                i:i + seq_len
            ]

        )

        y.append(

            target[
                i + seq_len:
                i + seq_len + horizon
            ]

        )

    X = np.array(X)

    y = np.array(y)

    print(f"Input Shape : {X.shape}")

    print(f"Target Shape : {y.shape}")

    return X, y

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

def split_dataset(X, y):

    print("\nSplitting Dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        shuffle=False,
        random_state=42
    )

    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


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

def create_dataloaders(
    X_train,
    X_test,
    y_train,
    y_test
):

    train_dataset = TrafficDataset(
        X_train,
        y_train
    )

    test_dataset = TrafficDataset(
        X_test,
        y_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    print("\nDataLoaders Created")

    print(f"Train Batches : {len(train_loader)}")

    print(f"Test Batches  : {len(test_loader)}")

    return (
        train_loader,
        test_loader
    )


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

        dropout=Config.DROPOUT

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

        weight_decay=1e-5

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

        min_lr=1e-6

    )

    print("Scheduler : ReduceLROnPlateau")

    return scheduler


# ============================================================
# INITIALIZE COMPLETE TRAINING COMPONENTS
# ============================================================

def initialize_training():

    print("\nInitializing Training Pipeline...")

    df = load_dataset(Config.DATA_PATH)

    df = clean_dataset(df)

    X_scaled, y_scaled = preprocess_features(df)

    X, y = create_sequences(
        X_scaled,
        y_scaled
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_dataset(X, y)

    (
        train_loader,
        test_loader
    ) = create_dataloaders(
        X_train,
        X_test,
        y_train,
        y_test
    )

    model = build_model()

    criterion = build_loss()

    optimizer = build_optimizer(model)

    scheduler = build_scheduler(optimizer)

    print("\nTraining Pipeline Ready.")

    return (

        model,

        train_loader,

        test_loader,

        criterion,

        optimizer,

        scheduler

    )
    # ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    epoch
):
    """
    Train model for one epoch.

    Returns
    -------
    float
        Average epoch loss.
    """

    model.train()

    running_loss = 0.0

    total_batches = len(train_loader)

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        inputs = inputs.to(Config.DEVICE)

        targets = targets.to(Config.DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        loss.backward()

        # Prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:

            print(
                f"Epoch [{epoch+1}/{Config.EPOCHS}] "
                f"Batch [{batch_idx+1}/{total_batches}] "
                f"Loss : {loss.item():.6f}"
            )

    epoch_loss = running_loss / max(total_batches, 1)

    return epoch_loss


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler
):

    history = {

        "epoch": [],

        "train_loss": [],

        "val_loss": []

    }

    best_loss = float("inf")

    patience_counter = 0

    print("\n")
    print("=" * 70)
    print("TRAINING STARTED")
    print("=" * 70)

    for epoch in range(Config.EPOCHS):

        train_loss = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            epoch

        )

        validation_loss, _, _, _ = validate_model(
            model,
            test_loader,
            criterion
        )
        val_loss = validation_loss

        history["epoch"].append(epoch + 1)

        history["train_loss"].append(train_loss)

        history["val_loss"].append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print("-" * 70)

        print(f"Epoch        : {epoch+1}")

        print(f"Train Loss   : {train_loss:.6f}")

        print(f"Valid Loss   : {val_loss:.6f}")

        print(f"LearningRate : {current_lr:.8f}")

        print("-" * 70)

        if val_loss < best_loss:

            best_loss = val_loss

            patience_counter = 0

            
            print("Best Model Saved.")

        else:

            patience_counter += 1

            print(
                f"No Improvement "
                f"({patience_counter}/{Config.PATIENCE})"
            )

        if patience_counter >= Config.PATIENCE:

            print("\nEarly stopping activated.")

            break

    history = pd.DataFrame(history)

    history.to_csv(
        Config.TRAIN_HISTORY,
        index=False
    )

    print("\nTraining History Saved.")

    return history
  
  # ============================================================
# VALIDATE MODEL
# ============================================================

def validate_model(
    model,
    test_loader,
    criterion
):
    """
    Validate trained LSTM model.

    Returns
    -------
    validation_loss
    metrics
    predictions
    actual_values
    """

    model.eval()

    running_loss = 0.0

    predictions = []

    actual_values = []

    scaler_dict = joblib.load(
        Config.SCALER_SAVE_PATH
    )

    target_scaler = scaler_dict["target_scaler"]

    with torch.no_grad():

        for inputs, targets in test_loader:

            inputs = inputs.to(Config.DEVICE)

            targets = targets.to(Config.DEVICE)

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            running_loss += loss.item()

            outputs = outputs.cpu().numpy()

            targets = targets.cpu().numpy()

            outputs = outputs.reshape(-1, 1)

            targets = targets.reshape(-1, 1)

            outputs = target_scaler.inverse_transform(outputs)

            targets = target_scaler.inverse_transform(targets)

            predictions.extend(outputs.flatten())

            actual_values.extend(targets.flatten())

    validation_loss = running_loss / max(len(test_loader), 1)

    predictions = np.array(predictions)

    actual_values = np.array(actual_values)

    mae = mean_absolute_error(
        actual_values,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual_values,
            predictions
        )
    )

    r2 = r2_score(
        actual_values,
        predictions
    )

    metrics = {

        "Validation Loss": validation_loss,

        "MAE": mae,

        "RMSE": rmse,

        "R2 Score": r2

    }

    print("\n")
    print("=" * 60)
    print("VALIDATION RESULT")
    print("=" * 60)

    print(f"Validation Loss : {validation_loss:.6f}")

    print(f"MAE             : {mae:.6f}")

    print(f"RMSE            : {rmse:.6f}")

    print(f"R2 Score        : {r2:.6f}")

    return (

        validation_loss,

        metrics,

        predictions,

        actual_values

    )


# ============================================================
# VALIDATION CALLBACK
# ============================================================

def validation_callback(
    model,
    criterion
):
    """
    Callback used inside training loop.
    """

    (
        validation_loss,
        _,
        _,
        _
    ) = validate_model(

        model,

        test_loader,

        criterion

    )

    return validation_loss

# ============================================================
# COMPLETE TRAINING LOOP
# ============================================================

def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler
):

    print("\n")
    print("=" * 80)
    print("STARTING MODEL TRAINING")
    print("=" * 80)

    history = {
        "epoch": [],
        "train_loss": [],
        "validation_loss": [],
        "learning_rate": []
    }

    best_validation_loss = float("inf")

    early_stop_counter = 0

    for epoch in range(Config.EPOCHS):

        ####################################################
        # TRAINING
        ####################################################

        model.train()

        running_train_loss = 0.0

        total_batches = len(train_loader)

        for batch_index, (inputs, targets) in enumerate(train_loader):

            inputs = inputs.to(Config.DEVICE)

            targets = targets.to(Config.DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()

            # Prevent exploding gradients

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            running_train_loss += loss.item()

        average_train_loss = (
            running_train_loss / max(total_batches, 1)
        )

        ####################################################
        # VALIDATION
        ####################################################

        (
            validation_loss,
            metrics,
            predictions,
            actual_values
        ) = validate_model(

            model,

            test_loader,

            criterion

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

        history["train_loss"].append(
            average_train_loss
        )

        history["validation_loss"].append(
            validation_loss
        )

        history["learning_rate"].append(
            current_lr
        )

        ####################################################
        # Epoch Summary
        ####################################################

        print("\n")

        print("-" * 80)

        print(
            f"Epoch : {epoch+1}/{Config.EPOCHS}"
        )

        print(
            f"Training Loss : {average_train_loss:.6f}"
        )

        print(
            f"Validation Loss : {validation_loss:.6f}"
        )

        print(
            f"MAE : {metrics['MAE']:.6f}"
        )

        print(
            f"RMSE : {metrics['RMSE']:.6f}"
        )

        print(
            f"R² Score : {metrics['R2 Score']:.6f}"
        )

        print(
            f"Learning Rate : {current_lr:.8f}"
        )

        ####################################################
        # BEST MODEL
        ####################################################

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            early_stop_counter = 0

            torch.save(

                model.state_dict(),

                Config.MODEL_SAVE_PATH

            )

            print("\n✅ Best model updated.")

        else:

            early_stop_counter += 1

            print(
                f"\nNo improvement ({early_stop_counter}/{Config.PATIENCE})"
            )

        ####################################################
        # EARLY STOPPING
        ####################################################

        if early_stop_counter >= Config.PATIENCE:

            print("\n")
            print("=" * 80)
            print("EARLY STOPPING TRIGGERED")
            print("=" * 80)

            break
    # SAVE HISTORY

    history = pd.DataFrame(history)

    history.to_csv(

        Config.TRAIN_HISTORY,

        index=False

    )

    print("\nTraining history saved.")

    return (

        history,

        predictions,

        actual_values,

        metrics

    )
    # ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(metrics):

    metrics_df = pd.DataFrame([metrics])

    metrics_df.to_csv(
        Config.METRICS_PATH,
        index=False
    )

    print("Metrics Saved Successfully.")


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(predictions, actual_values):

    prediction_df = pd.DataFrame({

        "Actual Queue Length": actual_values,

        "Predicted Queue Length": predictions

    })

    prediction_df.to_csv(

        "outputs/predictions.csv",

        index=False

    )

    print("Prediction CSV Saved.")


# ============================================================
# LOSS CURVE
# ============================================================

def plot_loss(history):

    plt.figure(figsize=(10,5))

    plt.plot(

        history["epoch"],

        history["train_loss"],

        label="Training Loss"

    )

    plt.plot(

        history["epoch"],

        history["validation_loss"],

        label="Validation Loss"

    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training vs Validation Loss")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        Config.LOSS_GRAPH,

        dpi=300

    )

    plt.close()

    print("Loss Curve Saved.")


# ============================================================
# PREDICTION GRAPH
# ============================================================

def plot_predictions(

    actual,

    predicted

):

    plt.figure(figsize=(12,6))

    plt.plot(

        actual,

        label="Actual"

    )

    plt.plot(

        predicted,

        label="Predicted"

    )

    plt.xlabel("Samples")

    plt.ylabel("Queue Length")

    plt.title("Actual vs Predicted Queue Length")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        Config.PREDICTION_GRAPH,

        dpi=300

    )

    plt.close()

    print("Prediction Graph Saved.")


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def run_complete_training():

    (
        model,

        train_loader,

        test_loader,

        criterion,

        optimizer,

        scheduler

    ) = initialize_training()

    (

        history,

        predictions,

        actual_values,

        metrics

    ) = train_model(

        model,

        train_loader,

        test_loader,

        criterion,

        optimizer,

        scheduler

    )

    save_metrics(metrics)

    save_predictions(

        predictions,

        actual_values

    )

    plot_loss(history)

    plot_predictions(

        actual_values,

        predictions

    )

    print("\n")

    print("="*80)

    print("LSTM TRAINING COMPLETED SUCCESSFULLY")

    print("="*80)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_complete_training()
    
    
    # ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(metrics):

    metrics_df = pd.DataFrame([metrics])

    metrics_df.to_csv(

        Config.METRICS_PATH,

        index=False

    )

    print("Metrics saved successfully.")


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(

    predictions,

    actual_values

):

    prediction_df = pd.DataFrame({

        "Actual": actual_values,

        "Predicted": predictions

    })

    prediction_path = os.path.join(

        "outputs",

        "predictions.csv"

    )

    prediction_df.to_csv(

        prediction_path,

        index=False

    )

    print("Predictions saved successfully.")


# ============================================================
# PLOT TRAINING LOSS
# ============================================================

def plot_loss(history):

    plt.figure(figsize=(10,5))

    plt.plot(

        history["epoch"],

        history["train_loss"],

        marker="o",

        label="Train Loss"

    )

    plt.plot(

        history["epoch"],

        history["validation_loss"],

        marker="o",

        label="Validation Loss"

    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Training vs Validation Loss")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        Config.LOSS_GRAPH,

        dpi=300

    )

    plt.close()

    print("Loss graph saved.")


# ============================================================
# PLOT PREDICTIONS
# ============================================================

def plot_predictions(

    actual,

    predicted

):

    plt.figure(figsize=(12,6))

    plt.plot(

        actual,

        label="Actual"

    )

    plt.plot(

        predicted,

        label="Predicted"

    )

    plt.xlabel("Samples")

    plt.ylabel("Queue Length")

    plt.title("Actual vs Predicted Queue Length")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        Config.PREDICTION_GRAPH,

        dpi=300

    )

    plt.close()

    print("Prediction graph saved.")


# ============================================================
# COMPLETE TRAINING PIPELINE
# ============================================================

def run_complete_training():

    print("\n")

    print("=" * 80)

    print("INITIALIZING TRAINING PIPELINE")

    print("=" * 80)

    (
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler
    ) = initialize_training()

    (
        history,
        predictions,
        actual_values,
        metrics
    ) = train_model(

        model,

        train_loader,

        test_loader,

        criterion,

        optimizer,

        scheduler

    )

    save_metrics(metrics)

    save_predictions(

        predictions,

        actual_values

    )

    plot_loss(history)

    plot_predictions(

        actual_values,

        predictions

    )

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

    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_complete_training()


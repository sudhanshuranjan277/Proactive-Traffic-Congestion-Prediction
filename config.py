"""
Project Configuration

Proactive Traffic Congestion Prediction
and Adaptive Traffic Signal Control System
"""

import os


# ======================================
# Project Information
# ======================================

PROJECT_NAME = (
    "Proactive Traffic Congestion Prediction and Adaptive Traffic Signal Control using Graph Deep Learning"
)

AUTHOR = "Sudhanshu Ranjan"

VERSION = "2.0.0"


# ======================================
# Project Directories
# ======================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
)

RAW_DATA_DIR = os.path.join(
    DATA_DIR,
    "raw",
)

PROCESSED_DATA_DIR = os.path.join(
    DATA_DIR,
    "processed",
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
)

# ======================================
# Output Directories
# ======================================

GRAPHS_DIR = os.path.join(
    OUTPUT_DIR,
    "graphs",
)

LOGS_DIR = os.path.join(
    OUTPUT_DIR,
    "logs",
)

METRICS_DIR = os.path.join(
    OUTPUT_DIR,
    "metrics",
)

REPORTS_DIR = os.path.join(
    OUTPUT_DIR,
    "reports",
)

PREDICTIONS_DIR = os.path.join(
    OUTPUT_DIR,
    "predictions",
)

CHECKPOINTS_DIR = os.path.join(
    OUTPUT_DIR,
    "checkpoints",
)

# ======================================
# Evaluation
# ======================================

EVALUATION_RESULTS_DIR = os.path.join(
    OUTPUT_DIR,
    "results",
)

MAP_DIR = os.path.join(
    PROJECT_ROOT,
    "maps",
)

EVALUATION_DIR = os.path.join(
    PROJECT_ROOT,
    "evaluation",
)


# ======================================
# SUMO Configuration
# ======================================

SUMO_BINARY = "sumo"

SUMO_CONFIG = os.path.join(
    PROJECT_ROOT,
    "maps",
    "location_2",
    "osm.sumocfg",
)


# ======================================
# Simulation Settings
# ======================================

SIMULATION_STEP = 1

SIMULATION_TIME = 3600

OBSERVATION_WINDOW = 15


# ======================================
# Traffic Prediction
# ======================================

LOOKBACK = 30

PREDICTION_HORIZON = 10

FEATURE_COLUMNS = [
    "traffic_flow",
    "traffic_event_type",
    "remaining_green_time",
    "downstream_occupancy",
    "queue_length",
    "average_speed",
    "waiting_time",
    "current_signal_phase",
    "downstream_queue_length",
    "travel_time",
    "arrival_rate",
    "departure_rate",
]

TARGET_COLUMNS = [
    "queue_length",
    "downstream_occupancy",
    "average_speed",
    "waiting_time",
]


# ======================================
# Model Selection
# ======================================

BASELINE_MODEL = "LSTM"

GRAPH_MODEL = "GraphTransformer"

AVAILABLE_MODELS = [
    "Persistence",
    "LSTM",
    "GraphTransformer",
]

GRAPH_SEQUENCE_LENGTH = LOOKBACK

GRAPH_NUM_TARGETS = len(TARGET_COLUMNS)

GRAPH_NUM_FEATURES = len(FEATURE_COLUMNS)

USE_GRAPH_TRANSFORMER = True


# ======================================
# RL Input Features
# ======================================

INPUT_FEATURES = [
    "traffic_flow",
    "traffic_event_type",
    "remaining_green_time",
    "downstream_occupancy",
    "queue_length",
    "average_speed",
    "waiting_time",
    "current_signal_phase",
    "downstream_queue_length",
    "travel_time",
    "arrival_rate",
    "departure_rate",
]

CURRENT_STATE_FEATURES = [
    "traffic_flow",
    "remaining_green_time",
    "downstream_occupancy",
    "queue_length",
    "average_speed",
    "waiting_time",
    "downstream_queue_length",
    "arrival_rate",
    "departure_rate",
]

FUTURE_FEATURE_STATS = {
    "queue_length": ["mean", "max", "delta"],
    "downstream_occupancy": ["mean", "max"],
    "average_speed": ["mean"],
    "waiting_time": ["mean", "max"],
}


# ======================================
# LSTM Training
# ======================================

TRAIN_RATIO = 0.80

LSTM_HIDDEN_SIZE = 64

LSTM_NUM_LAYERS = 2

LSTM_DROPOUT = 0.20

LSTM_BATCH_SIZE = 32

LSTM_EPOCHS = 50

LSTM_LEARNING_RATE = 0.001

LSTM_WEIGHT_DECAY = 1e-5

EARLY_STOPPING_PATIENCE = 10

MIN_DELTA = 1e-4

SAVE_BEST_MODEL_ONLY = True


# ======================================
# Graph Transformer Training
# (multi-junction joint prediction — see prediction/gnn_model.py)
# ======================================

GNN_HIDDEN_SIZE = 64

GNN_NUM_LAYERS = 2

GNN_NUM_HEADS = 4

GNN_NUM_ATTENTION_LAYERS = 2

GNN_DROPOUT = 0.20

GNN_BATCH_SIZE = 16

GNN_EPOCHS = 50

GNN_LEARNING_RATE = 0.001

GNN_WEIGHT_DECAY = 1e-5

GNN_MODEL_FILENAME = "gnn_model.pth"

GNN_SCALER_FILENAME = "gnn_scalers.pkl"


# ======================================
# Reinforcement Learning
# ======================================

RL_LEARNING_RATE = 0.0001

RL_GAMMA = 0.99

RL_BATCH_SIZE = 32

RL_TARGET_UPDATE_FREQUENCY = 10

RL_MEMORY_CAPACITY = 10000

RL_NUM_ACTIONS = 3

RL_EXTENSION_SECONDS = 5

RL_EPSILON_START = 1.0

RL_EPSILON_END = 0.05

RL_MAX_EPISODES = 5

RL_MAX_STEPS_PER_EPISODE = 30


# ======================================
# Reward Configuration
# ======================================

REWARD_QUEUE_WEIGHT = 1.00

REWARD_WAITING_TIME_WEIGHT = 0.50

REWARD_OCCUPANCY_WEIGHT = 2.00

REWARD_DOWNSTREAM_QUEUE_WEIGHT = 0.75

REWARD_SPEED_WEIGHT = 0.10

REWARD_SLOW_TRAFFIC_PENALTY = 0.25

REWARD_CONGESTION_PENALTY = 1.00

REWARD_EXTEND_PENALTY = 0.02

REWARD_NEXT_PHASE_PENALTY = 0.05

REWARD_INVALID_ACTION_PENALTY = 0.50

# ======================================
# DDQN Network Architecture
# ======================================

DDQN_HIDDEN_DIM = 128

DDQN_GRADIENT_CLIP_NORM = 1.0

DDQN_TARGET_SYNC_INTERVAL = 10

DDQN_SAVE_INTERVAL = 1


# ======================================
# Device Configuration
# ======================================

FORCE_CPU = False

import torch

DEVICE = (
    "cpu"
    if FORCE_CPU
    else "cuda" if torch.cuda.is_available() else "cpu"
)


# ======================================
# Output Formatting
# ======================================

PRINT_WIDTH = 60

LOG_LEVEL = "INFO"

ENABLE_PROGRESS_BAR = True


# ======================================
# Gradient and Epsilon Bounds
# ======================================

EPSILON_MIN = 0.0

EPSILON_MAX = 1.0


# ======================================
# Traffic Collector Thresholds
# ======================================

COLLECTOR_STOPPED_SPEED_THRESHOLD = 0.1

COLLECTOR_CONGESTION_STOPPED_VEHICLES = 5

COLLECTOR_CONGESTION_SPEED_THRESHOLD = 3

COLLECTOR_SLOW_TRAFFIC_SPEED_THRESHOLD = 8

COLLECTOR_ROUNDING_PRECISION = 2


# ======================================
# Reproducibility
# ======================================

RANDOM_SEED = 42


def seed_everything(seed=RANDOM_SEED):

    import random
    import numpy as np
    import torch

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================
# Dataset / Model Files
# ======================================

DEFAULT_LOCATION_ID = "location_2"

DATASET_FILENAME = (
    f"{DEFAULT_LOCATION_ID}_dataset.csv"
)

LSTM_MODEL_FILENAME = (
    "lstm_model.pth"
)

LSTM_SCALER_FILENAME = (
    "lstm_scalers.pkl"
)

DDQN_MODEL_FILENAME = (
    "ddqn_agent.pth"
)

TRAINING_HISTORY_FILENAME = (
    "training_history.csv"
)

METRICS_FILENAME = (
    "lstm_metrics.csv"
)

PREDICTIONS_FILENAME = (
    "prediction_results.csv"
)

LOSS_CURVE_FILENAME = (
    "loss_curve.png"
)

ACTUAL_VS_PREDICTED_FILENAME = (
    "actual_vs_predicted.png"
)

RESIDUAL_PLOT_FILENAME = (
    "residual_plot.png"
)

RESIDUAL_HISTOGRAM_FILENAME = (
    "residual_histogram.png"
)

TRAINING_REPORT_FILENAME = (
    "lstm_training_report.pdf"
)


# ======================================
# Traffic Event Types
# ======================================

TRAFFIC_EVENT_TYPES = {
    0: "Normal Traffic",
    1: "Accident",
    2: "Road Maintenance",
    3: "Emergency Vehicle",
    4: "Special Event",
}


# ======================================
# Real-Time Engine
# ======================================

REALTIME_STEP_DELAY_SECONDS = 0.1

REALTIME_HISTORY_LENGTH = 200


# ======================================
# API Configuration
# ======================================

API_HOST = "127.0.0.1"

API_PORT = 8000

API_BASE_URL = f"http://{API_HOST}:{API_PORT}"

WEBSOCKET_POLL_INTERVAL_SECONDS = 0.5

FRONTEND_REFRESH_INTERVAL_MS = 1000


# ======================================
# Dashboard
# ======================================

DASHBOARD_TITLE = (
    "AI Traffic Monitoring Dashboard"
)

MAX_GRAPH_POINTS = 200

AUTO_REFRESH_SECONDS = 1


# ======================================
# File Paths
# ======================================

DATASET_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    DATASET_FILENAME,
)

LSTM_MODEL_PATH = os.path.join(
    MODEL_DIR,
    LSTM_MODEL_FILENAME,
)

# ======================================
# Graph Transformer File Paths
# ======================================

GNN_MODEL_PATH = os.path.join(
    MODEL_DIR,
    GNN_MODEL_FILENAME,
)

GNN_SCALER_PATH = os.path.join(
    MODEL_DIR,
    GNN_SCALER_FILENAME,
)

LSTM_SCALER_PATH = os.path.join(
    MODEL_DIR,
    LSTM_SCALER_FILENAME,
)

DDQN_MODEL_PATH = os.path.join(
    MODEL_DIR,
    DDQN_MODEL_FILENAME,
)

TRAINING_HISTORY_PATH = os.path.join(
    LOGS_DIR,
    TRAINING_HISTORY_FILENAME,
)

METRICS_PATH = os.path.join(
    METRICS_DIR,
    METRICS_FILENAME,
)

GNN_METRICS_FILENAME = (
    "gnn_metrics.csv"
)

GNN_METRICS_PATH = os.path.join(
    METRICS_DIR,
    GNN_METRICS_FILENAME,
)

PREDICTIONS_PATH = os.path.join(
    PREDICTIONS_DIR,
    PREDICTIONS_FILENAME,
)

LOSS_CURVE_PATH = os.path.join(
    GRAPHS_DIR,
    LOSS_CURVE_FILENAME,
)

ACTUAL_VS_PREDICTED_PATH = os.path.join(
    GRAPHS_DIR,
    ACTUAL_VS_PREDICTED_FILENAME,
)

RESIDUAL_PLOT_PATH = os.path.join(
    GRAPHS_DIR,
    RESIDUAL_PLOT_FILENAME,
)

RESIDUAL_HISTOGRAM_PATH = os.path.join(
    GRAPHS_DIR,
    RESIDUAL_HISTOGRAM_FILENAME,
)

TRAINING_REPORT_PATH = os.path.join(
    REPORTS_DIR,
    TRAINING_REPORT_FILENAME,
)

GNN_CHECKPOINT_PATH = os.path.join(
    CHECKPOINTS_DIR,
    GNN_MODEL_FILENAME,
)

COMPARISON_REPORT_FILENAME = (
    "model_comparison.csv"
)

COMPARISON_REPORT_PATH = os.path.join(
    METRICS_DIR,
    COMPARISON_REPORT_FILENAME,
)


# ======================================
# Create Required Directories
# ======================================

for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    GRAPHS_DIR,
    LOGS_DIR,
    METRICS_DIR,
    REPORTS_DIR,
    PREDICTIONS_DIR,
    CHECKPOINTS_DIR,
    EVALUATION_RESULTS_DIR,
    EVALUATION_DIR,
]:
    os.makedirs(
        directory,
        exist_ok=True,
    )
"""
Project Configuration

Proactive Traffic Congestion Prediction
and Adaptive Traffic Signal Control System
"""

import os


## Project Information
# ======================================

PROJECT_NAME = (
    "Proactive Traffic Congestion Prediction"
)

AUTHOR = "Sudhanshu Ranjan"

VERSION = "1.0.0"


# Project Directories


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
    "osm",
    "osm.sumocfg",
)


# ======================================
# Simulation Settings
# ======================================

SIMULATION_STEP = 1

SIMULATION_TIME = 3600

# Lowered from 60 -> 15: each simulation run now emits ~4x more
# observation rows (one row every 15s instead of every 60s), directly
# addressing small-dataset issues (168 sequences from one run was too
# few — see Phase 1 accuracy diagnosis). Rates like arrival_rate /
# departure_rate are computed per-window internally, so this changes
# the time resolution of a "step", not any downstream code — nothing
# hardcodes the old value of 60 elsewhere (verified).
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
# RL Environment Features
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

LSTM_DROPOUT = 0.2

LSTM_BATCH_SIZE = 32

LSTM_EPOCHS = 50

LSTM_LEARNING_RATE = 0.001


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


# Reward Configuration

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


# ======================================
# Device Configuration
# ======================================

FORCE_CPU = False


# ======================================
# Output Formatting
# ======================================

PRINT_WIDTH = 60


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


## Dataset and Model Files


DEFAULT_LOCATION_ID = "location_1"

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


# ======================================
# Traffic Event Types
# (from architecture diagram — used to decode
# the raw traffic_event_type integer for display)
# ======================================

TRAFFIC_EVENT_TYPES = {
    0: "Normal Traffic",
    1: "Accident",
    2: "Road Maintenance",
    3: "Emergency Vehicle",
    4: "Special Event",
}


# ======================================
# Real-Time Engine (live dashboard)
# ======================================

# Seconds of real wall-clock delay per simulated second, so a live
# demo plays out visibly over time instead of finishing in a couple
# of seconds. 3600 simulated seconds * 0.1 = 360s (~6 min) real time.
REALTIME_STEP_DELAY_SECONDS = 0.1

# How many recent per-junction observations the frontend keeps for
# trend charts (separate from LOOKBACK, which is only for the LSTM).
REALTIME_HISTORY_LENGTH = 200


# ======================================
# API / Backend
# ======================================

API_HOST = "127.0.0.1"

API_PORT = 8000

API_BASE_URL = f"http://{API_HOST}:{API_PORT}"

# How often (seconds) the WebSocket checks for a new snapshot to push,
# and how often the Streamlit frontend polls the REST API as a fallback.
WEBSOCKET_POLL_INTERVAL_SECONDS = 0.5

FRONTEND_REFRESH_INTERVAL_MS = 1000


## Create Required Directories
#
for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    EVALUATION_DIR,
]:

    os.makedirs(
        directory,
        exist_ok=True,
    )
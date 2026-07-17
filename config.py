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

OBSERVATION_WINDOW = 60


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
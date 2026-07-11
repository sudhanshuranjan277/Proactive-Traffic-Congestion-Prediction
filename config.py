"""
Project Configuration
AI Traffic Control System
"""

import os

# ======================================
# Project Information
# ======================================

PROJECT_NAME = "Proactive Traffic Congestion Prediction"
AUTHOR = "Sudhanshu Ranjan"
VERSION = "1.0.0"

# ======================================
# Project Directories
# ======================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MAP_DIR = os.path.join(PROJECT_ROOT, "maps")

# ======================================
# SUMO Configuration
# ======================================

SUMO_BINARY = "sumo"

SUMO_CONFIG = os.path.join(
    PROJECT_ROOT,
    "maps",
    "osm",
    "osm.sumocfg"
)



# ======================================
# Simulation Settings
# ======================================

SIMULATION_STEP = 1
SIMULATION_TIME = 3600

# ======================================
# Reinforcement Learning
# ======================================

LEARNING_RATE = 0.001
GAMMA = 0.99
EPISODES = 100

# ======================================
# Deep Learning
# ======================================

LOOKBACK = 30
PREDICTION_HORIZON = 10

# ======================================
# Create Required Directories
# ======================================

for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
]:
    os.makedirs(directory, exist_ok=True)
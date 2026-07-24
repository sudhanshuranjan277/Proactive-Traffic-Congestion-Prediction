import os

# ===============================
# Project Paths
# ===============================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(ROOT_DIR, "datasets")
MODEL_DIR = os.path.join(ROOT_DIR, "saved_models")

# ===============================
# Dataset
# ===============================

DATASET_NAME = "METR-LA"

# ===============================
# Training
# ===============================

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# ===============================
# Device
# ===============================

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Random Seed
# ===============================

SEED = 42
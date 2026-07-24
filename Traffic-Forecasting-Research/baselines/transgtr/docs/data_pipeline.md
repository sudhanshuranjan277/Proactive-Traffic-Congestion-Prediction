# Data Pipeline Design

## Objective

This document describes the complete dataset generation pipeline for all baseline models.

The same processed dataset must be used across all baseline models to ensure a fair comparison.

---

# Input Dataset

Supported datasets

- METR-LA
- PEMS-BAY
- PEMSD7M

Future

- SUMO Generated Dataset

---

# Dataset Pipeline

                    Raw Dataset
                          │
                          ▼
                Dataset Validation
                          │
                          ▼
                 Missing Value Check
                          │
                          ▼
                 Time Index Validation
                          │
                          ▼
                 Feature Validation
                          │
                          ▼
                    Normalization
                          │
                          ▼
               Sliding Window Generator
                          │
                          ▼
                 Graph Construction
                          │
                          ▼
                  Dataset Splitter
                          │
                          ▼
                 Processed Dataset
                          │
                          ▼
                   PyTorch Dataset
                          │
                          ▼
                      DataLoader

---

# Stage 1

Dataset Validation

Checks

- Missing values
- Duplicate timestamps
- Invalid sensor ids
- Feature consistency
- Dataset dimensions

Output

Validated Dataset

---

# Stage 2

Preprocessing

Operations

- Missing value handling
- Normalization
- Standardization
- Feature ordering

Output

Clean Dataset

---

# Stage 3

Sliding Window Generation

Input

Historical observations

Output

Input Sequence

Target Sequence

Configurable

window_size

prediction_horizon

stride

---

# Stage 4

Graph Construction

Methods

Distance Graph

Adaptive Graph

Learned Graph

Output

Adjacency Matrix

Edge Index

Node Features

---

# Stage 5

Dataset Split

Training

Validation

Testing

No data leakage allowed.

---

# Stage 6

Processed Dataset

Saved format

.pt

or

.npz

This dataset will be used directly by the training pipeline.

No preprocessing should happen during training.

---

# Future Support

The pipeline should also support

SUMO Generated Dataset

without modifying any training code.

Only configuration should change.

---

# Design Rules

No hardcoded paths

No hardcoded dataset names

No hardcoded feature names

Everything configurable.

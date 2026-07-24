# TransGTR Architecture Design Document

**Paper**
Transferable Graph Structure Learning for Graph-based Traffic Forecasting Across Cities (KDD 2023)

---

# Objective

This project aims to reproduce the TransGTR model exactly as described in the research paper before proposing any improvements.

The implementation must be:

- Paper-driven
- Modular
- Configurable
- Reproducible
- Free from hardcoded values

---

# Overall Pipeline

```
Paper
        │
        ▼
Implementation Blueprint
        │
        ▼
Dataset Pipeline
        │
        ▼
Model Components
        │
        ▼
Training
        │
        ▼
Evaluation
        │
        ▼
Reproduced Results
```

---

# Dataset Pipeline

```
Raw Dataset
        │
        ▼
Validation
        │
        ▼
Preprocessing
        │
        ▼
Sliding Window Generation
        │
        ▼
Graph Construction
        │
        ▼
Processed Dataset
        │
        ▼
PyTorch Dataset
```

---

# Model Pipeline

```
Processed Dataset
        │
        ▼
TSFormer
        │
        ▼
Knowledge Distillation
        │
        ▼
Structure Generator (GTS)
        │
        ▼
Graph WaveNet
        │
        ▼
Prediction
```

---

# Training Pipeline

```
Dataset
        │
        ▼
DataLoader
        │
        ▼
Forward Pass
        │
        ▼
Loss Computation
        │
        ▼
Backward Pass
        │
        ▼
Optimizer
        │
        ▼
Checkpoint
```

---

# Evaluation Pipeline

```
Checkpoint
        │
        ▼
Inference
        │
        ▼
Prediction
        │
        ▼
Metrics

MAE

RMSE

MAPE
```

---

# Main Components

## Dataset Module

Responsible for:

- Dataset loading
- Validation
- Normalization
- Sliding window generation
- Graph construction

---

## TSFormer

Responsible for:

- Temporal feature extraction
- Node embedding generation

---

## Knowledge Distillation

Responsible for:

- Teacher model
- Student model
- Distillation loss

---

## Structure Generator

Responsible for:

- Graph structure learning
- Adjacency matrix generation

---

## Graph WaveNet

Responsible for:

- Spatial-temporal forecasting

---

## Trainer

Responsible for:

- Training loop
- Optimizer
- Scheduler
- Checkpoints

---

## Evaluator

Responsible for:

- Testing
- Metrics
- Visualization

---

# Configuration Policy

The project must not contain:


Everything must be loaded from configuration files.

---

# Development Policy

Every implementation will follow:

Paper

↓

Equation

↓

Algorithm

↓

Pseudo Code

↓

Python Implementation

No implementation should skip any step.

---

# Future Extension

After successful reproduction, the proposed model will introduce:

- Graph Transformer
- Adaptive Signal Control
- SUMO Integration
- Multi-Horizon Prediction

These improvements will remain outside the baseline implementation.
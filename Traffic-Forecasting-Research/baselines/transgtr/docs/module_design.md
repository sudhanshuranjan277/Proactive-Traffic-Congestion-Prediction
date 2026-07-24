# Module Design Document

## Objective

This document defines the responsibilities, inputs, outputs, dependencies, and implementation details of every module used in the TransGTR baseline implementation.

Every module must have a single responsibility and should be independent, reusable, and configurable.

No module should contain hardcoded values.

---

# Overall Module Architecture

```
                     Configuration
                           │
                           ▼
                     Dataset Pipeline
                           │
                           ▼
                    Feature Processing
                           │
                           ▼
                     Graph Generation
                           │
                           ▼
                  Model Architecture
                           │
                           ▼
                      Training Engine
                           │
                           ▼
                    Evaluation Engine
                           │
                           ▼
                     Experiment Report
```

---

# Module 1 : Dataset Loader

## Purpose

Load datasets from disk.

Supported datasets

- METR-LA
- PEMS-BAY
- PEMSD7M
- SUMO (Future)

### Input

Dataset Path

Configuration

### Output

Raw Dataset

### Responsibilities

- Read dataset
- Validate file existence
- Load metadata
- Return dataset object

### Dependencies

- pandas
- numpy

---

# Module 2 : Dataset Validator

## Purpose

Verify dataset quality.

### Checks

Missing Values

Duplicate Rows

Timestamp Order

Feature Count

Sensor Count

Invalid Records

### Output

Validated Dataset

---

# Module 3 : Data Preprocessor

## Purpose

Prepare dataset for training.

### Responsibilities

Normalization

Standardization

Missing Value Handling

Feature Ordering

Scaling

### Output

Clean Dataset

---

# Module 4 : Window Generator

## Purpose

Generate temporal sequences.

### Input

Clean Dataset

Window Size

Prediction Horizon

Stride

### Output

Input Tensor

Target Tensor

---

# Module 5 : Graph Builder

## Purpose

Construct graph representation.

### Methods

Distance Graph

Adaptive Graph

Learned Graph

### Output

Adjacency Matrix

Edge Index

Node Features

---

# Module 6 : TSFormer

## Paper Component

Temporal Spatial Transformer

### Purpose

Generate node embeddings.

### Input

Historical Traffic Sequence

### Output

Node Embeddings

### Responsibilities

Temporal Attention

Spatial Attention

Feature Encoding

---

# Module 7 : Knowledge Distillation

## Paper Component

Teacher Student Learning

### Purpose

Transfer knowledge from teacher model.

### Responsibilities

Teacher Network

Student Network

Distillation Loss

Feature Alignment

### Output

Student Embeddings

---

# Module 8 : Structure Generator

## Paper Component

Graph Structure Learning

### Purpose

Generate graph dynamically.

### Input

Node Embeddings

### Output

Adaptive Graph

Adjacency Matrix

---

# Module 9 : Graph WaveNet

## Paper Component

Forecasting Network

### Purpose

Predict future traffic.

### Input

Adaptive Graph

Traffic Features

### Output

Traffic Prediction

---

# Module 10 : Trainer

## Purpose

Train model.

### Responsibilities

Forward Pass

Loss Calculation

Backward Pass

Gradient Update

Checkpoint Saving

Learning Rate Scheduler

---

# Module 11 : Evaluator

## Purpose

Evaluate trained model.

### Metrics

MAE

RMSE

MAPE

Inference Time

Memory Usage

---

# Module 12 : Experiment Manager

## Purpose

Manage experiments.

### Responsibilities

Run Tracking

Configuration Logging

Metric Logging

Checkpoint Management

Result Export

---

# Module Dependencies

```
Dataset Loader
        │
        ▼
Dataset Validator
        │
        ▼
Preprocessor
        │
        ▼
Window Generator
        │
        ▼
Graph Builder
        │
        ▼
TSFormer
        │
        ▼
Knowledge Distillation
        │
        ▼
Structure Generator
        │
        ▼
Graph WaveNet
        │
        ▼
Trainer
        │
        ▼
Evaluator
```

---

# Configuration Policy

Every module must receive configuration from configuration files.

No module should contain

❌ Hardcoded paths

❌ Hardcoded hyperparameters

❌ Hardcoded dataset names

❌ Hardcoded feature names

---

# Error Handling Policy

Every module must validate its input.

Raise descriptive exceptions.

Never silently ignore errors.

---

# Logging Policy

Every module should log

Input Shape

Output Shape

Execution Time

Warnings

Errors

No print() statements should be used.

---

# Testing Policy

Each module must support independent testing.

Unit tests

Integration tests

End-to-end tests

---

# Documentation Policy

Every Python file must include

- Module Purpose
- Paper Section
- Related Equations
- Input
- Output
- Dependencies

before implementation begins.

---

# Future Extension

The architecture should support

- New datasets
- New graph generators
- New forecasting models
- New evaluation metrics

without modifying existing modules.
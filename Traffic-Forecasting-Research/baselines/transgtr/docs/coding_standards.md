# Coding Standards

## Objective

This document defines the coding standards and development practices followed throughout the TransGTR implementation.

The goal is to ensure that the codebase remains:

- Clean
- Readable
- Maintainable
- Modular
- Reproducible
- Research-friendly

---

# General Principles

Every source file must follow:

- Single Responsibility Principle (SRP)
- Separation of Concerns
- Modular Design
- Configurable Implementation
- Reusable Components

---

# Python Version

Python >= 3.11

---

# Code Style

The project follows

- PEP 8
- PEP 257 (Docstrings)
- Black Formatting
- isort Import Ordering

Maximum Line Length

88 Characters

Indentation

4 Spaces

Encoding

UTF-8

---

# Project Structure

Every module should have a clear purpose.

Example

models/

datasets/

training/

evaluation/

utils/

configs/

tests/

docs/

No business logic should be written inside utility modules.

---

# Naming Convention

## Files

snake_case.py

Example

dataset_loader.py

graph_builder.py

knowledge_distillation.py

graph_wavenet.py

---

## Classes

PascalCase

Example

DatasetLoader

GraphBuilder

GraphWaveNet

Trainer

Evaluator

---

## Functions

snake_case

Example

load_dataset()

build_graph()

train_epoch()

evaluate_model()

---

## Variables

snake_case

Good

node_features

edge_index

window_size

learning_rate

Bad

NodeFeatures

LearningRate

X

Y

---

## Constants

UPPER_CASE

Example

DEFAULT_WINDOW_SIZE

MAX_EPOCHS

DEVICE_CPU

---

# Type Hints

Every public function must include type hints.

Example

```python
def load_dataset(path: str) -> pd.DataFrame:
    ...
```

---

# Docstring Standard

Every module starts with

```python
"""
Module:
dataset_loader.py

Purpose:
Loads traffic datasets.

Paper Section:
Section 4

Related Equations:
N/A

Author:
Research Team
"""
```

Every function

```python
def build_graph(...):
    """
    Build adjacency matrix.

    Parameters
    ----------
    ...

    Returns
    -------
    ...

    Raises
    ------
    ...
    """
```

---

# Import Order

Standard Library

↓

Third Party

↓

Project Imports

Example

```python
import os
import logging

import numpy as np
import pandas as pd
import torch

from utils.logger import get_logger
```

No wildcard imports.

Never

```python
from torch import *
```

---

# Configuration Policy

No hardcoded values.

Everything should come from

configs/

Examples

Dataset Path

Learning Rate

Batch Size

Epochs

Window Size

Prediction Horizon

Device

Random Seed

---

# Logging Standard

Never use

```python
print()
```

Always use

```python
logger.info()

logger.warning()

logger.error()

logger.exception()
```

Each module should log

- Start
- Completion
- Execution Time
- Errors
- Warnings

---

# Error Handling

Do not silently ignore exceptions.

Bad

```python
try:
    ...
except:
    pass
```

Good

```python
try:
    ...
except FileNotFoundError as e:
    logger.exception(e)
    raise
```

---

# Comments

Avoid obvious comments.

Bad

```python
x = x + 1
# increment x
```

Good

Explain

- why

not

- what

---

# Function Design

Each function should perform

One task only.

Recommended

20–40 lines

Maximum

60 lines

Longer functions should be split.

---

# Class Design

Each class should have one responsibility.

Avoid God Classes.

---

# Dependency Injection

Pass dependencies through constructors.

Avoid global variables.

---

# Randomness

Every experiment must be reproducible.

Random Seed

NumPy

PyTorch

Python Random

should all be initialized.

---

# Testing Standard

Each module must have

Unit Test

Integration Test

End-to-End Test

---

# Performance

Avoid unnecessary loops.

Prefer

NumPy

PyTorch

Vectorization

over Python loops.

---

# Security

Never hardcode

Passwords

API Keys

Tokens

Absolute Paths

Secrets

---

# Documentation

Every module must document

Purpose

Input

Output

Dependencies

Configuration

Related Paper Section

---

# Git Commit Style

Use meaningful commit messages.

Good

feat: implement TSFormer encoder

fix: correct adjacency matrix normalization

docs: update architecture document

refactor: simplify graph builder

Avoid

update

changes

fixed bug

---

# Branch Strategy

main

Stable implementation

develop

Active development

feature/<module-name>

Individual features

Example

feature/tsformer

feature/graph-builder

feature/trainer

---

# Code Review Checklist

Before merging

✓ Follows PEP8

✓ No hardcoded values

✓ Uses configuration

✓ Logging added

✓ Error handling implemented

✓ Type hints added

✓ Docstrings added

✓ Unit tests passed

✓ Integration tests passed

✓ Documentation updated

---

# Research Principles

Implementation order must always follow

Paper

↓

Equations

↓

Algorithm

↓

Pseudo Code

↓

Implementation

↓

Testing

↓

Documentation

Code should never be written directly without understanding the corresponding paper section.

---

# Final Goal

Every source file should be understandable without referring to external explanations.

A new researcher should be able to understand the implementation using only

- Documentation
- Code
- Comments
- Paper Mapping
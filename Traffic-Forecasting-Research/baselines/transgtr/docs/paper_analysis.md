# Appendix Extraction

**Paper Title:** Transferable Graph Structure Learning for Graph-based Traffic Forecasting Across Cities (KDD 2023)

**Purpose:**
This document records all implementation-specific information extracted from the paper and appendix. It serves as the single source of truth while reproducing the baseline model.

---

# 1. Dataset Information

## Supported Datasets

| Dataset | Available | Notes |
|----------|-----------|-------|
| METR-LA | ☐ | |
| PEMS-BAY | ☐ | |
| PEMSD7M | ☐ | |
| Other | ☐ | |

---

## Dataset Statistics

| Parameter | Value | Source |
|-----------|-------|--------|
| Number of Sensors | | |
| Number of Nodes | | |
| Number of Features | | |
| Sampling Interval | | |
| Total Samples | | |
| Prediction Horizon | | |
| Input Sequence Length | | |

---

## Dataset Split

| Split | Value |
|-------|-------|
| Train | |
| Validation | |
| Test | |

---

# 2. Data Preprocessing

## Missing Value Handling

-

## Normalization

-

## Window Generation

| Parameter | Value |
|-----------|-------|
| Input Window | |
| Forecast Window | |
| Stride | |

---

# 3. TSFormer

## Architecture

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | |
| Hidden Dimension | |
| Number of Layers | |
| Number of Attention Heads | |
| Dropout | |
| Positional Encoding | |

---

## Input / Output

| Parameter | Value |
|-----------|-------|
| Input Shape | |
| Output Shape | |

---

# 4. Knowledge Distillation

| Parameter | Value |
|-----------|-------|
| Teacher Model | |
| Student Model | |
| Temperature | |
| Alpha | |
| Beta | |
| Distillation Loss | |

---

# 5. Graph Structure Generator (GTS)

## Graph Learning

| Parameter | Value |
|-----------|-------|
| Adaptive Graph | |
| Graph Learning Method | |
| Self Loop | |
| Graph Normalization | |

---

# 6. GraphWaveNet

| Parameter | Value |
|-----------|-------|
| Residual Channels | |
| Dilation Channels | |
| Skip Channels | |
| End Channels | |
| Kernel Size | |
| Number of Blocks | |
| Layers per Block | |
| Dropout | |

---

# 7. Loss Function

| Parameter | Value |
|-----------|-------|
| Primary Loss | |
| Auxiliary Loss | |
| Total Loss Equation | |

---

# 8. Optimizer

| Parameter | Value |
|-----------|-------|
| Optimizer | |
| Learning Rate | |
| Weight Decay | |
| Momentum | |
| Scheduler | |

---

# 9. Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | |
| Batch Size | |
| Early Stopping | |
| Gradient Clipping | |
| Random Seed | |

---

# 10. Evaluation

## Metrics

- MAE
- RMSE
- MAPE

---

## Prediction Horizons

-

---

# 11. Hardware

| Parameter | Value |
|-----------|-------|
| GPU | |
| CPU | |
| CUDA Version | |
| Training Time | |

---

# 12. Reproducibility Notes

## Important Implementation Details

-

## Paper Assumptions

-

## Missing Information

-

---

# 13. Configuration Mapping

| Paper Section | Config File |
|---------------|-------------|
| Dataset | dataset.yaml |
| Model | model.yaml |
| Training | training.yaml |
| Evaluation | evaluation.yaml |
| Logging | logging.yaml |

---

# 14. Verification Checklist

## Dataset

- [ ] Dataset downloaded
- [ ] Dataset verified
- [ ] Dataset preprocessing verified

## Model

- [ ] TSFormer verified
- [ ] Knowledge Distillation verified
- [ ] GTS verified
- [ ] GraphWaveNet verified

## Training

- [ ] Hyperparameters verified
- [ ] Optimizer verified
- [ ] Loss verified

## Evaluation

- [ ] MAE verified
- [ ] RMSE verified
- [ ] MAPE verified

---

# Status

| Section | Completed |
|----------|-----------|
| Dataset | ☐ |
| TSFormer | ☐ |
| Knowledge Distillation | ☐ |
| GTS | ☐ |
| GraphWaveNet | ☐ |
| Training | ☐ |
| Evaluation | ☐ |

---

**Last Updated:** YYYY-MM-DD

**Updated By:** __________________
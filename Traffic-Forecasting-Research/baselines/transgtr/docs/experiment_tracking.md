# Experiment Tracking

## Objective

This document records every experiment conducted during the implementation and evaluation of the TransGTR baseline model.

The objective is to ensure:

- Reproducibility
- Traceability
- Fair comparison
- Easy analysis
- Accurate reporting

Every experiment must be recorded before and after execution.

---

# Experiment Naming Convention

Experiment ID Format

EXP-001

EXP-002

EXP-003

...

---

# Experiment Status

Possible Status

- Planned
- Running
- Completed
- Failed
- Repeated
- Archived

---

# Experiment Template

## Experiment Information

| Field | Value |
|--------|-------|
| Experiment ID | EXP-001 |
| Date | YYYY-MM-DD |
| Researcher | Sudhanshu Ranjan |
| Model | TransGTR |
| Dataset | METR-LA |
| Status | Planned |

---

## Objective

Example

Reproduce the baseline results using the original paper hyperparameters.

---

## Paper Reference

Section

Appendix

Equation

Algorithm

Figure

---

## Configuration

Dataset

Window Size

Prediction Horizon

Batch Size

Learning Rate

Epochs

Optimizer

Scheduler

Random Seed

Device

Configuration File Used

---

## Dataset Details

Dataset Name

Number of Sensors

Number of Samples

Train Split

Validation Split

Test Split

Normalization Method

---

## Model Configuration

TSFormer

Knowledge Distillation

Structure Generator

Graph WaveNet

Number of Parameters

Model Size

---

## Training Details

Training Time

GPU

CPU

RAM

Mixed Precision

Gradient Clipping

Checkpoint Interval

---

## Results

| Metric | Value |
|---------|-------|
| MAE | |
| RMSE | |
| MAPE | |
| Training Loss | |
| Validation Loss | |
| Test Loss | |

---

## Comparison with Paper

| Metric | Paper | Our Result | Difference |
|---------|--------|------------|------------|
| MAE | | | |
| RMSE | | | |
| MAPE | | | |

---

## Observations

Example

Training converged after 48 epochs.

Validation loss stabilized.

No overfitting observed.

---

## Problems Encountered

Example

CUDA memory overflow

Gradient explosion

Dataset mismatch

Incorrect adjacency matrix

---

## Solution Applied

Describe the solution used.

---

## Files Generated

Checkpoint

Logs

Plots

Configuration

Evaluation Report

---

## Final Conclusion

Summarize the experiment outcome.

Example

Baseline successfully reproduced.

or

Requires further investigation.

---

# Experiment Summary Table

| ID | Model | Dataset | Status | MAE | RMSE | MAPE | Notes |
|----|--------|----------|--------|-----|------|------|-------|
| EXP-001 | TransGTR | METR-LA | Planned | - | - | - | Baseline |
| EXP-002 | TransGTR | PEMS-BAY | Planned | - | - | - | Baseline |

---

# Reproducibility Checklist

Before marking an experiment as completed:

- [ ] Configuration saved
- [ ] Random seed fixed
- [ ] Logs generated
- [ ] Checkpoint saved
- [ ] Metrics verified
- [ ] Results compared with paper
- [ ] Documentation updated

---

# Experiment Repository

Each experiment should generate the following artifacts:

experiments/

├── configs/

├── logs/

├── checkpoints/

├── metrics/

├── plots/

└── reports/

---

# Version History

| Version | Date | Changes |
|----------|------|----------|
| 1.0 | YYYY-MM-DD | Initial experiment template |

---

# Notes

- Every experiment must have a unique Experiment ID.
- Never overwrite previous experiment results.
- Keep all configuration files used for each experiment.
- Maintain complete reproducibility.
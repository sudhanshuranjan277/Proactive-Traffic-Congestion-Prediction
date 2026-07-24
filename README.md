# Proactive Traffic Congestion Prediction and Adaptive Signal Control

AI-powered proactive traffic congestion prediction and adaptive traffic signal control using SUMO, Python, Reinforcement Learning, and Deep Learning.

## Project Overview

This repository implements a research-oriented traffic control framework that:
- collects traffic features from SUMO in real time;
- aggregates data into fixed junction observations;
- trains an LSTM model to predict future traffic targets;
- uses predicted future traffic information in a DDQN agent for signal control.

## Key Modules

- `environment/sumo_env.py`: SUMO environment management and TraCI connection.
- `integration/collector.py`: dynamic junction discovery and traffic feature collection.
- `integration/pipeline.py`: fixed schema validation and time-window aggregation.
- `prediction/preprocessing.py`: dataset loading, validation, chronological sequence creation, and scaling.
- `prediction/lstm.py`: LSTM model architecture for traffic prediction.
- `prediction/predictor.py`: loads saved model and scalers for inference.
- `rl/model.py`: DDQN neural network model.
- `rl/memory.py`: replay buffer for reinforcement learning.
- `rl/ddqn.py`: Double DQN learning algorithm.
- `rl/agent.py`: traffic signal agent wrapper.
- `rl/environment.py`: gym-like traffic environment integrating SUMO, predictor, and controller.
- `integration/controller.py`: safe action execution for SUMO traffic lights.
- `evaluation/metrics.py`: dataset metric summaries.
- `visualization/plot_metrics.py`: example plotting of dataset summaries.

## Installation

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

1. Generate the dataset from SUMO:

```bash
python scripts/create_dataset.py
```

2. Train the LSTM predictor:

```bash
python scripts/train_model.py
```

3. Train the DDQN traffic agent:

```bash
python scripts/train_ddqn.py
```

4. Evaluate a processed dataset:

```bash
python scripts/evaluate_dataset.py
```

5. Plot dataset metrics:

```bash
python visualization/plot_metrics.py
```

## Testing

Run the unit test suite for independent modules:

```bash
python -m unittest discover -s tests
```

## Design Notes

- Configuration values are centralized in `config.py`.
- The collector uses a fixed feature schema across all junctions.
- Prediction preprocessing keeps location and junction sequences isolated.
- The RL environment is gym-style and returns `(state, reward, done, info)`.
- The predictor supports multi-target future traffic predictions.

## Assumptions

- The LSTM currently trains on historical junction observations and predicts multiple traffic targets.
- The DDQN agent uses predicted queue length and saturation heuristics in its state.
- Safe signal transitions are handled by the controller, but further SUMO-specific phase constraints may be added later.

This project combines:
- OpenStreetMap and SUMO for realistic traffic simulation.
- A dynamic traffic feature collector and observation pipeline.
- An LSTM predictor for future queue length.
- A Double Deep Q-Network (DDQN) agent for adaptive traffic signal control.

New components added:
- `rl/memory.py`, `rl/ddqn.py`, `rl/reward.py`, `rl/environment.py` for reinforcement learning.
- `integration/controller.py` for safe SUMO signal actions.
- `scripts/train_ddqn.py` for training the DDQN agent using the trained queue predictor.

To train the DDQN agent, first ensure the LSTM model is trained and available in `models/lstm_model.pth` and `models/lstm_scalers.pkl`.
Then run:

```bash
python scripts/train_ddqn.py
```

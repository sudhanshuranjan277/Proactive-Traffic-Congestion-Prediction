"""
DDQN Traffic Signal Control Training Script

This script demonstrates how to train a DDQN agent
using a trained LSTM queue predictor and the SUMO
traffic simulation environment.
"""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    MODEL_DIR,
    LOOKBACK,
    OBSERVATION_WINDOW,
    RL_EPSILON_END,
    RL_EPSILON_START,
    RL_GAMMA,
    RL_LEARNING_RATE,
    RL_MAX_EPISODES,
    RL_MEMORY_CAPACITY,
    RL_NUM_ACTIONS,
    RL_TARGET_UPDATE_FREQUENCY,
    RL_EXTENSION_SECONDS,
    LSTM_MODEL_FILENAME,
    LSTM_SCALER_FILENAME,
    DDQN_MODEL_FILENAME,
    DEFAULT_LOCATION_ID,
)
from environment.sumo_env import SumoEnvironment
from integration.collector import TrafficCollector
from rl.agent import TrafficSignalAgent
from rl.environment import JunctionTrafficEnvironment
from prediction.predictor import TrafficPredictor


MODEL_PATH = os.path.join(
    MODEL_DIR,
    LSTM_MODEL_FILENAME,
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    LSTM_SCALER_FILENAME,
)


def select_exploration_rate(step, max_steps):
    if max_steps <= 0:
        return RL_EPSILON_END
    return float(
        max(
            RL_EPSILON_END,
            RL_EPSILON_START
            - (RL_EPSILON_START - RL_EPSILON_END)
            * (step / float(max_steps))
        )
    )


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"LSTM model file not found: {MODEL_PATH}"
        )

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler file not found: {SCALER_PATH}"
        )

    env = SumoEnvironment()
    collector = TrafficCollector()
    predictor = TrafficPredictor(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
    )

    print("=" * 60)
    print("DDQN Traffic Signal Training")
    print("=" * 60)
    env.show_configuration()

    if not env.connect():
        raise RuntimeError("SUMO connection failed.")

    try:
        if not collector.initialized:
            collector.initialize_junctions()

        junction_ids = list(collector.junction_lanes.keys())

        if not junction_ids:
            raise RuntimeError(
                "No signalized junctions discovered in SUMO network."
            )

        junction_id = junction_ids[0]
        print(f"Training on junction: {junction_id}")

        training_environment = JunctionTrafficEnvironment(
            sumo_env=env,
            collector=collector,
            predictor=predictor,
            junction_id=junction_id,
            location_id=DEFAULT_LOCATION_ID,
            lookback=LOOKBACK,
            observation_window=OBSERVATION_WINDOW,
            max_steps=RL_MAX_EPISODES,
            extension_seconds=RL_EXTENSION_SECONDS,
        )

        state = training_environment.reset()
        state_dim = state.shape[0]
        action_dim = RL_NUM_ACTIONS

        agent = TrafficSignalAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            memory_capacity=RL_MEMORY_CAPACITY,
            gamma=RL_GAMMA,
            learning_rate=RL_LEARNING_RATE,
            target_update_frequency=RL_TARGET_UPDATE_FREQUENCY,
        )

        batch_size = RL_BATCH_SIZE
        num_training_steps = RL_MAX_EPISODES
        total_reward = 0.0

        for step in range(num_training_steps):
            epsilon = select_exploration_rate(
                step,
                num_training_steps,
            )

            action = agent.select_action(
                state,
                epsilon,
            )

            next_state, reward, done, info = training_environment.step(
                action
            )

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
            )

            loss = agent.update(batch_size)

            total_reward += reward
            state = next_state

            print(
                f"Step {step + 1}/{num_training_steps} | "
                f"Action: {action} | "
                f"Reward: {reward:.4f} | "
                f"Loss: {loss:.6f} | "
                f"Epsilon: {epsilon:.3f} | "
                f"Time: {info['simulation_time']:.0f}s"
            )

            if done:
                print("Episode finished.")
                break

        model_save_path = os.path.join(
            MODEL_DIR,
            DDQN_MODEL_FILENAME,
        )

        agent.save(model_save_path)
        print(f"Trained DDQN agent saved to: {model_save_path}")
        print(f"Total reward collected: {total_reward:.4f}")

    finally:
        env.disconnect()


if __name__ == "__main__":
    main()

"""
Episodic DDQN Traffic Signal Control Training Script

Trains a Double Deep Q-Network traffic signal agent
using proactive LSTM traffic predictions and fresh
SUMO simulation episodes.

Training Architecture:

Fresh SUMO Episode
    ->
Traffic Collector
    ->
Traffic Pipeline
    ->
LSTM Historical Warm-Up
    ->
Proactive DDQN State
    ->
Traffic Signal Action
    ->
SUMO Traffic Transition
    ->
Reward
    ->
Experience Replay
    ->
Double DQN Learning
    ->
Fresh SUMO Episode

The DDQN agent and replay memory persist across
episodes while SUMO and traffic collection state
are restarted for every episode.
"""

import os
import sys


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )


from config import (
    MODEL_DIR,
    LOOKBACK,
    OBSERVATION_WINDOW,
    RL_BATCH_SIZE,
    RL_EPSILON_END,
    RL_EPSILON_START,
    RL_GAMMA,
    RL_LEARNING_RATE,
    RL_MAX_EPISODES,
    RL_MAX_STEPS_PER_EPISODE,
    RL_MEMORY_CAPACITY,
    RL_NUM_ACTIONS,
    RL_TARGET_UPDATE_FREQUENCY,
    RL_EXTENSION_SECONDS,
    LSTM_MODEL_FILENAME,
    LSTM_SCALER_FILENAME,
    DDQN_MODEL_FILENAME,
    DEFAULT_LOCATION_ID,
)

from environment.sumo_env import (
    SumoEnvironment,
)

from integration.collector import (
    TrafficCollector,
)

from prediction.predictor import (
    TrafficPredictor,
)

from rl.agent import (
    TrafficSignalAgent,
)

from rl.environment import (
    JunctionTrafficEnvironment,
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    LSTM_MODEL_FILENAME,
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    LSTM_SCALER_FILENAME,
)

DDQN_MODEL_PATH = os.path.join(
    MODEL_DIR,
    DDQN_MODEL_FILENAME,
)


def select_exploration_rate(
    global_step,
    total_training_steps,
):
    """
    Calculate linearly decaying epsilon
    across the complete DDQN training run.
    """

    if total_training_steps <= 1:
        return float(
            RL_EPSILON_END
        )

    progress = (
        global_step
        / float(
            total_training_steps - 1
        )
    )

    progress = min(
        1.0,
        max(
            0.0,
            progress,
        ),
    )

    epsilon = (
        RL_EPSILON_START
        - (
            RL_EPSILON_START
            - RL_EPSILON_END
        )
        * progress
    )

    return float(
        max(
            RL_EPSILON_END,
            epsilon,
        )
    )


def validate_training_files():
    """
    Validate required trained LSTM artifacts.
    """

    if not os.path.exists(
        MODEL_PATH
    ):
        raise FileNotFoundError(
            "LSTM model file not found: "
            f"{MODEL_PATH}"
        )

    if not os.path.exists(
        SCALER_PATH
    ):
        raise FileNotFoundError(
            "LSTM scaler file not found: "
            f"{SCALER_PATH}"
        )


def validate_training_configuration():
    """
    Validate DDQN training configuration.
    """

    if RL_MAX_EPISODES <= 0:
        raise ValueError(
            "RL_MAX_EPISODES must be "
            "greater than zero."
        )

    if RL_MAX_STEPS_PER_EPISODE <= 0:
        raise ValueError(
            "RL_MAX_STEPS_PER_EPISODE "
            "must be greater than zero."
        )

    if RL_BATCH_SIZE <= 0:
        raise ValueError(
            "RL_BATCH_SIZE must be "
            "greater than zero."
        )

    if RL_NUM_ACTIONS <= 0:
        raise ValueError(
            "RL_NUM_ACTIONS must be "
            "greater than zero."
        )


def create_episode_environment(
    sumo_env,
    predictor,
    junction_id,
):
    """
    Create fresh traffic collection and
    RL environment state for one episode.
    """

    collector = TrafficCollector()

    collector.initialize_junctions()

    if junction_id not in (
        collector.junction_lanes
    ):
        raise RuntimeError(
            "Training junction was not "
            "discovered after SUMO restart: "
            f"{junction_id}"
        )

    training_environment = (
        JunctionTrafficEnvironment(
            sumo_env=sumo_env,
            collector=collector,
            predictor=predictor,
            junction_id=junction_id,
            location_id=DEFAULT_LOCATION_ID,
            lookback=LOOKBACK,
            observation_window=(
                OBSERVATION_WINDOW
            ),
            max_steps=(
                RL_MAX_STEPS_PER_EPISODE
            ),
            extension_seconds=(
                RL_EXTENSION_SECONDS
            ),
        )
    )

    return training_environment


def main():

    validate_training_files()

    validate_training_configuration()

    num_episodes = int(
        RL_MAX_EPISODES
    )

    max_steps_per_episode = int(
        RL_MAX_STEPS_PER_EPISODE
    )

    batch_size = int(
        RL_BATCH_SIZE
    )

    total_training_steps = (
        num_episodes
        * max_steps_per_episode
    )

    sumo_env = SumoEnvironment()

    predictor = TrafficPredictor(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
    )

    print(
        "=" * 60
    )

    print(
        "Episodic DDQN Traffic Signal Training"
    )

    print(
        "=" * 60
    )

    print(
        "Training Mode     : "
        "Fresh SUMO Episodes"
    )

    print(
        f"Episodes          : "
        f"{num_episodes}"
    )

    print(
        f"Steps Per Episode : "
        f"{max_steps_per_episode}"
    )

    print(
        f"Maximum RL Steps  : "
        f"{total_training_steps}"
    )

    print(
        f"Replay Batch Size : "
        f"{batch_size}"
    )

    print(
        f"LSTM Lookback     : "
        f"{LOOKBACK}"
    )

    print(
        f"Observation Window: "
        f"{OBSERVATION_WINDOW}s"
    )

    print(
        f"Action Count      : "
        f"{RL_NUM_ACTIONS}"
    )

    print(
        "=" * 60
    )

    sumo_env.show_configuration()

    if not sumo_env.connect():
        raise RuntimeError(
            "SUMO connection failed."
        )

    try:

        initial_collector = (
            TrafficCollector()
        )

        initial_collector.initialize_junctions()

        junction_ids = list(
            initial_collector
            .junction_lanes
            .keys()
        )

        if not junction_ids:
            raise RuntimeError(
                "No signalized junctions "
                "discovered in SUMO network."
            )

        junction_id = junction_ids[0]

        print(
            "\nTraining Junction:"
        )

        print(
            junction_id
        )

        print(
            "\nCreating initial "
            "training environment..."
        )

        training_environment = (
            create_episode_environment(
                sumo_env=sumo_env,
                predictor=predictor,
                junction_id=junction_id,
            )
        )

        print(
            "\nInitializing LSTM "
            "traffic history..."
        )

        state = (
            training_environment.reset()
        )

        state_dim = int(
            state.shape[0]
        )

        action_dim = int(
            RL_NUM_ACTIONS
        )

        if state_dim != 17:
            raise ValueError(
                "Unexpected DDQN state dimension. "
                f"Expected 17, received "
                f"{state_dim}."
            )

        if action_dim != 3:
            raise ValueError(
                "Unexpected DDQN action dimension. "
                f"Expected 3, received "
                f"{action_dim}."
            )

        print(
            "\nEnvironment Validation"
        )

        print(
            f"State Dimension  : "
            f"{state_dim}"
        )

        print(
            f"Action Dimension : "
            f"{action_dim}"
        )

        print(
            f"Warm-Up End Time : "
            f"{training_environment.previous_row['simulation_time']:.0f}s"
        )

        agent = TrafficSignalAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            memory_capacity=(
                RL_MEMORY_CAPACITY
            ),
            gamma=RL_GAMMA,
            learning_rate=(
                RL_LEARNING_RATE
            ),
            target_update_frequency=(
                RL_TARGET_UPDATE_FREQUENCY
            ),
        )

        global_step = 0

        total_reward = 0.0

        total_loss = 0.0

        learning_updates = 0

        action_counts = {
            action: 0
            for action in range(
                action_dim
            )
        }

        episode_rewards = []

        print(
            "\nStarting DDQN learning...\n"
        )

        for episode in range(
            num_episodes
        ):

            if episode > 0:

                sumo_env.restart()

                training_environment = (
                    create_episode_environment(
                        sumo_env=sumo_env,
                        predictor=predictor,
                        junction_id=junction_id,
                    )
                )

                state = (
                    training_environment.reset()
                )

            episode_reward = 0.0

            episode_loss = 0.0

            episode_updates = 0

            completed_episode_steps = 0

            print(
                "\n"
                + "-" * 60
            )

            print(
                f"Episode "
                f"{episode + 1}/"
                f"{num_episodes}"
            )

            print(
                f"Control Start Time: "
                f"{training_environment.previous_row['simulation_time']:.0f}s"
            )

            print(
                "-" * 60
            )

            for episode_step in range(
                max_steps_per_episode
            ):

                epsilon = (
                    select_exploration_rate(
                        global_step,
                        total_training_steps,
                    )
                )

                action = (
                    agent.select_action(
                        state,
                        epsilon,
                    )
                )

                (
                    next_state,
                    reward,
                    done,
                    info,
                ) = (
                    training_environment.step(
                        action
                    )
                )

                agent.remember(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                loss = agent.update(
                    batch_size
                )

                reward = float(
                    reward
                )

                loss = float(
                    loss
                )

                episode_reward += reward

                total_reward += reward

                if loss > 0.0:

                    episode_loss += loss

                    total_loss += loss

                    episode_updates += 1

                    learning_updates += 1

                action_counts[
                    action
                ] += 1

                completed_episode_steps += 1

                global_step += 1

                state = next_state

                print(
                    f"Episode "
                    f"{episode + 1:03d} | "
                    f"Step "
                    f"{episode_step + 1:02d}/"
                    f"{max_steps_per_episode:02d} | "
                    f"Global "
                    f"{global_step:04d}/"
                    f"{total_training_steps:04d} | "
                    f"Action: {action} | "
                    f"Executed: "
                    f"{info['action_valid']} | "
                    f"Reward: "
                    f"{reward:.4f} | "
                    f"Loss: "
                    f"{loss:.6f} | "
                    f"Epsilon: "
                    f"{epsilon:.3f} | "
                    f"Memory: "
                    f"{len(agent.memory)} | "
                    f"Time: "
                    f"{info['simulation_time']:.0f}s"
                )

                if done:
                    break

            episode_rewards.append(
                episode_reward
            )

            if episode_updates > 0:

                average_episode_loss = (
                    episode_loss
                    / episode_updates
                )

            else:

                average_episode_loss = 0.0

            print(
                "\nEpisode Summary"
            )

            print(
                f"Episode Reward : "
                f"{episode_reward:.4f}"
            )

            print(
                f"Episode Steps  : "
                f"{completed_episode_steps}"
            )

            print(
                f"Learning Updates: "
                f"{episode_updates}"
            )

            print(
                f"Average Loss   : "
                f"{average_episode_loss:.6f}"
            )

            print(
                f"Replay Samples : "
                f"{len(agent.memory)}"
            )

        agent.save(
            DDQN_MODEL_PATH
        )

        if global_step > 0:

            average_reward = (
                total_reward
                / global_step
            )

        else:

            average_reward = 0.0

        if learning_updates > 0:

            average_loss = (
                total_loss
                / learning_updates
            )

        else:

            average_loss = 0.0

        print(
            "\n"
            + "=" * 60
        )

        print(
            "DDQN Training Completed"
        )

        print(
            "=" * 60
        )

        print(
            f"Completed Episodes : "
            f"{len(episode_rewards)}"
        )

        print(
            f"Completed RL Steps : "
            f"{global_step}"
        )

        print(
            f"Replay Samples     : "
            f"{len(agent.memory)}"
        )

        print(
            f"Learning Updates   : "
            f"{learning_updates}"
        )

        print(
            f"Total Reward       : "
            f"{total_reward:.4f}"
        )

        print(
            f"Average Reward     : "
            f"{average_reward:.4f}"
        )

        print(
            f"Average DDQN Loss  : "
            f"{average_loss:.6f}"
        )

        print(
            "\nAction Distribution:"
        )

        for (
            action,
            count,
        ) in action_counts.items():

            print(
                f"Action {action}: "
                f"{count}"
            )

        print(
            "\nDDQN Model File:"
        )

        print(
            DDQN_MODEL_PATH
        )

    finally:

        sumo_env.disconnect()


if __name__ == "__main__":
    main()